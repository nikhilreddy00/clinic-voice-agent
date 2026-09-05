"""Phase 16 — the OTel span tree, and the PHI boundary on it.

Two claims, tested separately because they fail for different reasons.

THE SHAPE. `spans_from_events` is pure, so the tree it builds from a hand-written stream is
exactly assertable — no SDK, no collector, no clock. The boundaries have to match the ones
every latency number in this project already uses (`telemetry`), or the APM and the trace
viewer will disagree about what a "turn" is and nobody will know which to believe.

THE BOUNDARY. Spans leave the building; `logs/traces/` does not. Every recorded call in the
Tier-1 corpus is replayed through the mapper and every exported attribute value is checked
against the names, dates of birth, and utterances those calls contain. That is a stronger test
than a hand-written one — the corpus is real speech from real calls, so it contains the phrasing
nobody would think to write into a fixture.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from clinic_agent.core import events as ev
from clinic_agent.core.otel import (
    ATTRIBUTES,
    OtelExporter,
    SpanSpec,
    _attrs,
    emit_tree,
    spans_from_events,
)
from clinic_agent.core.recorder import load_trace

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS = REPO_ROOT / "eval" / "traces"


def _stream(*specs) -> list[ev.Event]:
    """Events with monotonically increasing seq, and `t` given per event."""
    from dataclasses import replace

    return [replace(e, seq=i) for i, e in enumerate(specs, start=1)]


def _by_name(span: SpanSpec, name: str) -> list[SpanSpec]:
    return [s for s in span.walk() if s.name == name]


# --- the shape ------------------------------------------------------------------------------


def _booking_turn() -> list[ev.Event]:
    return _stream(
        ev.CallStarted(t=0.0, call_id="CALL-1", mode="telephony"),
        ev.CallerPresent(t=0.1, participant_id="sip_caller", phone="+15550001111"),
        ev.BotStartedSpeaking(t=0.5, utterance_id="utt-greeting"),
        ev.BotStoppedSpeaking(t=4.0, utterance_id="utt-greeting"),

        ev.SpeechStarted(t=5.0),
        ev.SpeechStopped(t=6.0),
        ev.FinalTranscript(t=6.4, text="I'd like to book a checkup", confidence=0.97),
        ev.LLMStarted(t=6.41, request_id="req-1"),
        ev.ModelRouted(t=6.42, request_id="req-1", tier="standard", model="haiku-4.5",
                       reason="scheduling dialogue"),
        ev.LLMToolUse(t=7.0, request_id="req-1", tool_call_id="tu-1", name="check_availability",
                      arguments={"date": "2026-09-07"}),
        ev.LLMCompleted(t=7.1, request_id="req-1", stop_reason="tool_use", text=""),
        ev.ToolCompleted(t=7.3, tool_call_id="tu-1", name="check_availability", ok=True,
                         latency_ms=200.0, http_status=200,
                         result={"ok": True, "count": 2}),
        ev.LLMStarted(t=7.31, request_id="req-2"),
        ev.LLMCompleted(t=8.0, request_id="req-2", stop_reason="end_turn",
                        text="I have Monday at 1 PM."),
        ev.BotStartedSpeaking(t=8.3, utterance_id="utt-2"),
        ev.BotStoppedSpeaking(t=11.0, utterance_id="utt-2"),
        ev.Hangup(t=12.0, reason="caller_left"),
    )


def test_the_tree_is_one_call_with_one_turn_and_its_stages():
    root = spans_from_events(_booking_turn())

    assert root.name == "call"
    assert root.attributes["call.id"] == "CALL-1"
    assert root.attributes["call.mode"] == "telephony"
    assert root.attributes["call.turns"] == 1
    assert root.attributes["call.tool_calls"] == 1

    (turn,) = _by_name(root, "turn")
    assert turn.attributes["turn.index"] == 0
    # Two LLM requests (the tool round trip), and the reply's TTS belongs to the turn — only
    # the greeting's does not, because there was no turn in flight when it played.
    assert {s.name for s in turn.children} == {"stt", "llm", "tool.check_availability", "tts"}
    assert len(_by_name(turn, "llm")) == 2


def test_a_turn_is_voice_to_voice_not_end_of_playback():
    """SpeechStopped -> BotStartedSpeaking, the same boundary as every latency number here.

    Ending it at BotStoppedSpeaking would fold the length of the reply into the latency, so a
    wordy answer would read as a slow one and the APM would disagree with inspect_call.py.
    """
    (turn,) = _by_name(spans_from_events(_booking_turn()), "turn")
    assert turn.start == 6.0
    assert turn.end == 8.3
    assert turn.duration_ms == pytest.approx(2300)


def test_the_greeting_hangs_off_the_call_not_off_a_turn():
    """It is real TTS with real latency and it happens before any caller turn exists."""
    root = spans_from_events(_booking_turn())
    greeting = [s for s in root.children if s.name == "tts"]
    assert len(greeting) == 1
    assert greeting[0].attributes["tts.utterance_id"] == "utt-greeting"


def test_a_tool_span_is_placed_by_its_measured_latency():
    """ToolCompleted carries the latency, so the span is anchored backwards from it — there is
    no ToolStarted event and inventing one would mean a second source of truth for the number."""
    (tool,) = _by_name(spans_from_events(_booking_turn()), "tool.check_availability")
    assert tool.end == 7.3
    assert tool.start == pytest.approx(7.1)
    assert tool.attributes["tool.ok"] is True and tool.attributes["tool.http_status"] == 200


def test_routing_decorates_the_llm_span_it_belongs_to():
    spans = _by_name(spans_from_events(_booking_turn()), "llm")
    routed = [s for s in spans if s.attributes.get("llm.request_id") == "req-1"]
    assert routed and routed[0].attributes["llm.model"] == "haiku-4.5"
    assert routed[0].attributes["llm.tier"] == "standard"


def test_a_booking_marks_the_call():
    stream = _booking_turn() + _stream(
        ev.ToolCompleted(t=9.0, tool_call_id="tu-2", name="confirm_booking", ok=True,
                         latency_ms=90.0, http_status=200,
                         result={"ok": True, "confirmation_id": "A754E2BC"}),
    )
    root = spans_from_events(stream)
    assert root.attributes["call.booked"] is True


def test_a_turn_the_caller_abandoned_ends_when_they_speak_again():
    """Not at the end of the call. A turn that never got a reply would otherwise run for the
    remaining minutes and dominate every percentile it appears in."""
    root = spans_from_events(_stream(
        ev.CallStarted(t=0.0, call_id="C", mode="local"),
        ev.SpeechStopped(t=1.0),
        ev.FinalTranscript(t=1.3, text="um", confidence=0.5),
        ev.SpeechStopped(t=2.0),
        ev.FinalTranscript(t=2.4, text="sorry, go on", confidence=0.9),
        ev.BotStartedSpeaking(t=3.0, utterance_id="u1"),
        ev.Hangup(t=90.0, reason="caller_left"),
    ))
    first, second = _by_name(root, "turn")
    assert first.end == 2.0
    assert second.end == 3.0


def test_an_empty_stream_produces_no_trace():
    """A process that started and stopped is not a call, and exporting one would put a row in
    the APM for every restart."""
    assert spans_from_events([]) is None


def test_a_degraded_provider_is_visible_on_the_call():
    root = spans_from_events(_stream(
        ev.CallStarted(t=0.0, call_id="C", mode="telephony"),
        ev.ProviderDegraded(t=1.0, provider="tts", reason="Context closed: cancelled", fatal=False),
        ev.Hangup(t=2.0),
    ))
    assert root.attributes["call.degraded"] == "tts"
    (span,) = _by_name(root, "provider.degraded")
    assert span.attributes["provider.fatal"] is False
    assert span.attributes["provider.reason"] == "Context closed"   # class only, not the message


# --- the boundary ---------------------------------------------------------------------------


def test_an_unlisted_attribute_is_refused_at_the_seam():
    """Raising, not dropping. A key that is not on the list is either a mistake or something
    somebody meant to add, and both deserve to be noticed here rather than to appear silently in
    an APM the following week."""
    with pytest.raises(KeyError):
        _attrs(caller__name="Nicholas Kumar")


def test_the_allowlist_has_no_field_that_could_hold_speech():
    for key in ATTRIBUTES:
        assert not any(word in key for word in ("text", "transcript", "utterance_text", "name.")), key
    for banned in ("stt.text", "caller.name", "caller.phone", "patient.dob", "llm.text"):
        assert banned not in ATTRIBUTES


def test_every_allowed_attribute_is_one_the_mapper_can_actually_emit():
    """An allowlist entry nothing emits is a claim about coverage that is not true.

    `call.emergency` was exactly that: the emergency flag is reducer state and no event carries
    it, so the attribute would have sat in the list permanently absent while looking supported.
    A reviewer reading ATTRIBUTES would have believed the APM could answer "how many emergency
    calls" and it could not.

    Driven by a maximal event stream rather than by reading the source, so it also proves the
    mapper reaches every branch.
    """
    stream = _stream(
        ev.CallStarted(t=0.0, call_id="MAX", mode="telephony"),
        ev.BotStartedSpeaking(t=0.5, utterance_id="utt-0"),
        ev.BotStoppedSpeaking(t=3.0, utterance_id="utt-0", completed=True),

        ev.SpeechStopped(t=4.0),
        ev.FinalTranscript(t=4.4, text="I need to cancel", confidence=0.95),
        ev.TurnHeld(t=4.5, tail="cancel", seconds=0.7, window=1, windows_max=3),
        ev.IntentClassified(t=5.0, intent="cancel_appointment", confidence=0.93, latency_ms=900.0),
        ev.LLMStarted(t=5.1, request_id="req-1"),
        ev.ModelRouted(t=5.11, request_id="req-1", tier="standard", model="haiku-4.5",
                       reason="cancel dialogue"),
        ev.LLMFailed(t=5.6, request_id="req-1", error="overloaded_error: try again"),
        ev.ProviderDegraded(t=5.7, provider="llm", reason="overloaded_error: try again",
                            fatal=False),
        ev.LLMStarted(t=5.8, request_id="req-2"),
        ev.LLMCompleted(t=6.2, request_id="req-2", stop_reason="tool_use", text=""),
        ev.ToolCompleted(t=6.5, tool_call_id="tu-1", name="confirm_booking", ok=True,
                         latency_ms=120.0, http_status=200, result={"ok": True}),
        ev.UserInterrupted(t=6.6),
        ev.BotStartedSpeaking(t=6.9, utterance_id="utt-1"),
        ev.BotStoppedSpeaking(t=7.0, utterance_id="utt-1", completed=False),
        ev.TransferFailed(t=7.5, reason="no transfer number configured"),
        ev.Hangup(t=8.0),
    )

    emitted = set()
    for span in spans_from_events(stream).walk():
        emitted |= set(span.attributes)

    assert ATTRIBUTES - emitted == set(), f"never emitted: {sorted(ATTRIBUTES - emitted)}"


def test_the_transcript_is_reduced_to_a_length():
    """Long enough to correlate a slow turn with a long utterance; useless to anyone who gets
    hold of the trace."""
    (stt,) = _by_name(spans_from_events(_booking_turn()), "stt")
    assert stt.attributes["stt.chars"] == len("I'd like to book a checkup")
    assert "stt.text" not in stt.attributes


def _corpus_secrets(path: Path) -> set[str]:
    """Every caller utterance and tool-result string in one recorded call.

    Read from the raw file rather than the replayed events so nothing is missed to replay
    divergence — the question is what the FILE contains, not what the reducer still accepts.
    """
    secrets: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        for key in ("text", "tail"):
            value = (obj.get(key) or "").strip() if isinstance(obj.get(key), str) else ""
            # Phrases, not words. A single word from a transcript ("slot", "Monday") appears in
            # tool names and enum values by coincidence, and a test that fails on a coincidence
            # gets deleted rather than fixed. A multi-word phrase in a span attribute is a leak
            # with no innocent explanation.
            if len(value) >= 12 and " " in value:
                secrets.add(value)
        result = obj.get("result")
        if isinstance(result, dict):
            for key in ("patient_name", "date_of_birth", "symptom_notes", "confirmation_id"):
                value = result.get(key)
                if isinstance(value, str) and value.strip():
                    secrets.add(value.strip())
    return secrets


@pytest.mark.parametrize("path", sorted(CORPUS.glob("*.jsonl")), ids=lambda p: p.stem)
def test_no_recorded_call_leaks_speech_or_identity_into_a_span(path):
    """The real test of the boundary: seven recorded calls, every attribute value checked
    against the names, birthdays, confirmation numbers, and sentences those calls contain."""
    root = spans_from_events(load_trace(path))
    if root is None:
        pytest.skip("empty trace")

    secrets = _corpus_secrets(path)
    assert secrets, f"{path.name} contains nothing sensitive — this test would pass vacuously"

    for span in root.walk():
        for key, value in span.attributes.items():
            assert key in ATTRIBUTES, f"{key} escaped the allowlist"
            if isinstance(value, str):
                for secret in secrets:
                    assert secret not in value, f"{span.name}.{key} leaked {secret!r}"


# --- the exporter ---------------------------------------------------------------------------


def test_spans_reach_an_exporter_with_the_recorded_times_not_the_export_times():
    """The SDK would happily time the spans itself, which is exactly what must not happen —
    these windows are the recorded ones. Re-timing them would report how long the export took.
    """
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    memory = InMemorySpanExporter()
    provider = TracerProvider(resource=Resource.create({"service.name": "test"}))
    provider.add_span_processor(SimpleSpanProcessor(memory))

    root = spans_from_events(_booking_turn())
    emit_tree(provider.get_tracer("test"), root, lambda t: int(t * 1e9))
    provider.force_flush()

    spans = {s.name: s for s in memory.get_finished_spans()}
    assert "call" in spans and "turn" in spans

    turn = spans["turn"]
    assert turn.start_time == int(6.0 * 1e9)
    assert turn.end_time == int(8.3 * 1e9)

    # One trace for the whole call, and the turn hangs off the call rather than being a root.
    assert turn.context.trace_id == spans["call"].context.trace_id
    assert turn.parent.span_id == spans["call"].context.span_id
    assert spans["stt"].parent.span_id == turn.context.span_id


def test_the_exporter_swallows_a_broken_backend():
    """The call is over by the time this runs. A telemetry backend must not be able to fail a
    session — same rule `_ship_metrics` follows, for the same reason."""
    exporter = OtelExporter("CALL-1")
    for event in _booking_turn():
        exporter.record(event)

    import clinic_agent.core.otel as otel_mod

    def boom(*_a, **_kw):
        raise RuntimeError("collector unreachable")

    original, otel_mod.export_spans = otel_mod.export_spans, boom
    try:
        exporter.close()          # must not raise
    finally:
        otel_mod.export_spans = original


def test_the_exporter_is_bounded():
    exporter = OtelExporter("RUNAWAY")
    for i in range(OtelExporter.MAX_EVENTS + 100):
        exporter.record(ev.LLMTextDelta(seq=i, t=float(i), request_id="r", text="x"))
    assert len(exporter._events) == OtelExporter.MAX_EVENTS
    assert exporter._dropped == 100


# --- the credential, in the shape Grafana actually hands you --------------------------------


@pytest.mark.parametrize("raw, expected", [
    # OTLP's documented header spec.
    ("Authorization=Basic abc123", {"Authorization": "Basic abc123"}),
    ("Authorization=Bearer tok, X-Scope-OrgID=42",
     {"Authorization": "Bearer tok", "X-Scope-OrgID": "42"}),
    # What Grafana Cloud's connection-details page gives you: base64(instanceID:token), on its
    # own. Requiring the header to be hand-assembled around it is a papercut that gets got wrong
    # once and then debugged as "the exporter is broken".
    ("MTgxOTUxNTpnbGNfZXlKdklqb2lNVGt3TU==", {"Authorization": "Basic MTgxOTUxNTpnbGNfZXlKdklqb2lNVGt3TU=="}),
    ("Basic MTgxOTUxNTpnbGNf", {"Authorization": "Basic MTgxOTUxNTpnbGNf"}),
    # dotenv strips quotes; a shell `export` does not.
    ('"Authorization=Basic abc"', {"Authorization": "Basic abc"}),
    ("", {}),
])
def test_the_credential_is_read_in_either_form(raw, expected):
    from clinic_agent.core.otel import _parse_headers

    assert _parse_headers(raw) == expected

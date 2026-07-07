"""Unit tests for the Phase-6 latency collector (dependency-free — no pipeline/frames needed).

Drives LatencyCollector via its event hooks with synthetic monotonic timestamps and asserts the
emitted turn/tool/summary records — the same lifecycle the MetricsTap feeds it at runtime.
"""

from __future__ import annotations

import json

import pytest

from pipecat.frames.frames import (
    FunctionCallInProgressFrame,
    LLMFullResponseEndFrame,
    LLMTextFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    UserStoppedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)

from clinic_agent.metrics import LatencyCollector, percentiles, record_frame_event


@pytest.fixture
def collector(tmp_path, monkeypatch):
    monkeypatch.setenv("CLINIC_LOG_DIR", str(tmp_path))
    return LatencyCollector(mode="telephony", call_id="C1")


def _events(collector) -> list[dict]:
    if not collector.log_path.exists():
        return []  # nothing was ever written (e.g. a call with no turns)
    return [json.loads(line) for line in collector.log_path.read_text().splitlines()]


def _summary(collector) -> dict:
    """The call_summary for THIS collector's call_id (collectors may share a log file)."""
    return next(
        e for e in _events(collector)
        if e["event"] == "call_summary" and e["call_id"] == collector.call_id
    )


def test_percentiles_nearest_rank():
    assert percentiles([]) == {"p50": None, "p95": None, "p99": None, "count": 0}
    p = percentiles([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
    assert (p["p50"], p["p95"], p["p99"], p["count"]) == (500, 1000, 1000, 10)


def test_normal_turn_all_stages(collector):
    collector.on_user_stopped(0.0)
    collector.on_transcript(0.20, 0.97)
    collector.on_llm_first_output(0.65)
    collector.on_llm_end(0.95)
    collector.on_output_audio(1.25)
    (turn,) = _events(collector)
    assert turn["event"] == "turn"
    assert (turn["asr_ms"], turn["llm_ms"], turn["tts_ms"], turn["e2e_ms"]) == (200, 450, 300, 1250)
    assert turn["asr_confidence"] == 0.97
    assert turn["had_tool_call"] is False


def test_tool_call_turn_llm_ends_at_first_tool_call(collector):
    collector.on_user_stopped(0.0)
    collector.on_transcript(0.10, None)  # confidence missing -> null, must not corrupt the turn
    collector.on_llm_first_output(0.40, is_tool_call=True)  # llm_ms ends here, not at llm_end
    collector.on_llm_end(1.00)
    collector.on_output_audio(1.30)
    turn = next(e for e in _events(collector) if e["event"] == "turn")
    assert turn["llm_ms"] == 300  # 0.40 - 0.10, i.e. up to the tool call
    assert turn["had_tool_call"] is True
    assert turn["asr_confidence"] is None


def test_greeting_audio_without_user_turn_is_ignored(collector):
    collector.on_output_audio(1.0)  # bot greeting before any caller turn
    assert [e for e in _events(collector) if e["event"] == "turn"] == []


def test_tts_null_when_llm_end_missing_before_audio(collector):
    collector.on_user_stopped(0.0)
    collector.on_transcript(0.15, 0.9)
    collector.on_llm_first_output(0.40)
    collector.on_output_audio(0.80)  # audio arrives before any llm_end -> tts unmeasurable
    turn = next(e for e in _events(collector) if e["event"] == "turn")
    assert turn["tts_ms"] is None
    assert turn["e2e_ms"] == 800


def test_outcome_booked_overrides_escalation(collector):
    collector.mark_escalation()  # a 0-slot window fired earlier...
    collector.record_tool("/confirm-booking", 200, 20.0, True)  # ...but a booking still happened
    collector.finalize()
    assert _summary(collector)["outcome"] == "booked"


def test_outcome_escalated_then_abandoned(tmp_path, monkeypatch):
    monkeypatch.setenv("CLINIC_LOG_DIR", str(tmp_path))
    esc = LatencyCollector(mode="local", call_id="ESC")
    esc.mark_escalation()
    esc.finalize()
    assert _summary(esc)["outcome"] == "escalated"

    ab = LatencyCollector(mode="local", call_id="AB")
    ab.finalize()
    assert _summary(ab)["outcome"] == "abandoned"


def test_real_frames_through_dispatch_produce_a_turn(collector):
    """Push the ACTUAL pipeline frame classes through record_frame_event (not the collector hooks).

    This is the boundary that regressed live: the VADProcessor emits VADUserStoppedSpeakingFrame,
    not UserStoppedSpeakingFrame, so the tap never opened a turn and every call logged turns=0.
    Exercising real frame instances here fails loudly if that mapping is ever wrong again.
    """
    clock = iter([0.0, 0.20, 0.65, 0.95, 1.25])  # asr 200 / llm 450 / tts 300 / e2e 1250 ms
    frames = [
        VADUserStoppedSpeakingFrame(),
        TranscriptionFrame("I'd like to book an appointment", "caller", "2026-07-07T00:00:00Z"),
        LLMTextFrame("Sure — "),
        LLMFullResponseEndFrame(),
        TTSAudioRawFrame(b"\x00\x00" * 240, 24000, 1),
    ]
    for frame in frames:
        record_frame_event(collector, frame, next(clock))

    turn = next(e for e in _events(collector) if e["event"] == "turn")
    assert (turn["asr_ms"], turn["llm_ms"], turn["tts_ms"], turn["e2e_ms"]) == (200, 450, 300, 1250)


def test_plain_user_stopped_frame_does_not_open_a_turn(collector):
    """Negative guard for the exact confusion that caused the live turns=0 bug.

    The higher-level UserStoppedSpeakingFrame is NOT the VAD-silence signal in this pipeline —
    only VADUserStoppedSpeakingFrame is. If someone (re)adds UserStoppedSpeakingFrame to the tap's
    dispatch, this test's transcript+audio would wrongly produce a turn and this assertion fails.
    """
    record_frame_event(collector, UserStoppedSpeakingFrame(), 0.0)
    record_frame_event(
        collector, TranscriptionFrame("hello", "caller", "2026-07-07T00:00:00Z"), 0.1
    )
    record_frame_event(collector, TTSAudioRawFrame(b"\x00\x00" * 10, 24000, 1), 0.5)
    assert [e for e in _events(collector) if e["event"] == "turn"] == []


def test_vad_processor_still_emits_the_frame_we_listen_for():
    """Canary for the Pipecat CONTRACT, not just our code.

    The VAD-silence boundary depends on VADProcessor emitting VADUserStoppedSpeakingFrame. Our own
    dispatch tests would keep passing even if a future Pipecat version stopped emitting that frame
    (they construct it directly) — production would silently record turns=0 again. This ties the
    class we listen for to what VADProcessor actually references in its source, so a rename or
    replacement on a Pipecat upgrade fails the suite loudly and forces a re-check.
    """
    import inspect

    from pipecat.processors.audio.vad_processor import VADProcessor

    source = inspect.getsource(VADProcessor)
    assert VADUserStoppedSpeakingFrame.__name__ in source, (
        "VADProcessor no longer references VADUserStoppedSpeakingFrame — update the VAD-silence "
        "boundary in clinic_agent/metrics.py to whatever frame it now emits (see the turns=0 "
        "regression that this guards against)."
    )


def test_real_tool_call_frame_ends_llm_latency(collector):
    clock = iter([0.0, 0.10, 0.40, 1.30])
    frames = [
        VADUserStoppedSpeakingFrame(),
        TranscriptionFrame("book me for tuesday", "caller", "2026-07-07T00:00:00Z"),
        FunctionCallInProgressFrame("check_availability", "call_1", {"date": "2026-07-08"}),
        TTSAudioRawFrame(b"\x00\x00" * 240, 24000, 1),
    ]
    for frame in frames:
        record_frame_event(collector, frame, next(clock))
    turn = next(e for e in _events(collector) if e["event"] == "turn")
    assert turn["llm_ms"] == 300 and turn["had_tool_call"] is True


def test_finalize_is_idempotent(collector):
    collector.on_user_stopped(0.0)
    collector.on_transcript(0.2, 0.9)
    collector.on_llm_first_output(0.5)
    collector.on_llm_end(0.7)
    collector.on_output_audio(1.0)
    collector.finalize()
    collector.finalize()  # second call must not write a second summary
    summaries = [e for e in _events(collector) if e["event"] == "call_summary"]
    assert len(summaries) == 1
    assert summaries[0]["turns"] == 1

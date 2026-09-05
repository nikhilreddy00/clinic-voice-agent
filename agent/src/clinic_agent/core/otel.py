"""Phase 16 — one OpenTelemetry trace per call, one span per turn.

    CLINIC_OTEL_ENDPOINT=https://otlp-gateway-<region>.grafana.net/otlp
    CLINIC_OTEL_HEADERS="Authorization=Basic <base64 instance:token>"

Unset endpoint means no exporter, no SDK import, no cost. That is the default.

WHY THIS EXISTS WHEN logs/traces/ ALREADY DOES
-----------------------------------------------
The JSONL event trace is complete and exact and it is the thing to read when debugging one
call. What it cannot do is answer a question ACROSS calls — "which stage got slower this week",
"is the p95 tool latency the API or the network", "how often does the hedge fire" — without
someone writing the query by hand each time. That is what an APM is for, and OTel is how you
talk to one without picking a vendor. `docs/build_spec.md` records the Grafana dashboards built
on top of this.

THE SHAPE, AND WHY IT IS BUILT AT THE END
------------------------------------------
Spans are derived from the finished event stream and exported with explicit start/end
timestamps, rather than opened and closed live as the call runs. Three things fall out of that,
all of them good:

  * `spans_from_events` is PURE — same discipline as `telemetry.record_event` and `reduce`. It
    is a list in, a tree out, and it unit-tests without an SDK, a collector, or a clock.
  * nothing in the voice path can block on, or be broken by, a telemetry backend. The export is
    one call at teardown, after the caller has hung up.
  * ANY recorded trace can be exported, including one from `logs/traces/` recorded months ago.
    A call worth investigating does not have to have been instrumented in advance.

The boundaries match the ones every latency number in this project already uses (see
`telemetry`): a turn is `SpeechStopped -> BotStartedSpeaking`, which is voice-to-voice. Ending
it at `BotStoppedSpeaking` instead would fold the length of the reply into the latency and make
a wordy answer look like a slow one.

ONE THING TO KNOW BEFORE QUERYING TURN DURATION: a turn the engine deliberately HELD (the
caller paused mid-sentence and `core/endpointing.py` bought them grace) has no reply, so its
span runs to whenever they spoke again — seconds, legitimately. Those carry `turn.held=true`
and any latency percentile has to exclude them, or the graph measures how long callers think.

NO PHI ON A SPAN. NOT ONE FIELD.
---------------------------------
Spans leave the building. `logs/traces/` does not, and it holds caller utterances precisely
because it is local and git-ignored. So the transcript, the caller's name, their date of birth,
the phone number, and the symptom notes are all absent here — what is exported is the SHAPE of
the call: how long each stage took, which tools ran, which intent was chosen, what failed.

The enforcement is `ATTRIBUTES`, a closed set of keys, checked by `_attrs`. A new attribute has
to be added there deliberately, and `test_otel.py` asserts no known name or date of birth
appears in any exported value. Phase 17 generalises this into one tagged boundary; this is the
part of it that is free today.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Iterable

from loguru import logger

from . import events as ev

# Every attribute key that may ever reach a span. A closed set rather than a redaction pass:
# blocking known-bad names is a losing game, and the list of things that ARE safe to export is
# short, stable, and reviewable in one screen.
ATTRIBUTES = frozenset({
    # call
    "call.id", "call.mode", "call.turns", "call.tool_calls", "call.interruptions",
    "call.intent", "call.degraded", "call.booked", "call.transfer_failed_reason",
    # turn
    "turn.index", "turn.intent", "turn.held", "turn.interrupted",
    # stt
    "stt.confidence", "stt.chars",
    # classifier
    "classifier.intent", "classifier.confidence",
    # llm
    "llm.request_id", "llm.tier", "llm.model", "llm.stop_reason", "llm.chars",
    "llm.error", "llm.routing_reason",
    # tool
    "tool.name", "tool.ok", "tool.http_status", "tool.call_id",
    # tts
    "tts.utterance_id", "tts.completed",
    # provider health
    "provider.name", "provider.reason", "provider.fatal",
})


@dataclass
class SpanSpec:
    """One span: a name, a monotonic window, plain attributes, and its children.

    Times are `time.monotonic()` seconds, exactly as the events carry them. Converting to wall
    clock is the exporter's job and needs an anchor the pure mapper does not have.
    """

    name: str
    start: float
    end: float
    attributes: dict[str, Any] = field(default_factory=dict)
    children: list["SpanSpec"] = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        return (self.end - self.start) * 1000

    def walk(self) -> Iterable["SpanSpec"]:
        yield self
        for child in self.children:
            yield from child.walk()


def _attrs(**kwargs: Any) -> dict[str, Any]:
    """Drop `None`s and refuse anything not on the allowlist.

    Raising rather than dropping is deliberate: a key that is not on the list is either a
    mistake or a new attribute somebody meant to add, and both deserve to be noticed at the
    seam rather than to appear silently in an APM the following week.
    """
    out = {}
    for key, value in kwargs.items():
        key = key.replace("__", ".")
        if value is None:
            continue
        if key not in ATTRIBUTES:
            raise KeyError(f"{key!r} is not an allowed span attribute — see otel.ATTRIBUTES")
        out[key] = value
    return out


# --- the pure mapper ------------------------------------------------------------------------


def spans_from_events(events: list[ev.Event]) -> SpanSpec | None:
    """Fold one call's event stream into a span tree. Pure: no clock, no I/O, no SDK.

    Returns ``None`` for an empty stream — a session that produced no events is not a call, and
    exporting a zero-length trace for it would put a row in the APM for every process restart.
    """
    events = sorted(events, key=lambda e: e.seq)
    if not events:
        return None

    first, last = events[0].t, events[-1].t
    call_id = next((e.call_id for e in events if isinstance(e, ev.CallStarted)), "")
    mode = next((e.mode for e in events if isinstance(e, ev.CallStarted)), "local")

    root = SpanSpec(name="call", start=first, end=last)
    turns: list[SpanSpec] = []
    current: SpanSpec | None = None

    # Cross-call tallies, collected in the same pass rather than a second one over the list.
    tool_calls = interruptions = 0
    booked = False
    # Only a FAILED transfer is an event. A successful one is a state transition the reducer
    # makes and LiveKit carries out, so there is nothing in the stream to hang a span on; the
    # attribute is named for what it actually records rather than for what a reader might
    # assume. Adding a TransferSucceeded event is the honest fix if this is ever needed.
    transfer_failed_reason = ""
    degraded: list[str] = []
    intent = ""

    # Open stage windows, keyed by what closes them.
    stt_open: float | None = None
    llm_open: dict[str, float] = {}
    tts_open: dict[str, float] = {}
    # ModelRouted arrives with the REQUEST, and the llm span is not built until the request
    # completes, so the routing decision waits here rather than looking for a span that does
    # not exist yet.
    routing: dict[str, dict[str, Any]] = {}

    def attach(span: SpanSpec) -> None:
        """Stage spans belong to the turn in flight, or to the call itself before one starts.

        The greeting is the case that matters: it is a real utterance with real TTS latency and
        it happens before any caller turn exists, so it hangs off the root.
        """
        (current or root).children.append(span)

    for event in events:
        t = event.t

        if isinstance(event, ev.SpeechStopped):
            # Voice-to-voice starts at VAD silence. Any previous turn that never got a reply
            # (the caller spoke again, or hung up) ends here rather than running to the end of
            # the call and dominating every percentile.
            if current is not None and current.end <= current.start:
                current.end = t
            current = SpanSpec(name="turn", start=t, end=t,
                               attributes=_attrs(turn__index=len(turns)))
            turns.append(current)
            root.children.append(current)
            stt_open = t

        elif isinstance(event, ev.FinalTranscript):
            if stt_open is not None:
                attach(SpanSpec(name="stt", start=stt_open, end=t, attributes=_attrs(
                    # The LENGTH of what was said, never the words. Enough to correlate a slow
                    # turn with a long utterance; useless to anyone who gets hold of the trace.
                    stt__chars=len(event.text),
                    stt__confidence=event.confidence,
                )))
                stt_open = None

        elif isinstance(event, ev.TurnHeld) and current is not None:
            current.attributes["turn.held"] = True

        elif isinstance(event, ev.UserInterrupted):
            interruptions += 1
            if current is not None:
                current.attributes["turn.interrupted"] = True

        elif isinstance(event, ev.LLMStarted):
            llm_open[event.request_id] = t

        elif isinstance(event, (ev.LLMCompleted, ev.LLMFailed)):
            start = llm_open.pop(event.request_id, None)
            if start is not None:
                failed = isinstance(event, ev.LLMFailed)
                attach(SpanSpec(name="llm", start=start, end=t, attributes=_attrs(
                    llm__request_id=event.request_id,
                    llm__stop_reason=None if failed else event.stop_reason,
                    llm__chars=None if failed else len(event.text),
                    llm__error=_error_class(event.error) if failed else None,
                    **routing.pop(event.request_id, {}),
                )))

        elif isinstance(event, ev.ModelRouted):
            routing[event.request_id] = {
                "llm__tier": event.tier,
                "llm__model": event.model,
                "llm__routing_reason": event.reason,
            }

        elif isinstance(event, ev.ToolCompleted):
            tool_calls += 1
            if event.name == "confirm_booking" and event.ok:
                booked = True
            attach(SpanSpec(name=f"tool.{event.name}", start=t - event.latency_ms / 1000, end=t,
                            attributes=_attrs(
                                tool__name=event.name,
                                tool__ok=event.ok,
                                tool__call_id=event.tool_call_id,
                                tool__http_status=event.http_status,
                            )))

        elif isinstance(event, ev.IntentClassified):
            intent = event.intent
            if current is not None:
                current.attributes["turn.intent"] = event.intent
            attach(SpanSpec(name="classifier", start=t - event.latency_ms / 1000, end=t,
                            attributes=_attrs(classifier__intent=event.intent,
                                              classifier__confidence=event.confidence)))

        elif isinstance(event, ev.BotStartedSpeaking):
            # The turn ends at FIRST AUDIO OUT, not at the end of playback: ending it at
            # BotStoppedSpeaking would fold the length of the reply into the latency and make a
            # wordy answer look like a slow one.
            if current is not None and current.end <= current.start:
                current.end = t
            tts_open[event.utterance_id] = t

        elif isinstance(event, ev.BotStoppedSpeaking):
            start = tts_open.pop(event.utterance_id, None)
            if start is not None:
                attach(SpanSpec(name="tts", start=start, end=t, attributes=_attrs(
                    tts__utterance_id=event.utterance_id, tts__completed=event.completed,
                )))

        elif isinstance(event, ev.ProviderDegraded):
            degraded.append(event.provider)
            attach(SpanSpec(name="provider.degraded", start=t, end=t, attributes=_attrs(
                provider__name=event.provider,
                provider__reason=_error_class(event.reason),
                provider__fatal=event.fatal,
            )))

        elif isinstance(event, ev.TransferFailed):
            transfer_failed_reason = event.reason

    # A turn the call ended in the middle of.
    if current is not None and current.end <= current.start:
        current.end = last

    root.attributes = _attrs(
        call__id=call_id,
        call__mode=mode,
        call__turns=len(turns),
        call__tool_calls=tool_calls,
        call__interruptions=interruptions,
        call__booked=booked or None,
        call__transfer_failed_reason=transfer_failed_reason or None,
        call__degraded=",".join(sorted(set(degraded))) or None,
        # The last intent the classifier settled on. Intent is sticky (`intents.resolve_intent`),
        # so on a call that stayed on one task this is that task.
        call__intent=intent or None,
    )
    return root


def _error_class(message: str) -> str:
    """The first few words of an error, as a label — never a full message.

    A provider's error text can echo the request back, and the request contains what the caller
    said. A truncated class is enough to group by in an APM and cannot carry an utterance.
    """
    return (message or "").split(":")[0].strip()[:64] or "unknown"


# --- the exporter -------------------------------------------------------------------------


def otel_enabled() -> bool:
    return bool(os.getenv("CLINIC_OTEL_ENDPOINT"))


# A header NAME is a short token. A base64 credential is long, and its only "=" is the padding
# at the very end — so the left-hand side of its first split is the whole blob and the right-hand
# side is empty. That is what separates the two forms below.
_HEADER_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,39}")


def _parse_headers(raw: str) -> dict[str, str]:
    """Accept either OTLP's `"k=v,k2=v2"` header spec or a bare Grafana Cloud credential.

    Grafana's connection-details page hands you `base64(instanceID:token)` and nothing else, so
    requiring `Authorization=Basic <blob>` means hand-assembling a header — a papercut that gets
    got wrong once and then debugged as "the exporter is broken". Both forms work.
    """
    raw = raw.strip().strip('"').strip("'")
    if not raw:
        return {}

    pairs: dict[str, str] = {}
    for part in raw.split(","):
        name, sep, value = part.partition("=")
        # `value.strip("=")` is the discriminator: a base64 blob's only "=" is its trailing
        # padding, so splitting one yields a value made entirely of "=" — never a real header.
        if not (sep and value.strip().strip("=") and _HEADER_NAME.fullmatch(name.strip())):
            pairs = {}
            break
        pairs[name.strip()] = value.strip()
    if pairs:
        return pairs

    scheme = "" if raw.lower().startswith(("basic ", "bearer ")) else "Basic "
    return {"Authorization": f"{scheme}{raw}"}


class OtelExporter:
    """Collects a call's events and exports the span tree once, at teardown.

    Constructed only when `CLINIC_OTEL_ENDPOINT` is set, so the SDK is never imported and never
    paid for on a machine that is not sending anywhere.

    Every failure here is swallowed. The call is over by the time this runs and a telemetry
    backend must not be able to fail a session — the same rule `_ship_metrics` follows, for the
    same reason.
    """

    # ~300 events on a healthy call. A session that produces 20,000 has a loop problem, and the
    # cap keeps that from becoming a memory problem too.
    MAX_EVENTS = 20_000

    def __init__(self, call_id: str, *, service_name: str = "clinic-voice-agent") -> None:
        self.call_id = call_id
        self.service_name = service_name
        self._events: list[ev.Event] = []
        self._dropped = 0
        # Anchor the monotonic clock to wall time ONCE, here, so every span in the call is
        # converted with the same offset. Sampling it per span would let clock drift show up as
        # spans that overlap their own parents.
        self._wall0 = time.time()
        self._mono0 = time.monotonic()

    def record(self, event: ev.Event) -> None:
        if len(self._events) < self.MAX_EVENTS:
            self._events.append(event)
        else:
            self._dropped += 1

    def _to_ns(self, monotonic: float) -> int:
        return int((self._wall0 + (monotonic - self._mono0)) * 1e9)

    def close(self) -> None:
        """Build the spans and ship them. Best-effort; never raises."""
        try:
            root = spans_from_events(self._events)
            if root is None:
                return
            if self._dropped:
                logger.warning(f"[otel] {self._dropped} events over the cap were not traced")
            if not export_spans(root, self._to_ns, service_name=self.service_name):
                logger.warning("[otel] span export did not flush — the backend rejected it")
        except Exception as exc:  # noqa: BLE001 - telemetry must never fail a call
            logger.warning(f"[otel] export failed: {exc}")


def export_spans(root: SpanSpec, to_ns, *, service_name: str = "clinic-voice-agent") -> bool:
    """Emit one span tree through OTLP/HTTP. Imports the SDK lazily, on first use.

    Returns whether the flush completed. The SDK reports a rejected batch by logging and
    returning False, never by raising — so ignoring this return is how "exported 7 traces"
    gets printed for seven traces that went nowhere.
    """
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    global _PROVIDER
    if _PROVIDER is None:
        exporter = OTLPSpanExporter(
            endpoint=os.getenv("CLINIC_OTEL_ENDPOINT", "").strip().strip('"').rstrip("/")
            + "/v1/traces",
            headers=_parse_headers(os.getenv("CLINIC_OTEL_HEADERS", "")),
        )
        _PROVIDER = TracerProvider(resource=Resource.create({"service.name": service_name}))
        _PROVIDER.add_span_processor(BatchSpanProcessor(exporter))
    emit_tree(_PROVIDER.get_tracer("clinic_agent"), root, to_ns)
    return bool(_PROVIDER.force_flush(timeout_millis=5000))


# One provider per process, not per call. Each one owns a batch processor and an HTTP session;
# building 800 of them in a worker is the same mistake `shared_anthropic_client` fixed for the
# dialogue LLM (Phase 14).
_PROVIDER = None


def emit_tree(tracer, span: SpanSpec, to_ns, parent=None) -> None:
    """Walk the tree depth-first, opening each span with explicit start/end times.

    The SDK's context manager would time the spans itself, which is exactly what must not
    happen here — these windows are the recorded ones, and re-timing them would report how long
    the export took.
    """
    from opentelemetry import trace as _trace

    ctx = _trace.set_span_in_context(parent) if parent is not None else None
    otel_span = tracer.start_span(
        span.name, context=ctx, start_time=to_ns(span.start), attributes=span.attributes
    )
    for child in span.children:
        emit_tree(tracer, child, to_ns, parent=otel_span)
    otel_span.end(end_time=to_ns(span.end))

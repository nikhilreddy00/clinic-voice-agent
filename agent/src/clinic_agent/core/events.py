"""Phase 10 — typed events for the in-house orchestration loop.

Everything that happens on a call becomes one of these. Adapters (media, STT, LLM, TTS,
tools) push events onto the session's single ``asyncio.Queue``; the drain loop feeds each one
to the pure :func:`clinic_agent.core.reducer.reduce`, which returns the next state plus a list
of :mod:`clinic_agent.core.actions` for the adapters to execute. Executing an action produces
more events. That cycle is the whole engine.

WHAT IS AND IS NOT AN EVENT HERE
--------------------------------
Raw audio is **deliberately not** an event. 20 ms PCM frames would be ~50/second of
unserializable bytes, and the reducer has no use for them. Audio goes
``media -> TurnEngine``, and only the TurnEngine's *derived* decisions
(:class:`SpeechStarted`, :class:`SpeechStopped`, :class:`UserInterrupted`) reach the reducer.
That boundary is what makes a recorded call trace small enough to commit and exact enough to
replay: every event in this module is JSON round-trippable, so
``recorder.load_trace() -> reduce()`` in CI reproduces a real call with no audio, no network,
and no API keys.

Every event carries:
  ``seq`` — monotonically increasing per call; the replay ordering key.
  ``t``   — ``time.monotonic()`` at creation. The reducer NEVER calls a clock; all timing it
            needs arrives on the event, which is the other half of deterministic replay.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, ClassVar


@dataclass(frozen=True, slots=True)
class Event:
    """Base for every event. Subclasses add their own payload fields.

    ``kind`` is the wire discriminator used by the recorder; it defaults to the class name and
    is what :func:`event_from_dict` dispatches on.
    """

    seq: int = 0
    t: float = 0.0

    kind: ClassVar[str] = "Event"

    def to_dict(self) -> dict[str, Any]:
        return {"kind": type(self).kind, **asdict(self)}


# --- call lifecycle ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CallStarted(Event):
    """The session was constructed and adapters are coming up. Always event 0."""

    call_id: str = ""
    mode: str = "local"

    kind: ClassVar[str] = "CallStarted"


@dataclass(frozen=True, slots=True)
class CallerPresent(Event):
    """A human is actually on the other end — the moment to greet.

    The two transports reach this at different times (local: the mic opens at startup;
    telephony: the inbound SIP participant joins an until-then-empty room), which is exactly
    why it is an event rather than something the reducer infers.
    """

    participant_id: str = ""
    # The caller's number (E.164) from the SIP participant, empty on the local path. Phase 13
    # uses it to look up caller memory and to scope every verified tool call. It is NEVER
    # asked of the caller and never read back to them.
    phone: str = ""

    kind: ClassVar[str] = "CallerPresent"


@dataclass(frozen=True, slots=True)
class CallerMemoryLoaded(Event):
    """The caller-memory lookup for this ANI came back (Phase 13).

    Fired in parallel with the greeting, never awaited: the greeting is deterministic and must
    not wait on a database, so "welcome back" costs the caller no added latency and a memory
    lookup that fails or is slow simply means the agent greets the way it always did.

    Carries no identity on purpose — see the API's /caller-memory docstring. Recognising a
    number is not authentication, and this event exists before anyone has verified anything.
    """

    known: bool = False
    upcoming_appointments: int = 0

    kind: ClassVar[str] = "CallerMemoryLoaded"


@dataclass(frozen=True, slots=True)
class Hangup(Event):
    """The call is over (caller left, Ctrl-C, or the agent closed it)."""

    reason: str = "caller_left"

    kind: ClassVar[str] = "Hangup"


# --- turn taking (derived by the TurnEngine from audio; audio itself never gets here) ----


@dataclass(frozen=True, slots=True)
class SpeechStarted(Event):
    """VAD says the caller began speaking."""

    kind: ClassVar[str] = "SpeechStarted"


@dataclass(frozen=True, slots=True)
class SpeechStopped(Event):
    """VAD silence — the caller's turn ended. This is the start of the latency clock.

    Named after the boundary the Phase-6 metrics use (``VADUserStoppedSpeakingFrame``, not
    ``UserStoppedSpeakingFrame`` — getting that wrong silently recorded ``turns=0``).
    """

    kind: ClassVar[str] = "SpeechStopped"


@dataclass(frozen=True, slots=True)
class UserInterrupted(Event):
    """Sustained caller speech during bot playback got past the mic gate — a real barge-in."""

    kind: ClassVar[str] = "UserInterrupted"


# --- speech to text ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PartialTranscript(Event):
    """An interim, still-changing STT hypothesis. Feeds barge-in word counting."""

    text: str = ""

    kind: ClassVar[str] = "PartialTranscript"


@dataclass(frozen=True, slots=True)
class TurnHeld(Event):
    """The engine declined to end the caller's turn, or ran out of patience and ended it.

    Deepgram says the turn is over after 300 ms of silence; ``core.endpointing`` decides
    whether the caller agrees. That decision is turn-taking — the thing a caller feels most
    directly when it goes wrong — and it used to leave no trace at all, so a call where
    somebody was talked over mid-sentence looked identical to one where they weren't. It is a
    derived decision, not audio, so it belongs in the event stream by the same rule that keeps
    raw PCM out of it.

    ``released=True`` means the hold expired and the fragment was sent anyway.
    """

    tail: str = ""          # last few words only — enough to see WHY, without a transcript copy
    seconds: float = 0.0
    window: int = 0
    windows_max: int = 0
    released: bool = False

    kind: ClassVar[str] = "TurnHeld"


@dataclass(frozen=True, slots=True)
class FinalTranscript(Event):
    """A finalized STT utterance. This is what becomes a user message."""

    text: str = ""
    confidence: float | None = None

    kind: ClassVar[str] = "FinalTranscript"


# --- LLM ---------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LLMStarted(Event):
    """The request identified by ``request_id`` is now streaming."""

    request_id: str = ""

    kind: ClassVar[str] = "LLMStarted"


@dataclass(frozen=True, slots=True)
class LLMTextDelta(Event):
    """A streamed text chunk. The reducer accumulates these into speakable sentences."""

    request_id: str = ""
    text: str = ""

    kind: ClassVar[str] = "LLMTextDelta"


@dataclass(frozen=True, slots=True)
class LLMToolUse(Event):
    """The model asked to call a tool. Arrives fully-formed (arguments already parsed)."""

    request_id: str = ""
    tool_call_id: str = ""
    name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)

    kind: ClassVar[str] = "LLMToolUse"


@dataclass(frozen=True, slots=True)
class LLMCompleted(Event):
    """The stream ended. ``stop_reason`` is ``"tool_use"`` or ``"end_turn"``."""

    request_id: str = ""
    text: str = ""
    stop_reason: str = "end_turn"

    kind: ClassVar[str] = "LLMCompleted"


@dataclass(frozen=True, slots=True)
class LLMFailed(Event):
    """The request errored or timed out. Cancellation does NOT produce this."""

    request_id: str = ""
    error: str = ""

    kind: ClassVar[str] = "LLMFailed"


# --- tools -------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ToolCompleted(Event):
    """A tool finished. ``result`` is the LLM-facing dict; failures set ``ok=False`` rather
    than raising, so a scheduling-API outage is a dialogue event, not a crashed turn."""

    tool_call_id: str = ""
    name: str = ""
    result: dict[str, Any] = field(default_factory=dict)
    ok: bool = False
    latency_ms: float = 0.0
    http_status: int | None = None

    kind: ClassVar[str] = "ToolCompleted"


# --- TTS / playback ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BotStartedSpeaking(Event):
    """First audio frame of ``utterance_id`` reached the speaker / SIP egress."""

    utterance_id: str = ""

    kind: ClassVar[str] = "BotStartedSpeaking"


@dataclass(frozen=True, slots=True)
class BotStoppedSpeaking(Event):
    """Playback of ``utterance_id`` ended. ``completed=False`` means it was cut by a barge-in."""

    utterance_id: str = ""
    completed: bool = True

    kind: ClassVar[str] = "BotStoppedSpeaking"


# --- reasoning layer (Phase 12) ------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class IntentClassified(Event):
    """The classifier answered. Arrives asynchronously, in parallel with the dialogue turn.

    Never blocks a turn: if it lands after generation started and materially contradicts the
    flow in progress, the reducer re-plans. Putting it in series would spend its whole latency
    budget on the caller's very first impression.
    """

    intent: str = "unknown"
    confidence: float = 0.0
    latency_ms: float = 0.0

    kind: ClassVar[str] = "IntentClassified"


@dataclass(frozen=True, slots=True)
class IntentClassificationFailed(Event):
    """The classifier errored or timed out. The call continues unscoped — it is an optimization,
    not a dependency, and a caller must never lose a turn because a side model was unavailable."""

    error: str = ""

    kind: ClassVar[str] = "IntentClassificationFailed"


@dataclass(frozen=True, slots=True)
class ModelRouted(Event):
    """Which model served a turn, and why. Emitted when the adapter resolves the reducer's tier.

    Exists so the routing decision is visible in the call trace next to its latency and cost,
    rather than being an invisible property of a policy table someone has to go read.
    """

    request_id: str = ""
    tier: str = ""
    model: str = ""
    reason: str = ""

    kind: ClassVar[str] = "ModelRouted"


# --- provider health ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProviderDegraded(Event):
    """A provider is unhealthy. Recorded into ``state.degraded``; the Phase-15 ladder in the
    reducer consumes it (retry/hedge -> scripted line -> transfer)."""

    provider: str = ""
    reason: str = ""
    # Phase 15: the adapter has exhausted its reconnects/retries and is not coming back on its
    # own. That is the difference between "one socket blipped" and "this call can no longer
    # hear / speak", and only the second one should push the caller down the ladder.
    fatal: bool = False

    kind: ClassVar[str] = "ProviderDegraded"


@dataclass(frozen=True, slots=True)
class ProviderRecovered(Event):
    """A previously-degraded provider is serving again (Phase 15).

    The missing half of :class:`ProviderDegraded`. Before this existed a provider could only
    ever be added to ``state.degraded``, so one dropped socket downgraded the model tier for
    the remaining twenty turns of a call that had been healthy again since turn two.
    """

    provider: str = ""

    kind: ClassVar[str] = "ProviderRecovered"


@dataclass(frozen=True, slots=True)
class TransferFailed(Event):
    """The warm transfer could not be performed (Phase 15).

    No destination configured, the local path (there is no SIP leg to transfer), or the
    provider refused. The caller must still be told something — a hand-off that ends in a
    click is worse than never having offered one.
    """

    reason: str = ""

    kind: ClassVar[str] = "TransferFailed"


@dataclass(frozen=True, slots=True)
class ToolSlow(Event):
    """A tool call has been in flight past its budget and the caller is hearing nothing.

    Phase 15. The scheduling client's timeout used to sit inside the voice turn at 10 s with
    no filler, so a slow backend was ten seconds of dead air the caller reads as a dropped
    call. Emitted once per tool call by the executor; the reducer speaks one filler line per
    caller turn.
    """

    tool_call_id: str = ""
    name: str = ""
    waited_ms: float = 0.0

    kind: ClassVar[str] = "ToolSlow"


# --- (de)serialization for the recorder ---------------------------------------------------

_EVENT_TYPES: dict[str, type[Event]] = {
    cls.kind: cls
    for cls in (
        CallStarted,
        CallerPresent,
        CallerMemoryLoaded,
        Hangup,
        SpeechStarted,
        SpeechStopped,
        UserInterrupted,
        PartialTranscript,
        TurnHeld,
        FinalTranscript,
        LLMStarted,
        LLMTextDelta,
        LLMToolUse,
        LLMCompleted,
        LLMFailed,
        ToolCompleted,
        BotStartedSpeaking,
        BotStoppedSpeaking,
        ProviderDegraded,
        ProviderRecovered,
        ToolSlow,
        TransferFailed,
        IntentClassified,
        IntentClassificationFailed,
        ModelRouted,
    )
}


def event_from_dict(obj: dict[str, Any]) -> Event:
    """Rebuild an event from its recorded dict form.

    Unknown fields are dropped rather than raising: a trace recorded by an older build must
    still replay after a field is added, otherwise the regression corpus rots on every change.
    """
    payload = dict(obj)
    kind = payload.pop("kind", None)
    cls = _EVENT_TYPES.get(kind or "")
    if cls is None:
        raise ValueError(f"unknown event kind {kind!r}")
    known = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in payload.items() if k in known})

"""Phase 10 — the call state machine's data.

``CallState`` is immutable and contains only plain data. The reducer returns a new one via
``dataclasses.replace``; nothing mutates it in place. That is what lets a replay test compare
final states with ``==`` and get a meaningful answer.

This replaces the Phase-1..9 arrangement where per-call state lived in function locals of
``pipeline.run_agent()`` (``LatencyCollector``, ``SchedulingClient``, ``LLMContext``, the mic
gate). Those were built once per *process*, so on the telephony path a second caller inherited
the first caller's message history and wrote turns into a finalized summary. Here every field
below is constructed per ``CallSession``, so N sessions in one process cannot see each other.

The ``Phase`` enum is also the other half of the Phase-4 fix: dialogue control used to live
entirely in prompt text ("call hold_slot IMMEDIATELY", "ONE confirmation gate only"). The
prompt still guides *what to say*; this FSM decides *what the engine does*, and it is
enforceable and testable in a way prose is not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..intents import Intent


class Phase(str, Enum):
    """Where the call is in the turn cycle.

    Deliberately about the *turn*, not the dialogue script — the booking script
    (GREETING_DISCLOSURE → … → CLOSE in ``docs/build_spec.md``) is prompt-driven and can
    branch freely, while these six states are what the orchestration loop must know to route
    audio, cancel work, and decide whether the caller is allowed to interrupt.
    """

    INIT = "init"            # session up, nobody on the line yet
    GREETING = "greeting"    # speaking the deterministic AI-disclosure greeting
    LISTENING = "listening"  # caller's turn
    THINKING = "thinking"    # LLM request in flight
    TOOL_WAIT = "tool_wait"  # one or more scheduling-API calls in flight
    SPEAKING = "speaking"    # streaming the model's reply to TTS / playback
    EMERGENCY = "emergency"  # scripted 911 hand-off; the model is out of the loop for good
    CLOSED = "closed"        # terminal


# Phases in which caller audio should reach STT unconditionally. In every other phase the bot
# is (or is about to be) speaking, so the mic gate applies and only sustained speech gets past.
LISTENING_PHASES = frozenset({Phase.INIT, Phase.LISTENING})


@dataclass(frozen=True, slots=True)
class CallState:
    """Complete state of one call. Everything the reducer needs, nothing it doesn't."""

    call_id: str = ""
    mode: str = "local"
    phase: Phase = Phase.INIT

    # --- conversation -------------------------------------------------------------------
    # Anthropic-shaped messages (role + content blocks). The system prompt is NOT here: it is
    # adapter config, rebuilt per call so a long-lived telephony worker never crosses midnight
    # with a stale date table.
    messages: tuple[dict[str, Any], ...] = ()

    # --- in-flight work -----------------------------------------------------------------
    request_id: str | None = None      # LLM request currently streaming
    utterance_id: str | None = None    # TTS utterance currently open
    turn_text: str = ""                # all text deltas of the current request
    speech_buffer: str = ""            # text streamed but not yet handed to TTS
    turn_tool_uses: tuple[dict[str, Any], ...] = ()   # tool_use blocks from this request
    pending_tools: tuple[str, ...] = ()               # tool_call_ids still executing
    tool_results: tuple[dict[str, Any], ...] = ()     # completed results awaiting a send
    # tool_use ids that are already committed to `messages` inside an assistant message.
    # Anthropic rejects a conversation where an assistant tool_use has no matching
    # tool_result, so an interrupted turn must synthesize results for exactly these — and in
    # this order. Empty while a request is still streaming, because nothing is committed yet.
    committed_tool_ids: tuple[str, ...] = ()

    # Monotonic id counters. Ids are derived from state rather than uuid4() so that replaying a
    # trace produces byte-identical actions — a random id would make every replay comparison
    # fail for reasons that have nothing to do with the logic under test.
    request_seq: int = 0
    utterance_seq: int = 0

    # --- liveness -----------------------------------------------------------------------
    caller_present: bool = False
    bot_speaking: bool = False
    user_speaking: bool = False
    last_partial: str = ""

    # --- reasoning layer (Phase 12) ------------------------------------------------------
    # None until the classifier answers. The distinction from UNKNOWN matters: None means
    # "not asked yet" (use the full scheduling prompt, which is what this line is for),
    # while UNKNOWN means "asked, and the caller was genuinely ambiguous" (ask a question).
    intent: Intent | None = None
    intent_confidence: float = 0.0
    awaiting_clarification: bool = False

    # Set by the deterministic detector in the reducer, never by a model. Terminal: once an
    # emergency is recognized the agent does not resume booking, because "are you sure?" is
    # not a question an automated scheduler should be asking someone describing a crisis.
    emergency: bool = False
    emergency_category: str = ""

    # --- caller identity + memory (Phase 13) --------------------------------------------
    # The ANI, from the SIP participant. Empty on the local path. Present from CallerPresent
    # onwards, so it is available before the greeting finishes.
    caller_phone: str = ""
    # What the pre-greeting memory lookup found. Deliberately NOT a name: the lookup happens
    # before anyone has proved who they are, and a phone can be in anyone's hand.
    caller_known: bool = False
    upcoming_appointments: int = 0

    # The hard gate. No tool in scheduling_tools.VERIFICATION_REQUIRED_TOOLS executes while
    # this is false — the reducer refuses the call outright rather than sending it, so an
    # unverified caller's request never becomes an HTTP request at all.
    identity_verified: bool = False
    # The date of birth the API has already matched, re-sent on every subsequent PHI call so
    # the server can re-verify without the model handling it again. In memory only: never
    # logged, never traced, never spoken back.
    verified_dob: str = ""
    patient_name: str = ""
    # Dates of birth submitted by in-flight verify_identity calls, keyed by tool_call_id.
    # `verified_dob` and `patient_name` must always describe the SAME person, so the DOB that
    # gets promoted has to be the one from the call the API actually accepted — not whatever was
    # submitted most recently. See reducer._on_tool_completed.
    submitted_dobs: tuple[tuple[str, str], ...] = ()

    # The hold_id the API issued on the most recent successful `hold_slot`, re-sent verbatim on
    # `confirm_booking` so the model never has to copy it. A hold_id is a 36-character random
    # UUID with no redundancy: every character is load-bearing and none of it can be inferred,
    # which makes it the one argument a language model cannot reliably reproduce.
    #
    # Measured on a live call (trace 20260903T170114545113Z). `hold_slot` returned
    # c1154b66-3d70-4585-908c-2aa92646049f and the model sent
    # c1154b66-3d70-4585-908c-2aa92642049f to `confirm_booking` — one hex digit changed, 6 to 2.
    # The API correctly refused with a 409 the caller heard as a stumble, and the whole
    # hold-and-confirm round trip ran again: 30 seconds of the call spent re-doing work that
    # had already succeeded.
    #
    # Empty means no live hold, and then the model's own value is passed through untouched so
    # the API's existing 409 still speaks for itself. Cleared on a successful confirm so a
    # second booking in the same call cannot silently reuse a spent hold.
    active_hold_id: str = ""

    # --- counters / outcome -------------------------------------------------------------
    # One follow-through nudge per caller turn — see reducer._ACTION_CLAIM. Bounded so a model
    # that keeps promising cannot loop the engine.
    nudged: bool = False
    # Whether ANY tool was invoked during this caller turn. The nudge asks "did the model
    # promise an action and not take one?", and that question is about the caller's turn, not
    # about one request inside it: after a tool round trip the follow-up request narrates the
    # result and legitimately makes no call of its own. `turn_tool_uses` cannot answer it (it is
    # cleared at the end of every request) and neither can `committed_tool_ids` (cleared when
    # results are flushed back), which is how a successful hold came to be treated as an empty
    # promise. Reset alongside `nudged` on each new caller turn.
    turn_had_tool: bool = False

    # Phase 15. One filler line per caller turn while a tool is slow, and the counters the
    # degradation ladder runs on. All of them reset where they should: `filled` on each new
    # caller turn, the failure counters on the next SUCCESS of the same kind, because the
    # ladder asks "is this call failing repeatedly", not "has it ever failed".
    filled: bool = False
    llm_failures: int = 0
    tool_failures: int = 0
    no_match_count: int = 0
    low_confidence_count: int = 0
    # A transfer has been decided. Non-empty `transfer_reason` means the hand-off line is
    # playing and the engine takes no further turns — whatever it would say next is the
    # human's job. `transfer_fired` flips when the action has actually been handed to the
    # adapter, which happens when playback ENDS, not when the decision is made: firing a SIP
    # transfer while the agent is mid-sentence cuts the caller off in the middle of being told
    # what is about to happen, and on the emergency path that sentence is the 911 instruction.
    transfer_reason: str = ""
    transfer_urgent: bool = False
    transfer_fired: bool = False
    # The LiveKit participant identity of the caller, needed to transfer them. Present from
    # CallerPresent; empty on the local path, which is one of the reasons a transfer there
    # falls back to the callback line rather than pretending.
    caller_identity: str = ""

    turn_index: int = 0
    interruptions: int = 0
    booked: bool = False
    escalated: bool = False
    # The caller has said goodbye on a completed booking. The call does NOT end here — the
    # agent still owes them its own sign-off, and cutting the line mid-"take care" is worse
    # than the dead air this replaces. EndCall fires when that last utterance finishes playing.
    closing: bool = False

    # --- provider health (recorded in Phase 10; consumed by the Phase-15 ladder) ---------
    degraded: tuple[str, ...] = field(default_factory=tuple)

    @property
    def transferring(self) -> bool:
        """A hand-off has been decided; no further caller turns are taken."""
        return bool(self.transfer_reason)

    @property
    def busy(self) -> bool:
        """True when the engine owns the floor — generating, calling tools, or speaking."""
        return self.phase in (
            Phase.GREETING, Phase.THINKING, Phase.TOOL_WAIT, Phase.SPEAKING, Phase.EMERGENCY
        )

    @property
    def outcome(self) -> str:
        """Call outcome, using the same vocabulary as the Phase-6 metrics sink.

        A successful booking outranks an escalation: a call that hit an empty availability
        window and then found a slot on a second look is booked, not escalated.
        """
        if self.emergency:
            return "emergency"
        if self.booked:
            return "booked"
        if self.escalated:
            return "escalated"
        return "abandoned"

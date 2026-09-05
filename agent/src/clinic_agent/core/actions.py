"""Phase 10 — actions the reducer asks the adapters to perform.

An :class:`Action` is a *request*, never a result. ``reduce()`` returns them; the session's
executor hands each to the owning adapter; whatever comes back arrives as a new
:mod:`clinic_agent.core.events` event. Nothing here touches I/O, so a reducer test can assert
on the exact action sequence a call produces without any adapter existing.

Actions are compared by value in replay tests, so they are frozen dataclasses with only
plain-data fields (no callables, no adapter handles).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..intents import Intent
from .llm_router import Tier


@dataclass(frozen=True, slots=True)
class Action:
    """Base class — exists so the reducer's return type is nameable."""


@dataclass(frozen=True, slots=True)
class StartLLM(Action):
    """Begin a streaming LLM request.

    ``messages`` is the complete conversation to send (the reducer owns history, so the
    adapter is stateless and any request can be replayed in isolation). The system prompt is
    NOT in here — it is adapter-level config, rebuilt per call for date freshness.
    """

    request_id: str
    messages: tuple[dict[str, Any], ...] = ()
    # Phase 12: the reducer chooses a TIER (a dialogue decision, pure and replayable) and the
    # intent that scopes the system prompt and tool subset. The adapter maps tier -> model,
    # because that mapping reads environment variables and must stay out of the reducer.
    tier: Tier = Tier.STANDARD
    intent: Intent | None = None
    routing_reason: str = ""
    # Phase 13: per-CALL context appended to the per-INTENT system prompt — whether this number
    # is on file, and whether a date of birth has been matched. It lives on the action rather
    # than in the adapter because it is a state-derived fact, so a replayed trace reproduces
    # exactly the instructions the model was given about who it was talking to.
    context_note: str = ""


@dataclass(frozen=True, slots=True)
class CancelLLM(Action):
    """Abort an in-flight request. Cancelling an already-finished request is a no-op, which
    matters because the barge-in and completion races are genuinely simultaneous."""

    request_id: str


@dataclass(frozen=True, slots=True)
class Speak(Action):
    """Send text to TTS as part of utterance ``utterance_id``.

    Several ``Speak`` actions share one ``utterance_id``: the reducer emits each sentence as
    soon as the LLM has streamed it (see ``reducer._split_speakable``), and the TTS adapter
    appends them to a single Cartesia context so the caller hears one continuous reply rather
    than sentences with seams. ``final=True`` closes the context and flushes.

    ``deterministic`` marks speech that did not come from the model — the greeting with its
    mandatory AI disclosure, and the scripted error line. It is spoken verbatim so the
    governance wording is byte-identical every run, and it is flagged here so traces make the
    distinction visible.
    """

    utterance_id: str
    text: str
    final: bool = False
    deterministic: bool = False


@dataclass(frozen=True, slots=True)
class CancelSpeech(Action):
    """Stop synthesis and drop already-buffered playback audio immediately.

    Both halves are required: cancelling the Cartesia context alone still leaves queued PCM in
    the output device, and the caller keeps hearing the bot after interrupting it.
    """

    utterance_id: str


@dataclass(frozen=True, slots=True)
class InvokeTool(Action):
    """Execute one scheduling-API tool call."""

    tool_call_id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LoadCallerMemory(Action):
    """Look this number up, in parallel with the greeting (Phase 13).

    An action rather than something the session does on its own, so the lookup and its timing
    are visible in a replayed trace like everything else. Fire-and-forget by design: the
    greeting never waits on it.
    """

    phone: str


@dataclass(frozen=True, slots=True)
class EndCall(Action):
    """Tear the session down."""

    reason: str = "hangup"


@dataclass(frozen=True, slots=True)
class ClassifyIntent(Action):
    """Classify one utterance, off the critical path.

    Fired alongside ``StartLLM``, never before it. The dialogue turn does not wait for the
    answer; it arrives as an :class:`~clinic_agent.core.events.IntentClassified` event and
    scopes the *next* request, or re-plans the current one if it materially disagrees.
    """

    utterance: str


@dataclass(frozen=True, slots=True)
class TransferToHuman(Action):
    """Hand the caller to a person, with a reason and enough context for the handoff.

    A first-class action rather than only a failure path — an agent that knows what it cannot
    do is more useful than one that improvises.

    Phase 15 made it real: ``core/transfer.SIPTransfer`` moves the caller's SIP leg with
    LiveKit's ``TransferSIPParticipant``. Two properties of WHEN it is emitted are load-bearing
    and easy to undo:

    * the reducer emits it from ``_on_bot_stopped``, once the hand-off line has actually been
      heard — firing it at the moment the decision is made cuts the caller off mid-sentence,
      and on the emergency path that sentence is the 911 instruction;
    * if the transfer cannot happen (no ``CLINIC_TRANSFER_NUMBER``, the local path, a refusal),
      the adapter emits ``TransferFailed`` and the caller still hears a callback promise. A
      hand-off that ends in a click is worse than never having offered one.
    """

    reason: str
    summary: str = ""
    urgent: bool = False

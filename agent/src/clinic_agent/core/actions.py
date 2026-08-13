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
class EndCall(Action):
    """Tear the session down."""

    reason: str = "hangup"

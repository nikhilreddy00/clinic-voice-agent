"""Phase 10 — the pure reducer. This is the engine.

    reduce(state: CallState, event: Event) -> tuple[CallState, list[Action]]

Zero I/O. No network, no clock, no logging, no randomness. Everything time-dependent arrives
on the event (``event.t``); every identifier is derived from a counter in the state. That is
not stylistic — three things depend on it:

1. **Deterministic replay.** ``recorder.replay()`` feeds a recorded call's events back through
   this function in CI. Same events in, same state and same action sequence out, in
   milliseconds, with no audio and no API keys. It is the Phase-10 exit criterion and the
   Tier-1 eval in Phase 16.
2. **Speculative execution (Phase 14).** Because turn boundaries are decided here rather than
   emerging from a linear frame pipeline, the loop can start an LLM request on a partial
   transcript and cancel it if the caller keeps talking. Inside Pipecat's frame chain that is
   not expressible.
3. **Granular failover (Phase 15).** Provider degradation is just another event, so the
   degradation ladder becomes testable logic instead of try/except scattered across adapters.

The reducer owns conversation history, so LLM adapters are stateless and any single request is
reproducible from the action alone.
"""

from __future__ import annotations

import json
import re
from dataclasses import replace
from typing import Any

from ..intents import Intent, needs_clarification, resolve_intent
from ..prompts import EMERGENCY_RESPONSE, SYSTEM_ERROR_LINE, greeting_for
from . import events as ev
from .actions import (
    Action,
    CancelLLM,
    CancelSpeech,
    ClassifyIntent,
    EndCall,
    InvokeTool,
    Speak,
    StartLLM,
    TransferToHuman,
)
from .closing import is_farewell
from .intent import detect_emergency
from .llm_router import select_tier
from .state import CallState, Phase

# Flush a sentence to TTS as soon as its terminating punctuation is followed by whitespace —
# the whitespace is the proof the sentence is complete rather than mid-token. This is what
# recovers the streaming behavior Pipecat gave for free: the caller hears the first sentence
# while the model is still generating the second.
_BOUNDARY = re.compile(r"[.!?\n]+(?=\s)")
_WORD_BEFORE_PERIOD = re.compile(r"([A-Za-z]+)\.$")

# Periods that are not sentence ends. Without this the splitter cuts "with Dr. Aisha Patel?"
# into "with Dr." and "Aisha Patel?" — observed on the first end-to-end run of this engine.
# Every provider in the clinic is a "Dr.", so it happened on essentially every booking.
_ABBREVIATIONS = frozenset(
    {
        "dr", "mr", "mrs", "ms", "prof", "st", "rd", "ave", "blvd", "apt",
        "dept", "inc", "ltd", "jr", "sr", "vs", "etc", "approx", "fig", "min", "hr",
    }
)

# A clause with no terminal punctuation this long is flushed at a comma anyway. Without this a
# model that streams one long unpunctuated sentence produces total silence until it finishes.
_MAX_UNPUNCTUATED = 120


def _is_sentence_end(segment: str) -> bool:
    """Whether the punctuation run ending ``segment`` really terminates a sentence."""
    if not segment.endswith("."):
        return True  # '!', '?' and newlines are never ambiguous
    match = _WORD_BEFORE_PERIOD.search(segment)
    if match is None:
        return True
    word = match.group(1)
    if len(word) == 1:
        return False  # an initial, e.g. "Aisha B. Patel"
    return word.lower() not in _ABBREVIATIONS


def _split_speakable(buffer: str) -> tuple[list[str], str]:
    """Split accumulated LLM text into complete sentences plus an unfinished remainder."""
    chunks: list[str] = []
    pos = 0
    for match in _BOUNDARY.finditer(buffer):
        if match.start() < pos or not _is_sentence_end(buffer[: match.end()]):
            continue
        chunk = buffer[pos : match.end()].strip()
        if chunk:
            chunks.append(chunk)
        pos = match.end()
    remainder = buffer[pos:]

    if not chunks and len(remainder) >= _MAX_UNPUNCTUATED:
        cut = remainder.rfind(",", 0, _MAX_UNPUNCTUATED)
        if cut > 0:
            chunks.append(remainder[: cut + 1].strip())
            remainder = remainder[cut + 1 :]

    return chunks, remainder


def _text_block(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def _tool_result_block(tool_use_id: str, payload: Any) -> dict[str, Any]:
    """Anthropic tool_result block. Results are JSON-encoded because the API takes text."""
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": json.dumps(payload, default=str),
    }


def _ordered_results(state: CallState, extra: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Tool results ordered to match the assistant's tool_use blocks.

    Parallel tools finish out of order; keeping the reply aligned with the request order costs
    nothing and avoids depending on the API being lenient about it.
    """
    by_id = {b["tool_use_id"]: b for b in state.tool_results}
    by_id.update(extra)
    return [by_id[tid] for tid in state.committed_tool_ids if tid in by_id]


def _next_request(state: CallState) -> tuple[CallState, StartLLM]:
    """Allocate the next request id and build the StartLLM action for the current history.

    The tier is selected here — pure, from state — so it lands in the replayable action rather
    than being decided inside an adapter that reads environment variables.
    """
    seq = state.request_seq + 1
    request_id = f"req-{seq}"
    state = replace(
        state,
        request_seq=seq,
        request_id=request_id,
        turn_text="",
        speech_buffer="",
        turn_tool_uses=(),
        phase=Phase.THINKING,
    )
    tier, reason = select_tier(
        intent=state.intent, turn_index=state.turn_index, degraded=state.degraded
    )
    return state, StartLLM(
        request_id=request_id,
        messages=tuple(state.messages),
        tier=tier,
        intent=state.intent,
        routing_reason=reason,
    )


def _open_utterance(state: CallState) -> tuple[CallState, str]:
    seq = state.utterance_seq + 1
    utterance_id = f"utt-{seq}"
    return replace(state, utterance_seq=seq, utterance_id=utterance_id), utterance_id


def _abort_in_flight(state: CallState) -> tuple[CallState, list[Action]]:
    """Cancel whatever the engine is currently doing and leave history valid to resend.

    Called on barge-in, on a new caller utterance arriving mid-turn, and on hangup. The subtle
    part is dangling tool calls: if the assistant's tool_use message is already in ``messages``
    (``committed_tool_ids`` non-empty) every one of those ids needs a tool_result or the next
    request is rejected outright, so we synthesize cancellation results for the unfinished
    ones. If the request had not completed yet, nothing was committed and the whole partial
    turn is simply dropped — the in-flight ToolCompleted events are then ignored as stale.
    """
    actions: list[Action] = []
    if state.request_id:
        actions.append(CancelLLM(request_id=state.request_id))
    if state.utterance_id:
        actions.append(CancelSpeech(utterance_id=state.utterance_id))

    messages = state.messages
    if state.committed_tool_ids:
        cancelled = {
            tid: _tool_result_block(tid, {"ok": False, "error": "cancelled: caller interrupted"})
            for tid in state.pending_tools
        }
        blocks = _ordered_results(state, cancelled)
        if blocks:
            messages = messages + ({"role": "user", "content": blocks},)

    state = replace(
        state,
        messages=messages,
        request_id=None,
        utterance_id=None,
        turn_text="",
        speech_buffer="",
        turn_tool_uses=(),
        pending_tools=(),
        tool_results=(),
        committed_tool_ids=(),
        bot_speaking=False,
    )
    return state, actions


def _stale(state: CallState, request_id: str) -> bool:
    """True if this event belongs to a request that has already been superseded."""
    return state.request_id is None or state.request_id != request_id


def reduce(state: CallState, event: ev.Event) -> tuple[CallState, list[Action]]:
    """Advance the call by one event. Pure."""
    if state.phase is Phase.CLOSED and not isinstance(event, ev.CallStarted):
        return state, []

    handler = _HANDLERS.get(type(event))
    if handler is None:
        return state, []
    return handler(state, event)


# --- handlers ----------------------------------------------------------------------------


def _on_call_started(state: CallState, e: ev.CallStarted):
    return replace(state, call_id=e.call_id, mode=e.mode, phase=Phase.INIT), []


def _on_caller_present(state: CallState, e: ev.CallerPresent):
    """A caller is on the line — speak the greeting.

    The greeting is deterministic, never model-generated: the AI disclosure (and, on
    telephony, the call-recording consent) must be worded identically on every single call,
    which is a governance requirement, not a style preference.
    """
    if state.caller_present:
        return state, []
    state, utterance_id = _open_utterance(state)
    state = replace(state, caller_present=True, phase=Phase.GREETING)
    return state, [
        Speak(
            utterance_id=utterance_id,
            text=greeting_for(state.mode),
            final=True,
            deterministic=True,
        )
    ]


def _on_speech_started(state: CallState, e: ev.SpeechStarted):
    return replace(state, user_speaking=True), []


def _on_speech_stopped(state: CallState, e: ev.SpeechStopped):
    # The turn does not start here — it starts when STT finalizes. This event exists because
    # it is the honest start of the voice-to-voice latency clock (see metrics).
    return replace(state, user_speaking=False), []


def _on_partial(state: CallState, e: ev.PartialTranscript):
    return replace(state, last_partial=e.text), []


def _on_final_transcript(state: CallState, e: ev.FinalTranscript):
    """A caller utterance is final: check for an emergency, then generate a reply."""
    text = e.text.strip()
    if not text:
        return state, []

    # THE EMERGENCY CHECK COMES FIRST, before any model request exists. detect_emergency is a
    # pure function, so this fires identically on every call, on every model, during a provider
    # outage, and while the API is timing out — and it replays from a trace with no network.
    # Nothing below this line gets to run if it matches.
    emergency = detect_emergency(text)
    if emergency is not None:
        return _enter_emergency(state, text, emergency)

    # Arm the hang-up once the business is done and the caller signs off. Gated on `booked`
    # because ending a call is irreversible: before an appointment exists, "no thanks" is a
    # caller declining a slot, not leaving. The turn below still runs — the agent gets to say
    # goodbye, and _on_bot_stopped drops the line once that has actually played.
    if state.booked and is_farewell(text):
        state = replace(state, closing=True)

    if state.phase is Phase.EMERGENCY:
        # Already handed off. Repeat the guidance rather than resuming a booking flow — an
        # automated scheduler should not be talking someone out of calling 911.
        state, utterance_id = _open_utterance(state)
        return state, [
            Speak(
                utterance_id=utterance_id,
                text=EMERGENCY_RESPONSE,
                final=True,
                deterministic=True,
            )
        ]

    state, actions = _abort_in_flight(state)
    state = replace(
        state,
        messages=state.messages + ({"role": "user", "content": text},),
        turn_index=state.turn_index + 1,
        last_partial="",
    )
    state, start = _next_request(state)
    # Classification runs ALONGSIDE the turn, never before it. Intent is an optimization on the
    # prompt and tool surface; making the caller wait for it would spend its entire latency
    # budget on their first impression.
    return state, actions + [start, ClassifyIntent(utterance=text)]


def _enter_emergency(state: CallState, text: str, emergency):
    """Scripted 911 hand-off. No model is consulted, now or for the rest of the call."""
    state, actions = _abort_in_flight(state)
    state, utterance_id = _open_utterance(state)
    state = replace(
        state,
        phase=Phase.EMERGENCY,
        emergency=True,
        emergency_category=emergency.category,
        intent=Intent.EMERGENCY,
        intent_confidence=1.0,
        turn_index=state.turn_index + 1,
        last_partial="",
        # The utterance is deliberately NOT appended to `messages`. There is no model on this
        # path, and keeping a crisis description out of the conversation history means it never
        # reaches a provider on a later turn either.
    )
    return state, actions + [
        Speak(
            utterance_id=utterance_id,
            text=EMERGENCY_RESPONSE,
            final=True,
            deterministic=True,
        ),
        TransferToHuman(
            reason=f"emergency:{emergency.category}",
            summary=f"Caller described a possible {emergency.category} emergency.",
            urgent=True,
        ),
    ]


def _on_user_interrupted(state: CallState, e: ev.UserInterrupted):
    """Real barge-in: drop the bot's turn and hand the floor back.

    Only meaningful while the engine holds the floor. When it does not, this is echo or a
    stray VAD trigger and must be ignored — reacting would cancel nothing and desync state.
    """
    if not state.busy:
        return state, []
    state, actions = _abort_in_flight(state)
    state = replace(state, phase=Phase.LISTENING, interruptions=state.interruptions + 1)
    return state, actions


def _on_llm_started(state: CallState, e: ev.LLMStarted):
    return state, []


def _on_llm_delta(state: CallState, e: ev.LLMTextDelta):
    """Accumulate streamed text and speak each sentence the moment it is complete."""
    if _stale(state, e.request_id) or not e.text:
        return state, []

    buffer = state.speech_buffer + e.text
    chunks, remainder = _split_speakable(buffer)
    state = replace(state, turn_text=state.turn_text + e.text, speech_buffer=remainder)
    if not chunks:
        return state, []

    utterance_id = state.utterance_id
    if utterance_id is None:
        state, utterance_id = _open_utterance(state)
    return state, [Speak(utterance_id=utterance_id, text=c) for c in chunks]


def _on_tool_use(state: CallState, e: ev.LLMToolUse):
    if _stale(state, e.request_id):
        return state, []
    block = {
        "type": "tool_use",
        "id": e.tool_call_id,
        "name": e.name,
        "input": dict(e.arguments),
    }
    state = replace(
        state,
        turn_tool_uses=state.turn_tool_uses + (block,),
        pending_tools=state.pending_tools + (e.tool_call_id,),
    )
    return state, [InvokeTool(tool_call_id=e.tool_call_id, name=e.name, arguments=dict(e.arguments))]


def _on_llm_completed(state: CallState, e: ev.LLMCompleted):
    """The stream ended: commit the assistant message and flush the tail of the speech buffer."""
    if _stale(state, e.request_id):
        return state, []

    actions: list[Action] = []
    blocks: list[dict[str, Any]] = []
    if state.turn_text.strip():
        blocks.append(_text_block(state.turn_text))
    blocks.extend(state.turn_tool_uses)

    messages = state.messages
    if blocks:
        messages = messages + ({"role": "assistant", "content": blocks},)

    # Close the TTS context. Even with an empty tail this has to be sent so the adapter knows
    # no more text is coming and can flush; otherwise the last sentence sits in Cartesia's
    # buffer waiting for a continuation that never arrives.
    utterance_id = state.utterance_id
    tail = state.speech_buffer.strip()
    if tail and utterance_id is None:
        state, utterance_id = _open_utterance(state)
    if utterance_id is not None:
        actions.append(Speak(utterance_id=utterance_id, text=tail, final=True))

    committed = tuple(b["id"] for b in state.turn_tool_uses)
    state = replace(
        state,
        messages=messages,
        request_id=None,
        turn_text="",
        speech_buffer="",
        turn_tool_uses=(),
        committed_tool_ids=state.committed_tool_ids + committed,
    )

    if state.pending_tools:
        return replace(state, phase=Phase.TOOL_WAIT), actions
    if utterance_id is not None:
        return replace(state, phase=Phase.SPEAKING), actions
    return replace(state, phase=Phase.LISTENING), actions


def _on_llm_failed(state: CallState, e: ev.LLMFailed):
    """Speak a scripted apology rather than leaving the caller in dead air."""
    if _stale(state, e.request_id):
        return state, []

    state, actions = _abort_in_flight(state)
    state, utterance_id = _open_utterance(state)
    degraded = state.degraded if "llm" in state.degraded else state.degraded + ("llm",)
    state = replace(state, phase=Phase.SPEAKING, degraded=degraded)
    return state, actions + [
        Speak(utterance_id=utterance_id, text=SYSTEM_ERROR_LINE, final=True, deterministic=True)
    ]


def _on_tool_completed(state: CallState, e: ev.ToolCompleted):
    """Record a tool result; when the last one lands, send them all back to the model."""
    if e.tool_call_id not in state.pending_tools:
        return state, []  # stale result from a turn the caller already interrupted

    booked = state.booked or (e.name == "confirm_booking" and e.ok)
    # An empty availability window is the escalation trigger (a later successful booking
    # overrides it — see CallState.outcome).
    escalated = state.escalated or (
        e.name == "check_availability" and e.ok and e.result.get("count") == 0
    )

    pending = tuple(t for t in state.pending_tools if t != e.tool_call_id)
    state = replace(
        state,
        pending_tools=pending,
        tool_results=state.tool_results + (_tool_result_block(e.tool_call_id, e.result),),
        booked=booked,
        escalated=escalated,
    )
    if pending:
        return state, []

    blocks = _ordered_results(state, {})
    state = replace(
        state,
        messages=state.messages + ({"role": "user", "content": blocks},),
        tool_results=(),
        committed_tool_ids=(),
    )
    state, start = _next_request(state)
    return state, [start]


def _on_bot_started(state: CallState, e: ev.BotStartedSpeaking):
    phase = state.phase if state.phase in (Phase.GREETING, Phase.TOOL_WAIT) else Phase.SPEAKING
    return replace(state, bot_speaking=True, phase=phase), []


def _on_bot_stopped(state: CallState, e: ev.BotStoppedSpeaking):
    """Playback ended. Hand the floor back unless work is still in flight.

    TOOL_WAIT and THINKING deliberately survive this: the model can speak a sentence and then
    call a tool, and treating the end of that sentence as the end of the turn would let a
    stray transcript cancel a booking that is mid-commit.
    """
    state = replace(state, bot_speaking=False)

    # The caller said goodbye and the agent has now finished saying it back. Waiting for
    # playback to drain rather than ending on the farewell itself is the whole point: it is
    # what lets the caller actually hear "take care" before the line goes.
    if state.closing and state.phase not in (Phase.THINKING, Phase.TOOL_WAIT):
        return replace(state, phase=Phase.CLOSED, caller_present=False), [
            EndCall(reason="conversation_complete")
        ]
    if state.phase in (Phase.GREETING, Phase.SPEAKING):
        state = replace(state, phase=Phase.LISTENING, utterance_id=None)
    return state, []


def _on_intent_classified(state: CallState, e: ev.IntentClassified):
    """Record the classified intent, and re-plan only when it changes the tool surface.

    Re-planning cancels an in-flight request and pays its latency again, so it is gated on a
    change that actually matters: switching between a flow that can call the scheduling tools
    and one that cannot. A confidence nudge, or a move between two hand-off intents, changes
    nothing the caller would notice and is not worth the restart.
    """
    try:
        intent = Intent(e.intent)
    except ValueError:
        return state, []  # unknown value from a model that ignored the enum

    previous = state.intent
    # Sticky: an established intent is only replaced by a confident, genuinely different one.
    # Without this, mid-booking slot-fill answers classify as `unknown` in isolation and strip
    # the scheduling tools from the next request — see intents.resolve_intent.
    resolved = resolve_intent(previous, intent, e.confidence)
    state = replace(
        state,
        intent=resolved,
        intent_confidence=e.confidence,
        # We are awaiting clarification exactly when we still do not know what the caller
        # wants — which, once an intent is established, is never.
        awaiting_clarification=resolved is None or resolved is Intent.UNKNOWN,
    )
    intent = resolved

    if state.phase is not Phase.THINKING or state.request_id is None:
        return state, []  # nothing in flight; it scopes the next request
    if _tool_surface(previous) == _tool_surface(intent):
        return state, []

    state, actions = _abort_in_flight(state)
    state, start = _next_request(state)
    return state, actions + [start]


def _tool_surface(intent: Intent | None) -> bool:
    """Whether this intent gets the scheduling tools. The only difference worth re-planning for."""
    return intent is None or intent is Intent.SCHEDULE_APPOINTMENT


def _on_intent_failed(state: CallState, e: ev.IntentClassificationFailed):
    """Classification is an optimization, not a dependency — the turn already ran without it."""
    return state, []


def _on_model_routed(state: CallState, e: ev.ModelRouted):
    """Emitted by the adapter for trace visibility; the decision was already made."""
    return state, []


def _on_provider_degraded(state: CallState, e: ev.ProviderDegraded):
    if e.provider in state.degraded:
        return state, []
    return replace(state, degraded=state.degraded + (e.provider,)), []


def _on_hangup(state: CallState, e: ev.Hangup):
    state, actions = _abort_in_flight(state)
    return replace(state, phase=Phase.CLOSED, caller_present=False), actions + [
        EndCall(reason=e.reason)
    ]


_HANDLERS = {
    ev.CallStarted: _on_call_started,
    ev.CallerPresent: _on_caller_present,
    ev.SpeechStarted: _on_speech_started,
    ev.SpeechStopped: _on_speech_stopped,
    ev.PartialTranscript: _on_partial,
    ev.FinalTranscript: _on_final_transcript,
    ev.UserInterrupted: _on_user_interrupted,
    ev.LLMStarted: _on_llm_started,
    ev.LLMTextDelta: _on_llm_delta,
    ev.LLMToolUse: _on_tool_use,
    ev.LLMCompleted: _on_llm_completed,
    ev.LLMFailed: _on_llm_failed,
    ev.ToolCompleted: _on_tool_completed,
    ev.BotStartedSpeaking: _on_bot_started,
    ev.BotStoppedSpeaking: _on_bot_stopped,
    ev.IntentClassified: _on_intent_classified,
    ev.IntentClassificationFailed: _on_intent_failed,
    ev.ModelRouted: _on_model_routed,
    ev.ProviderDegraded: _on_provider_degraded,
    ev.Hangup: _on_hangup,
}

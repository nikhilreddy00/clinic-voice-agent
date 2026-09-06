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

from ..intents import HANDLED_INTENTS, Intent, needs_clarification, resolve_intent
from ..prompts import (
    EMERGENCY_RESPONSE,
    HANDOFF_LINE,
    HANDOFF_UNAVAILABLE_LINE,
    SYSTEM_ERROR_LINE,
    TOOL_FILLER_LINE,
    caller_context_note,
    greeting_for,
)
from ..scheduling_tools import CALLER_SCOPED_TOOLS, VERIFICATION_REQUIRED_TOOLS
from . import events as ev
from .actions import (
    Action,
    CancelLLM,
    CancelSpeech,
    ClassifyIntent,
    EndCall,
    InvokeTool,
    LoadCallerMemory,
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

# How much of a reply must exist before a comma is good enough to start speaking on.
#
# The floor is set by underrun, not by taste: the opening chunk has to take LONGER TO SPEAK
# than the next burst of model output takes to ARRIVE, or the caller hears the reply stutter
# mid-phrase. Measured 2026-09-04, gaps between text deltas are 79-102 ms p50 and ~335 ms p95,
# and ~20 characters is about a second of speech -- comfortably clear. Going much lower trades
# a silence the caller notices for a gap the caller notices, which is not a trade.
#
# A plain constant, NOT an env var, and that is deliberate: `reduce()` is pure so that a
# recorded trace replays identically anywhere, and reading the environment here would make the
# same trace produce different Speak actions on a differently-configured machine. Tuning this
# is a one-line code change and a replay diff, which is the right amount of ceremony for a
# change to how the agent sounds. `0` disables the split.
_FIRST_CLAUSE_MIN_CHARS = 20

# A clause boundary worth breathing at. The em/en dash needs no following space: models write
# "Good to hear from you—I'm happy to help", and that dash is exactly where a person pauses.
_CLAUSE_BOUNDARY = re.compile(r"[,;:](?=\s)|[\u2014\u2013]")


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


def _split_speakable(buffer: str, *, opening: bool = False) -> tuple[list[str], str]:
    """Split accumulated LLM text into complete sentences plus an unfinished remainder.

    ``opening`` means nothing has been spoken for this reply yet, which unlocks the
    clause-boundary split below. See :data:`_FIRST_CLAUSE_MIN_CHARS`.
    """
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

    # Phase 14: start speaking at the first CLAUSE of a reply rather than holding out for a
    # full sentence. Measured on the 2026-09-04 calls, the wait for a sentence-ending period
    # was 205-255 ms p50 -- most of the whole "speech queue" stage, and none of it TTS.
    #
    # Only the opening chunk, on purpose. Inside the body of a reply there is already audio
    # playing, so splitting earlier buys nothing and costs prosody; on the opening chunk the
    # alternative is silence. Every chunk still goes to the same Cartesia context with
    # `continue: true`, so the caller hears one continuous utterance either way.
    if opening and not chunks and _FIRST_CLAUSE_MIN_CHARS:
        match = _CLAUSE_BOUNDARY.search(remainder, _FIRST_CLAUSE_MIN_CHARS - 1)
        if match is not None:
            chunk = remainder[: match.end()].strip()
            if len(chunk) >= _FIRST_CLAUSE_MIN_CHARS:
                return [chunk], remainder[match.end() :]

    if not chunks and len(remainder) >= _MAX_UNPUNCTUATED:
        cut = remainder.rfind(",", 0, _MAX_UNPUNCTUATED)
        if cut > 0:
            chunks.append(remainder[: cut + 1].strip())
            remainder = remainder[cut + 1 :]

    return chunks, remainder


# A model may narrate its reasoning in <thinking> tags even when extended thinking is not
# enabled, and everything the reducer treats as text goes to TTS. On a live call one of these
# was spoken to the caller in full — including the internal hold UUID, which they heard as "a
# big number in a different format" and reasonably took for their confirmation code.
#
# Streaming makes this a state problem rather than a regex: the opening tag, the body, and the
# closing tag arrive in different deltas. So text from an OPEN tag onward is held back rather
# than spoken, and released only if a closing tag never comes (it is then dropped entirely) —
# the caller must never hear the inside of one of these.
_THINKING_BLOCK = re.compile(r"<thinking>.*?</thinking>", re.DOTALL | re.IGNORECASE)
_THINKING_OPEN = re.compile(r"<thinking>", re.IGNORECASE)
# A tag split across deltas ("…<thin"). Held one delta, then resolved either way.
_PARTIAL_TAG = re.compile(r"<[a-zA-Z/]{0,9}$")


def _strip_thinking(buffer: str) -> tuple[str, str]:
    """Split a buffer into (speakable, held-back). Pure.

    >>> _strip_thinking("Hi <thinking>secret</thinking> there")
    ('Hi  there', '')
    >>> _strip_thinking("Hi <thinking>half a thoug")
    ('Hi ', '<thinking>half a thoug')
    >>> _strip_thinking("Booking now<thin")
    ('Booking now', '<thin')
    """
    text = _THINKING_BLOCK.sub("", buffer)
    opened = _THINKING_OPEN.search(text)
    if opened:
        return text[: opened.start()], text[opened.start() :]
    partial = _PARTIAL_TAG.search(text)
    if partial:
        return text[: partial.start()], text[partial.start() :]
    return text, ""


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
        # Phase 13: what the engine knows about THIS caller that the static prompt cannot —
        # whether the number is on file, and whether a date of birth has been matched. It is
        # computed here, from state, so the model's instructions can never claim a caller is
        # verified when the gate below says they are not.
        context_note=caller_context_note(
            known=state.caller_known,
            upcoming=state.upcoming_appointments,
            verified=state.identity_verified,
            patient_name=state.patient_name,
        ),
    )


def _open_utterance(state: CallState) -> tuple[CallState, str]:
    seq = state.utterance_seq + 1
    utterance_id = f"utt-{seq}"
    return replace(state, utterance_seq=seq, utterance_id=utterance_id), utterance_id


# Phase 15 — the degradation ladder's thresholds.
#
# Two, not one: a single failure is a blip that the retry in the adapter has already absorbed,
# and the model recovers from one bad tool result perfectly well ("sorry, let me try that
# again"). Two IN A ROW is a pattern the caller is now paying for.
#
# Three for the conversation-quality signals, because those are noisier: one unintelligible
# utterance is a cough, and a caller repeating themselves twice is normal on a phone line.
_FAILURES_BEFORE_TRANSFER = 2
_NO_MATCHES_BEFORE_TRANSFER = 3

# Below this, Deepgram is guessing. Sustained low confidence means the line, the accent, or the
# background noise is beyond this stack, and no amount of asking again fixes it.
_MIN_ASR_CONFIDENCE = 0.6


def _begin_transfer(
    state: CallState,
    reason: str,
    *,
    urgent: bool = False,
    line: str | None = HANDOFF_LINE,
) -> tuple[CallState, list[Action]]:
    """Decide a hand-off: abort what is in flight, say what is about to happen, stop taking turns.

    The transfer ACTION is deliberately not returned here. It fires from `_on_bot_stopped`,
    once the line has actually played — firing it now would cut the caller off mid-sentence,
    and on the emergency path that sentence is the 911 instruction.

    ``line=None`` is for the paths that have already spoken (the emergency script) or that
    cannot speak at all (a dead TTS provider).
    """
    if state.transferring:
        return state, []
    state, actions = _abort_in_flight(state)
    if line is not None:
        state, utterance_id = _open_utterance(state)
        actions.append(
            Speak(utterance_id=utterance_id, text=line, final=True, deterministic=True)
        )
        phase = Phase.SPEAKING if state.phase is not Phase.EMERGENCY else state.phase
    else:
        phase = state.phase
    state = replace(
        state,
        transfer_reason=reason,
        transfer_urgent=urgent,
        escalated=True,
        phase=phase,
    )
    if line is None:
        # Nothing will ever play, so nothing will ever end: fire now or never.
        return _fire_transfer(state, actions)
    return state, actions


def _fire_transfer(state: CallState, actions: list[Action]) -> tuple[CallState, list[Action]]:
    """Hand the transfer to the adapter. Idempotent — a call transfers at most once."""
    if state.transfer_fired or not state.transfer_reason:
        return state, actions
    return replace(state, transfer_fired=True), actions + [
        TransferToHuman(
            reason=state.transfer_reason,
            summary=_transfer_summary(state),
            urgent=state.transfer_urgent,
        )
    ]


def _transfer_summary(state: CallState) -> str:
    """What the human needs to know, and nothing a PHI boundary would object to.

    No name, no date of birth, no symptom text. Whoever picks up has the caller on the line and
    can ask; a summary is a routing aid, and it is written to logs and traces where a name is
    exactly what must not appear.
    """
    intent = state.intent.value if state.intent else "unclassified"
    parts = [
        f"reason={state.transfer_reason}",
        f"intent={intent}",
        f"turns={state.turn_index}",
        f"verified={state.identity_verified}",
        f"booked={state.booked}",
    ]
    if state.degraded:
        parts.append(f"degraded={'+'.join(state.degraded)}")
    return ", ".join(parts)


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
    state = replace(
        state,
        caller_present=True,
        phase=Phase.GREETING,
        caller_phone=e.phone or "",
        # The LiveKit participant identity. Only ever used to transfer this caller's SIP leg;
        # it is not an identifier of the person and is never spoken or sent to a model.
        caller_identity=e.participant_id or "",
    )
    actions: list[Action] = [
        Speak(
            utterance_id=utterance_id,
            text=greeting_for(state.mode),
            final=True,
            deterministic=True,
        )
    ]
    # Ordered after the greeting deliberately: the lookup is fire-and-forget, and the caller
    # hears the disclosure whether or not the database answers.
    if e.phone:
        actions.append(LoadCallerMemory(phone=e.phone))
    return state, actions


def _on_caller_memory(state: CallState, e: ev.CallerMemoryLoaded):
    """Record what the ANI lookup found. Never changes the greeting, which has already been
    spoken and is deterministic by governance — it scopes the model's context from turn one."""
    return replace(
        state, caller_known=e.known, upcoming_appointments=e.upcoming_appointments
    ), []


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
        # Phase 15. An empty final is STT hearing something and making nothing of it. One is a
        # cough; three in a row is a line, an accent, or a room this stack cannot work with,
        # and asking a fourth time is not going to change that.
        misses = state.no_match_count + 1
        state = replace(state, no_match_count=misses)
        if misses >= _NO_MATCHES_BEFORE_TRANSFER:
            return _begin_transfer(state, "no_match")
        return state, []

    # Sustained low confidence is the same signal arriving through a different door: Deepgram
    # answered, but it is guessing. Reset on any confident turn — a single bad one mid-call is
    # normal on a phone line.
    if e.confidence is not None and e.confidence < _MIN_ASR_CONFIDENCE:
        weak = state.low_confidence_count + 1
        state = replace(state, low_confidence_count=weak)
        if weak >= _NO_MATCHES_BEFORE_TRANSFER:
            return _begin_transfer(state, "low_asr_confidence")
    else:
        state = replace(state, no_match_count=0, low_confidence_count=0)

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

    if state.transferring:
        # A hand-off is already under way. Starting a model turn here would have the agent
        # talking over the transfer it just promised. Placed AFTER the emergency block on
        # purpose: an emergency caller who keeps talking must keep hearing the 911 line.
        return state, []

    state, actions = _abort_in_flight(state)
    state = replace(
        state,
        messages=state.messages + ({"role": "user", "content": text},),
        turn_index=state.turn_index + 1,
        last_partial="",
        nudged=False,  # a fresh caller turn gets a fresh follow-through allowance
        turn_had_tool=False,
        filled=False,  # ...and a fresh allowance for one "one moment" filler
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
    actions = actions + [
        Speak(
            utterance_id=utterance_id,
            text=EMERGENCY_RESPONSE,
            final=True,
            deterministic=True,
        )
    ]
    # The 911 instruction IS the hand-off line, so no second line is spoken. The transfer fires
    # when it finishes playing (_on_bot_stopped): a SIP transfer mid-sentence would cut the
    # caller off in the middle of the single most important thing this system ever says.
    state = replace(
        state,
        transfer_reason=f"emergency:{emergency.category}",
        transfer_urgent=True,
        escalated=True,
    )
    return state, actions


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
    speakable, held = _strip_thinking(buffer)
    # `utterance_id is None` is exactly "nothing spoken for this reply yet" -- no new state
    # needed to know we are at the opening of the turn.
    chunks, remainder = _split_speakable(speakable, opening=state.utterance_id is None)
    # `held` follows `remainder` in the original stream, so order is preserved. turn_text keeps
    # the raw text: it becomes the assistant message, and the model's own history should say
    # what the model actually produced.
    state = replace(
        state, turn_text=state.turn_text + e.text, speech_buffer=remainder + held
    )
    if not chunks:
        return state, []

    utterance_id = state.utterance_id
    if utterance_id is None:
        state, utterance_id = _open_utterance(state)
    return state, [Speak(utterance_id=utterance_id, text=c) for c in chunks]


# Phrases that describe a TOOL CALL in progress. A turn that ends with one of these and no
# tool_use is the failure a caller experiences as the agent freezing: it says "I'm booking you
# right now", stops, and the line goes quiet until they say "hello?".
#
# Measured live: 100 seconds and three repetitions of that exact sentence, no tool call, nothing
# booked. The prompt has told the model not to do this since that call — in the core prompt, with
# a worked WRONG/RIGHT example — and the promptfoo suite still catches it, which is the argument
# for enforcing it here rather than asking again more loudly.
#
# It fires at most once per caller turn, never on a turn that already called a tool, and never on
# a reply that ends in a question. That last guard is not cosmetic: a false positive is NOT free.
# Starting the extra request closes the live TTS context, so the sentence being spoken is cut off
# mid-word (trace 20260904T173146047919Z, t=34.0 — "let me pull up your appointment first. What's
# your full name?" was gated behind verification, so no tool was possible and the model had
# ALREADY asked for what it needed, which is exactly what _ACTION_NUDGE demands. The caller heard
# 0.8s of a 6s sentence and said "Hello?"). A reply that asks the caller something is not leaving
# them in silence -- they have something to answer.
# The verbs are a list, and a list of verbs is always one live call behind the model's
# vocabulary. 2026-09-04: a caller asked for a prescription refill, the model had
# `request_refill` in its tool set, called nothing, and said "Got it — I'll SEND a refill
# request for ibuprofen to our staff, and they'll review it and follow up with you." The
# caller hung up believing a refill was queued. Nothing was queued, and the nudge — the exact
# safety net for this — stayed silent because "send" was not in the list. "Filing/forwarding/
# passing along" is the whole shape of the non-scheduling tools (`request_refill` files a
# staff task), so those verbs were the most important ones to have and the only ones missing.
#
# Past tense is here too, and it is the worse case: "I've sent that over" with no tool call is
# not a promise the model failed to keep, it is a completed action that never happened. A turn
# whose tool actually ran is already excluded by `turn_had_tool`, so this cannot fire on a
# legitimate read-back.
_ACTION_CLAIM = re.compile(
    r"\b("
    r"i'?m (holding|booking|checking|looking|pulling|scheduling|cancel[l]?ing|moving"
    r"|sending|submitting|filing|forwarding|putting|passing|requesting|ordering|adding)"
    r"|let me (check|look|see|pull|find|book|hold|get that|send|submit|file|forward|put)"
    r"|i'?ll (check|look|see|pull|find|book|hold|send|submit|file|forward|put|pass|request"
    r"|order|add|note|let (?:them|the team|our staff|the staff) know|make sure)"
    r"|i'?(?:ve| have) (sent|submitted|filed|forwarded|booked|cancel[l]?ed|scheduled|added"
    r"|put (?:that|it|this) (?:in|through)|passed (?:that|it|this) (?:on|along))"
    r"|one moment|hold on|bear with me|give me (a|one) (second|moment)"
    r")\b",
    re.IGNORECASE,
)

_ACTION_NUDGE = (
    "[system] Your last reply told the caller you were taking an action, but you made no tool "
    "call, so nothing happened and they are listening to silence. Either make that tool call "
    "now, or — if you still need something from them first — ask them for it directly. Do not "
    "repeat the claim."
)

# What the reducer tells the model when it refuses a PHI tool. Phrased as an instruction the
# model can act on, not an error: the recovery is to ask for the date of birth, and a bare
# "denied" invites it to apologize and stall instead.
# A credential the model has not collected yet must not become an HTTP request. Live
# 2026-09-04: the caller's opening sentence named them ("My name is Nikhil, and I want to
# cancel my appointment"), so the model called `verify_identity` immediately with
# `date_of_birth: ""`. The API correctly rejected it, but a raw `Client error '422
# Unprocessable...'` is not something a model can act on gracefully — it apologized to the
# caller for a mistake they could not see ("I apologize — let me ask that differently") and
# burned a turn. Answering with the instruction instead keeps the recovery invisible.
_MISSING_DOB_RESULT = {
    "ok": False,
    "error": (
        "date_of_birth is empty — ask the caller for their date of birth, then call "
        "verify_identity again with it. Do not apologize; you have not told them anything yet."
    ),
}

# A PARTIAL date is the same class of problem as a blank one, and a worse one to send. The API
# refuses it with the identical 403 a wrong date gets (it has to: differentiating them turns
# the endpoint into a patient-enumeration oracle), so from the model's side an incomplete
# birthday and a wrong birthday are indistinguishable — and it would tell the caller their date
# of birth does not match when in fact it never asked for the year.
#
# Answering here costs no round trip and names the actual problem.
_PARTIAL_DOB_RESULT = {
    "ok": False,
    "error": (
        "date_of_birth is incomplete — you need the month, the day AND the year. Ask the "
        "caller for the missing part, then call verify_identity again. Do not tell them their "
        "date of birth did not match; it has not been checked yet."
    ),
}


def _is_full_dob(value: str) -> bool:
    """Month, day AND year — MMDDYYYY once normalized.

    Mirrors `scheduling_api.app.db.normalize_dob` + `is_full_dob`. Duplicated rather than
    imported because the agent and the API are separate services that talk over HTTP and never
    import each other (CLAUDE.md), the same reason the percentile helpers are duplicated. The
    API is the security boundary and enforces this regardless; this copy exists only so the
    caller hears a useful question instead of a refusal.

    THE ZERO-PADDING IS NOT COSMETIC, and leaving it out is how this drifted on its first test:
    "12-8-2000" is seven digits, and a bare digit count rejects a caller who gave a complete
    date in a form the agent normalizes every day. A three-part value is padded to MM DD YYYY
    exactly as the API does; anything else keeps its digits and fails the length check, which is
    the intended outcome for "1990" and "March 1990".
    """
    parts = [g for g in re.split(r"\D+", (value or "").strip()) if g]
    if len(parts) == 3:
        month, day, year = parts
        return len(f"{month.zfill(2)}{day.zfill(2)}{year.zfill(4)}") == 8
    return len("".join(parts)) == 8

_UNVERIFIED_RESULT = {
    "ok": False,
    "error": (
        "identity not verified — call verify_identity with the caller's date of birth first, "
        "then try again"
    ),
}


def _scoped_arguments(state: CallState, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Complete a caller-scoped tool call from state (Phase 13).

    The phone number comes from the SIP ANI and the date of birth from the value the API
    already matched — never from the model. Anything the model *did* put in those fields is
    overwritten rather than merged: a caller (or a transcript that happens to contain a phone
    number) must not be able to redirect a lookup at somebody else's chart.
    """
    if name not in CALLER_SCOPED_TOOLS:
        return arguments
    scoped = dict(arguments)
    scoped["phone"] = state.caller_phone

    if name == "verify_identity":
        # NOTHING is substituted here. Both fields are credentials the caller just supplied,
        # and this is the call that tests them.
        #
        # Substituting the stashed DOB here was a live bug: the model called verify_identity
        # after collecting only the name, filling the required date field with the literal
        # string "placeholder". That stashed "placeholder" as the DOB-under-test, and the next
        # attempt — with the caller's real, correctly-transcribed date — had it substituted
        # straight back in. Every retry for the rest of the call re-sent "placeholder" and was
        # refused, so a caller with a valid appointment could never get in.
        return scoped

    if name == "confirm_booking" and state.active_hold_id:
        # See CallState.active_hold_id: the engine remembers the token, the model does not
        # retype it. Overwrite rather than merge, exactly like phone and date_of_birth — a
        # value the model got wrong is not more trustworthy than the one the API just issued.
        scoped["hold_id"] = state.active_hold_id

    if name != "confirm_booking":
        # confirm_booking collects the DOB from the caller as intake for a NEW appointment,
        # so the model's value is the right one there; everywhere else it is the verified one.
        scoped["date_of_birth"] = state.verified_dob or arguments.get("date_of_birth", "")
    if state.patient_name:
        # Afterwards it is settled, and re-sending it is what lets a caller with a withheld or
        # unknown number stay verified across the rest of the call.
        scoped["name"] = state.patient_name
    return scoped


def _on_tool_use(state: CallState, e: ev.LLMToolUse):
    if _stale(state, e.request_id):
        return state, []
    block = {
        "type": "tool_use",
        "id": e.tool_call_id,
        "name": e.name,
        "input": dict(e.arguments),
    }

    # THE GATE. A tool that discloses or changes an existing patient's record does not become
    # an HTTP request until this caller has matched a date of birth on file. The refusal is
    # synthesized here, so the tool never runs at all — there is no request to intercept, no
    # response to leak, and the whole decision is visible in a replayed trace.
    # Same mechanism as the gate below, one step earlier: synthesize the answer rather than
    # spend a round trip discovering the argument was blank.
    if e.name == "verify_identity":
        submitted_dob = str(e.arguments.get("date_of_birth") or "").strip()
        if not submitted_dob or not _is_full_dob(submitted_dob):
            state = replace(
                state,
                turn_tool_uses=state.turn_tool_uses + (block,),
                turn_had_tool=True,
                tool_results=state.tool_results + (_tool_result_block(
                    e.tool_call_id,
                    _MISSING_DOB_RESULT if not submitted_dob else _PARTIAL_DOB_RESULT,
                ),),
            )
            return state, []

    if e.name in VERIFICATION_REQUIRED_TOOLS and not state.identity_verified:
        state = replace(
            state,
            turn_tool_uses=state.turn_tool_uses + (block,),
            turn_had_tool=True,
            tool_results=state.tool_results
            + (_tool_result_block(e.tool_call_id, _UNVERIFIED_RESULT),),
        )
        return state, []

    arguments = _scoped_arguments(state, e.name, dict(e.arguments))

    # Stash the date of birth being submitted, TAGGED WITH THIS CALL. Nothing is promoted here:
    # only the API can say a date matched, so this is inert until a ToolCompleted says so.
    #
    # It used to stash into `verified_dob` directly, and only while unverified — an
    # anti-tampering rule that stopped a later attempt from overwriting a matched DOB. The rule
    # was right and the mechanism was wrong: a SUCCESSFUL second verification still updated
    # `patient_name` (that happens on the result), so the two fields ended up describing
    # different people. On a shared handset that is a cross-person disclosure. Measured: verify
    # as Joe, then verify as Nick, and the engine reports the caller as Nick while every PHI
    # call still carries Joe's date of birth and reads Joe's chart.
    #
    # Keying by tool_call_id keeps the protection (a FAILED attempt promotes nothing) without
    # the inconsistency (a SUCCEEDED one promotes name and date together).
    submitted = state.submitted_dobs
    if e.name == "verify_identity":
        submitted = submitted + (
            (e.tool_call_id, str(arguments.get("date_of_birth") or "")),
        )

    state = replace(
        state,
        turn_tool_uses=state.turn_tool_uses + (block,),
        turn_had_tool=True,
        pending_tools=state.pending_tools + (e.tool_call_id,),
        submitted_dobs=submitted,
    )
    return state, [InvokeTool(tool_call_id=e.tool_call_id, name=e.name, arguments=arguments)]


def _on_llm_completed(state: CallState, e: ev.LLMCompleted):
    """The stream ended: commit the assistant message and flush the tail of the speech buffer."""
    if _stale(state, e.request_id):
        return state, []

    actions: list[Action] = []
    blocks: list[dict[str, Any]] = []
    spoken = state.turn_text  # captured before the reset below clears it
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
    # An unclosed <thinking> block at the end of a turn is dropped, not spoken.
    tail = _strip_thinking(state.speech_buffer)[0].strip()
    if tail and utterance_id is None:
        state, utterance_id = _open_utterance(state)
    if utterance_id is not None:
        actions.append(Speak(utterance_id=utterance_id, text=tail, final=True))

    committed = tuple(b["id"] for b in state.turn_tool_uses)
    state = replace(
        state,
        # A completed request clears the streak: the ladder asks "is this call failing
        # repeatedly", not "has anything ever failed on it".
        llm_failures=0,
        messages=messages,
        request_id=None,
        turn_text="",
        speech_buffer="",
        turn_tool_uses=(),
        committed_tool_ids=state.committed_tool_ids + committed,
    )

    if state.pending_tools:
        return replace(state, phase=Phase.TOOL_WAIT), actions

    reply = spoken or e.text or ""
    if (
        not committed
        and not state.turn_had_tool
        and not state.nudged
        and not reply.rstrip().endswith("?")
        and _has_tools(state.intent)
        and _ACTION_CLAIM.search(reply)
    ):
        # Announced an action, took none. Give the model exactly one chance to follow through
        # before the caller is left waiting on a promise nothing is going to keep.
        state = replace(
            state,
            nudged=True,
            messages=state.messages + ({"role": "user", "content": _ACTION_NUDGE},),
        )
        state, start = _next_request(state)
        return state, actions + [start]

    if state.tool_results:
        # Every tool this turn was refused by the gate above, so no ToolCompleted will ever
        # arrive to drain the turn. Send the refusals back now; without this the call stalls
        # in silence with an assistant tool_use that has no matching result.
        state, start = _flush_tool_results(state)
        return state, actions + [start]
    if utterance_id is not None:
        return replace(state, phase=Phase.SPEAKING), actions
    return replace(state, phase=Phase.LISTENING), actions


def _has_tools(intent: Intent | None) -> bool:
    """Whether this intent has any tool the model could have called.

    Phase 15. The follow-through nudge asks "you announced an action — why did you not call a
    tool?", and on a tool-less intent the answer is "because there is none", every time. On
    SPEAK_TO_HUMAN it produced the worst version of that: the model correctly said "I'm passing
    you to a staff member", got nudged, and talked itself out of it — "I don't have the ability
    to transfer calls" — which as of this phase is also FALSE. The hand-off is performed by the
    engine, not by a tool, so there is nothing here for a nudge to fix.
    """
    return intent is None or intent in HANDLED_INTENTS


def _on_llm_failed(state: CallState, e: ev.LLMFailed):
    """Speak a scripted apology rather than leaving the caller in dead air."""
    if _stale(state, e.request_id):
        return state, []

    degraded = state.degraded if "llm" in state.degraded else state.degraded + ("llm",)
    failures = state.llm_failures + 1
    state = replace(state, degraded=degraded, llm_failures=failures)

    if failures >= _FAILURES_BEFORE_TRANSFER:
        # The scripted apology asks the caller to repeat themselves. Asking twice, when the
        # model is what failed both times, is how a caller ends up saying the same sentence
        # four times to a system that was never going to answer.
        return _begin_transfer(state, "llm_unavailable")

    state, actions = _abort_in_flight(state)
    state, utterance_id = _open_utterance(state)
    state = replace(state, phase=Phase.SPEAKING)
    return state, actions + [
        Speak(utterance_id=utterance_id, text=SYSTEM_ERROR_LINE, final=True, deterministic=True)
    ]


def _on_tool_completed(state: CallState, e: ev.ToolCompleted):
    """Record a tool result; when the last one lands, send them all back to the model."""
    if e.tool_call_id not in state.pending_tools:
        return state, []  # stale result from a turn the caller already interrupted

    booked = state.booked or (e.name == "confirm_booking" and e.ok)

    # Track the live hold so `_scoped_arguments` can re-send it verbatim. A successful confirm
    # spends it; a failed one does not, so the model can retry the same hold rather than
    # re-holding a slot it already owns.
    active_hold_id = state.active_hold_id
    if e.name == "hold_slot" and e.ok:
        active_hold_id = str(e.result.get("hold_id") or "")
    elif e.name == "confirm_booking" and e.ok:
        active_hold_id = ""

    # The gate opens HERE and nowhere else: only the API can say a date of birth matched, and
    # the DOB it matched is kept so every later PHI call re-sends it for server-side re-checks.
    # `verified_dob` never leaves this process and is not written to the trace or the logs.
    verified = state.identity_verified
    verified_dob = state.verified_dob
    patient_name = state.patient_name
    submitted_dobs = tuple(pair for pair in state.submitted_dobs if pair[0] != e.tool_call_id)
    if e.name == "verify_identity" and e.ok:
        # Promote the name and the date of birth from the SAME call — the one the API just
        # accepted. The API does not echo a DOB back (nothing should return the verification
        # secret), so it comes from what was submitted with this tool_call_id.
        #
        # Both together or neither: a caller on a shared handset may legitimately verify as a
        # second person mid-call, and when they do, every later PHI call must read that
        # person's chart, not the previous one's. A FAILED attempt still promotes nothing.
        verified = True
        patient_name = str(e.result.get("name") or "")
        matched = next((dob for tid, dob in state.submitted_dobs if tid == e.tool_call_id), "")
        if matched:
            verified_dob = matched
    # An empty availability window is the escalation trigger (a later successful booking
    # overrides it — see CallState.outcome).
    escalated = state.escalated or (
        e.name == "check_availability" and e.ok and e.result.get("count") == 0
    )

    # Phase 15. Only a tool the system could not SERVE counts toward the ladder: no status at
    # all (a timeout or a refused connection) or a 5xx. A 403, a 404 and a 409 are answers —
    # a wrong date of birth, a cancelled appointment, a slot someone else just took — and the
    # model handles every one of them in dialogue. Counting them would transfer a caller to a
    # human for mistyping their birthday twice.
    unserved = not e.ok and (e.http_status is None or e.http_status >= 500)
    tool_failures = state.tool_failures + 1 if unserved else 0

    pending = tuple(t for t in state.pending_tools if t != e.tool_call_id)
    state = replace(
        state,
        pending_tools=pending,
        tool_results=state.tool_results + (_tool_result_block(e.tool_call_id, e.result),),
        booked=booked,
        escalated=escalated,
        identity_verified=verified,
        verified_dob=verified_dob,
        patient_name=patient_name,
        submitted_dobs=submitted_dobs,
        active_hold_id=active_hold_id,
        tool_failures=tool_failures,
    )
    if pending:
        return state, []

    if tool_failures >= _FAILURES_BEFORE_TRANSFER:
        # The scheduling system is not answering. Letting the model apologize a third time and
        # ask the caller to try again is how a call ends in "abandoned" with nothing booked and
        # nobody told.
        return _begin_transfer(state, "scheduling_unavailable")

    state, start = _flush_tool_results(state)
    return state, [start]


def _flush_tool_results(state: CallState) -> tuple[CallState, StartLLM]:
    """Hand every completed tool result back to the model and start the follow-up request."""
    blocks = _ordered_results(state, {})
    state = replace(
        state,
        messages=state.messages + ({"role": "user", "content": blocks},),
        tool_results=(),
        committed_tool_ids=(),
    )
    return _next_request(state)


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

    # Phase 15. The hand-off line has now actually been heard, so the transfer can go. Doing it
    # any earlier cuts the caller off mid-sentence — on the emergency path, mid-911-instruction.
    if state.transferring and not state.transfer_fired:
        return _fire_transfer(state, [])

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

    # A caller asking for a person is an escalation, not a topic. It used to be nothing but a
    # prompt fragment: the model said it would pass them to staff, and the engine recorded
    # neither the request nor the outcome — so a call that ended in "I want a human" was
    # indistinguishable from one that simply stopped. TransferToHuman marks the call escalated
    # and carries the reason into the Phase-15 warm handoff.
    if intent is Intent.SPEAK_TO_HUMAN and previous is not Intent.SPEAK_TO_HUMAN:
        # Phase 15: this is now a real hand-off rather than a marker. The in-flight model turn
        # is abandoned on purpose — a caller who has asked for a person does not want to hear
        # the agent finish its sentence first.
        return _begin_transfer(state, "caller_requested_human")

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


def _on_tool_slow(state: CallState, e: ev.ToolSlow):
    """A tool is taking long enough that the caller is sitting in silence — say something.

    Three guards, all of them load-bearing:

    * only while tools are actually in flight (a result that landed a millisecond before this
      event must not produce a filler for work that is already done);
    * only when the agent is not already speaking, or the filler talks over the sentence that
      preceded the tool call;
    * once per caller turn, because two "one moment"s in a row is worse than the silence.
    """
    if state.phase is not Phase.TOOL_WAIT or e.tool_call_id not in state.pending_tools:
        return state, []
    if state.filled or state.bot_speaking:
        return state, []
    state, utterance_id = _open_utterance(state)
    return replace(state, filled=True), [
        Speak(utterance_id=utterance_id, text=TOOL_FILLER_LINE, final=True, deterministic=True)
    ]


def _on_provider_degraded(state: CallState, e: ev.ProviderDegraded):
    """Record the provider, and hand off when one of them is not coming back.

    ``fatal`` means the adapter has exhausted its reconnects. A call with no STT cannot hear
    the caller and a call with no TTS cannot answer them; either way there is nothing left for
    the agent to do that a person would not do better, and the alternative is a caller talking
    into a line that will never respond.
    """
    if e.provider not in state.degraded:
        state = replace(state, degraded=state.degraded + (e.provider,))
    if not e.fatal:
        return state, []
    # With TTS gone there is no way to speak the hand-off line, so the transfer goes straight
    # out. With STT gone the caller can still HEAR, so they get told what is happening.
    return _begin_transfer(
        state,
        f"{e.provider}_unavailable",
        line=None if e.provider == "tts" else HANDOFF_LINE,
    )


def _on_provider_recovered(state: CallState, e: ev.ProviderRecovered):
    """A provider is serving again — stop treating the rest of the call as degraded.

    Without this, `degraded` only ever grew: one dropped socket at turn two downgraded the
    model tier for the remaining twenty turns of a call that had been healthy since turn three.
    """
    if e.provider not in state.degraded:
        return state, []
    return replace(state, degraded=tuple(p for p in state.degraded if p != e.provider)), []


def _on_transfer_failed(state: CallState, e: ev.TransferFailed):
    """Nobody to transfer to. Say so, promise the callback, and end the call cleanly.

    The promise is one the clinic can keep: the escalation is in the trace and the metrics with
    the caller's number, which is the whole reason this is a spoken line and not a silent
    hangup.
    """
    if state.emergency:
        # The caller has already been told to hang up and call 911. "Someone will call you back
        # as soon as they can" is the wrong thing to say to them and the wrong thing to have
        # them wait for. The line closes instead.
        return replace(state, phase=Phase.CLOSED, caller_present=False), [
            EndCall(reason="emergency_handoff")
        ]

    state, utterance_id = _open_utterance(state)
    return replace(state, closing=True, phase=Phase.SPEAKING), [
        Speak(
            utterance_id=utterance_id,
            text=HANDOFF_UNAVAILABLE_LINE,
            final=True,
            deterministic=True,
        )
    ]


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
    ev.CallerMemoryLoaded: _on_caller_memory,
    ev.LLMToolUse: _on_tool_use,
    ev.LLMCompleted: _on_llm_completed,
    ev.LLMFailed: _on_llm_failed,
    ev.ToolCompleted: _on_tool_completed,
    ev.BotStartedSpeaking: _on_bot_started,
    ev.BotStoppedSpeaking: _on_bot_stopped,
    ev.IntentClassified: _on_intent_classified,
    ev.IntentClassificationFailed: _on_intent_failed,
    ev.ModelRouted: _on_model_routed,
    ev.ToolSlow: _on_tool_slow,
    ev.ProviderDegraded: _on_provider_degraded,
    ev.ProviderRecovered: _on_provider_recovered,
    ev.TransferFailed: _on_transfer_failed,
    ev.Hangup: _on_hangup,
}

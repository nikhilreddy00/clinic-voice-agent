"""Phase 10 — unit tests for the pure reducer.

These are the tests that replace "run the pipeline and listen to it". Because ``reduce()`` is
pure, a full booking conversation — greeting, slot fill, two tool calls, barge-in, hangup —
is an ordinary list of dataclasses and a list of assertions, running in microseconds with no
audio devices, no network, and no API keys.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from clinic_agent.core import events as ev
from clinic_agent.core.actions import (
    CancelLLM,
    CancelSpeech,
    ClassifyIntent,
    EndCall,
    InvokeTool,
    Speak,
    StartLLM,
)
from clinic_agent.core.llm_router import Tier
from clinic_agent.core.reducer import _split_speakable, reduce
from clinic_agent.core.state import CallState, Phase
from clinic_agent.intents import Intent
from clinic_agent.prompts import GREETING, TELEPHONY_GREETING


class Driver:
    """Applies events in order, assigning seq/t, and keeps every action produced."""

    def __init__(self, state: CallState | None = None) -> None:
        self.state = state or CallState()
        self.actions: list = []
        self._seq = 0

    def send(self, event: ev.Event) -> list:
        from dataclasses import replace as _replace

        self._seq += 1
        event = _replace(event, seq=self._seq, t=float(self._seq) * 0.1)
        self.state, produced = reduce(self.state, event)
        self.actions.extend(produced)
        return produced


def _greeted(mode: str = "local") -> Driver:
    """A driver past the greeting and listening for the caller."""
    d = Driver()
    d.send(ev.CallStarted(call_id="c1", mode=mode))
    d.send(ev.CallerPresent())
    d.send(ev.BotStartedSpeaking(utterance_id="utt-1"))
    d.send(ev.BotStoppedSpeaking(utterance_id="utt-1"))
    return d


# --- greeting -----------------------------------------------------------------------------


def test_greeting_is_deterministic_and_mode_specific():
    """The AI disclosure must be spoken verbatim, and telephony must add recording consent."""
    d = Driver()
    d.send(ev.CallStarted(call_id="c1", mode="telephony"))
    produced = d.send(ev.CallerPresent())

    assert produced == [
        Speak(utterance_id="utt-1", text=TELEPHONY_GREETING, final=True, deterministic=True)
    ]
    assert d.state.phase is Phase.GREETING

    local = Driver()
    local.send(ev.CallStarted(call_id="c2", mode="local"))
    assert local.send(ev.CallerPresent())[0].text == GREETING


def test_second_caller_present_does_not_re_greet():
    d = _greeted()
    assert d.send(ev.CallerPresent()) == []


def test_greeting_end_hands_the_floor_to_the_caller():
    assert _greeted().state.phase is Phase.LISTENING


# --- the ordinary turn --------------------------------------------------------------------


def test_final_transcript_starts_an_llm_request_with_full_history():
    d = _greeted()
    produced = d.send(ev.FinalTranscript(text="I'd like to book an appointment.", confidence=0.98))

    assert produced == [
        StartLLM(
            request_id="req-1",
            messages=({"role": "user", "content": "I'd like to book an appointment."},),
            tier=Tier.STANDARD,
            intent=None,
            routing_reason="intent not yet known",
        ),
        # Phase 12: classification is fired alongside the turn, never before it. Its position
        # here is the contract — a caller must not wait on a side model for their first reply.
        ClassifyIntent(utterance="I'd like to book an appointment."),
    ]
    assert d.state.phase is Phase.THINKING
    assert d.state.turn_index == 1


def test_empty_transcript_is_ignored():
    d = _greeted()
    assert d.send(ev.FinalTranscript(text="   ")) == []
    assert d.state.phase is Phase.LISTENING


def test_sentences_stream_to_tts_as_they_complete():
    """The first sentence must reach TTS before the model has finished the second."""
    d = _greeted()
    d.send(ev.FinalTranscript(text="hello"))

    assert d.send(ev.LLMTextDelta(request_id="req-1", text="Happy to help. ")) == [
        Speak(utterance_id="utt-2", text="Happy to help.")
    ]
    # A partial sentence is held back — it is not yet speakable.
    assert d.send(ev.LLMTextDelta(request_id="req-1", text="What is your")) == []
    assert d.send(ev.LLMTextDelta(request_id="req-1", text=" name? ")) == [
        Speak(utterance_id="utt-2", text="What is your name?")
    ]


def test_completion_flushes_the_tail_and_commits_the_assistant_message():
    d = _greeted()
    d.send(ev.FinalTranscript(text="hello"))
    d.send(ev.LLMTextDelta(request_id="req-1", text="Happy to help. What is your name?"))
    produced = d.send(
        ev.LLMCompleted(request_id="req-1", text="", stop_reason="end_turn")
    )

    assert produced == [Speak(utterance_id="utt-2", text="What is your name?", final=True)]
    assert d.state.phase is Phase.SPEAKING
    assert d.state.messages[-1] == {
        "role": "assistant",
        "content": [{"type": "text", "text": "Happy to help. What is your name?"}],
    }


def test_events_from_a_superseded_request_are_ignored():
    """Cancellation and completion race constantly; stale deltas must not reopen a dead turn."""
    d = _greeted()
    d.send(ev.FinalTranscript(text="hello"))
    d.send(ev.FinalTranscript(text="actually, wait"))  # cancels req-1, starts req-2

    assert d.state.request_id == "req-2"
    assert d.send(ev.LLMTextDelta(request_id="req-1", text="stale text. ")) == []
    assert d.send(ev.LLMCompleted(request_id="req-1", stop_reason="end_turn")) == []
    assert d.state.phase is Phase.THINKING


# --- tool loop ----------------------------------------------------------------------------


def test_tool_call_round_trip_resends_history_with_the_result():
    d = _greeted()
    d.send(ev.FinalTranscript(text="anything tomorrow?"))
    produced = d.send(
        ev.LLMToolUse(
            request_id="req-1",
            tool_call_id="tu_1",
            name="check_availability",
            arguments={"date": "2026-08-14"},
        )
    )
    assert produced == [
        InvokeTool(tool_call_id="tu_1", name="check_availability", arguments={"date": "2026-08-14"})
    ]

    d.send(ev.LLMCompleted(request_id="req-1", stop_reason="tool_use"))
    assert d.state.phase is Phase.TOOL_WAIT

    result = {"ok": True, "count": 1, "slots": [{"slot_id": 7}]}
    produced = d.send(
        ev.ToolCompleted(tool_call_id="tu_1", name="check_availability", result=result, ok=True)
    )

    assert len(produced) == 1 and isinstance(produced[0], StartLLM)
    assert produced[0].request_id == "req-2"
    assert d.state.phase is Phase.THINKING
    assert d.state.messages[-1] == {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "tu_1", "content": json.dumps(result)}
        ],
    }


def test_parallel_tool_results_are_ordered_to_match_the_requests():
    """Results are returned in tool_use order even when the calls finish out of order."""
    d = _greeted()
    d.send(ev.FinalTranscript(text="check two days"))
    d.send(ev.LLMToolUse(request_id="req-1", tool_call_id="a", name="check_availability"))
    d.send(ev.LLMToolUse(request_id="req-1", tool_call_id="b", name="check_availability"))
    d.send(ev.LLMCompleted(request_id="req-1", stop_reason="tool_use"))

    assert d.send(ev.ToolCompleted(tool_call_id="b", name="check_availability", ok=True)) == []
    d.send(ev.ToolCompleted(tool_call_id="a", name="check_availability", ok=True))

    ids = [b["tool_use_id"] for b in d.state.messages[-1]["content"]]
    assert ids == ["a", "b"]


def test_confirm_booking_success_sets_the_booked_outcome():
    d = _greeted()
    d.send(ev.FinalTranscript(text="yes"))
    d.send(ev.LLMToolUse(request_id="req-1", tool_call_id="tu_1", name="confirm_booking"))
    d.send(ev.LLMCompleted(request_id="req-1", stop_reason="tool_use"))
    d.send(
        ev.ToolCompleted(
            tool_call_id="tu_1",
            name="confirm_booking",
            result={"ok": True, "confirmation_id": "A1B2C3D4"},
            ok=True,
        )
    )
    assert d.state.booked is True
    assert d.state.outcome == "booked"


def test_empty_availability_marks_escalation_but_a_later_booking_wins():
    d = _greeted()
    d.send(ev.FinalTranscript(text="anything friday?"))
    d.send(ev.LLMToolUse(request_id="req-1", tool_call_id="tu_1", name="check_availability"))
    d.send(ev.LLMCompleted(request_id="req-1", stop_reason="tool_use"))
    d.send(
        ev.ToolCompleted(
            tool_call_id="tu_1",
            name="check_availability",
            result={"ok": True, "count": 0, "slots": []},
            ok=True,
        )
    )
    assert d.state.escalated is True
    assert d.state.outcome == "escalated"

    d.send(ev.LLMToolUse(request_id="req-2", tool_call_id="tu_2", name="confirm_booking"))
    d.send(ev.LLMCompleted(request_id="req-2", stop_reason="tool_use"))
    d.send(ev.ToolCompleted(tool_call_id="tu_2", name="confirm_booking", ok=True))
    assert d.state.outcome == "booked"


# --- barge-in and interruption safety -----------------------------------------------------


def test_barge_in_cancels_generation_and_playback():
    d = _greeted()
    d.send(ev.FinalTranscript(text="hello"))
    d.send(ev.LLMTextDelta(request_id="req-1", text="Let me tell you about our hours. "))
    d.send(ev.BotStartedSpeaking(utterance_id="utt-2"))

    produced = d.send(ev.UserInterrupted())

    assert produced == [CancelLLM(request_id="req-1"), CancelSpeech(utterance_id="utt-2")]
    assert d.state.phase is Phase.LISTENING
    assert d.state.interruptions == 1
    assert d.state.bot_speaking is False


def test_barge_in_while_idle_is_ignored():
    """With the floor already the caller's there is nothing to cancel — reacting would desync."""
    d = _greeted()
    assert d.send(ev.UserInterrupted()) == []
    assert d.state.interruptions == 0


def test_interrupting_a_tool_call_keeps_the_conversation_sendable():
    """A committed tool_use with no tool_result makes the next API call fail outright.

    So an interruption mid-tool must synthesize a cancellation result for every outstanding
    call. This is the single nastiest state in the loop and the reason tool ids are tracked
    separately from the pending set.
    """
    d = _greeted()
    d.send(ev.FinalTranscript(text="anything tomorrow?"))
    d.send(ev.LLMToolUse(request_id="req-1", tool_call_id="tu_1", name="check_availability"))
    d.send(ev.LLMCompleted(request_id="req-1", stop_reason="tool_use"))
    assert d.state.phase is Phase.TOOL_WAIT

    d.send(ev.UserInterrupted())

    assistant, tool_reply = d.state.messages[-2], d.state.messages[-1]
    assert assistant["content"][0]["type"] == "tool_use"
    assert tool_reply["role"] == "user"
    assert tool_reply["content"][0]["tool_use_id"] == "tu_1"
    assert "cancelled" in tool_reply["content"][0]["content"]
    assert d.state.pending_tools == ()

    # And the late result from that abandoned call changes nothing.
    assert d.send(ev.ToolCompleted(tool_call_id="tu_1", name="check_availability", ok=True)) == []


def test_interrupting_before_completion_drops_the_partial_turn_entirely():
    """Nothing was committed to history yet, so there is nothing to reconcile."""
    d = _greeted()
    d.send(ev.FinalTranscript(text="anything tomorrow?"))
    d.send(ev.LLMToolUse(request_id="req-1", tool_call_id="tu_1", name="check_availability"))
    before = len(d.state.messages)

    d.send(ev.UserInterrupted())

    assert len(d.state.messages) == before
    assert d.state.pending_tools == ()


def test_bot_stopped_speaking_does_not_end_a_turn_that_is_still_calling_tools():
    """The model can speak a sentence and then call a tool; that is one turn, not two."""
    d = _greeted()
    d.send(ev.FinalTranscript(text="book me in"))
    d.send(ev.LLMTextDelta(request_id="req-1", text="One moment. "))
    d.send(ev.LLMToolUse(request_id="req-1", tool_call_id="tu_1", name="hold_slot"))
    d.send(ev.LLMCompleted(request_id="req-1", stop_reason="tool_use"))
    d.send(ev.BotStartedSpeaking(utterance_id="utt-2"))
    d.send(ev.BotStoppedSpeaking(utterance_id="utt-2"))

    assert d.state.phase is Phase.TOOL_WAIT


# --- failure and teardown -----------------------------------------------------------------


def test_llm_failure_speaks_a_scripted_line_instead_of_dead_air():
    d = _greeted()
    d.send(ev.FinalTranscript(text="hello"))
    produced = d.send(ev.LLMFailed(request_id="req-1", error="503"))

    speak = [a for a in produced if isinstance(a, Speak)]
    assert len(speak) == 1 and speak[0].deterministic and speak[0].final
    assert d.state.phase is Phase.SPEAKING
    assert d.state.degraded == ("llm",)


def test_hangup_closes_the_call_and_ignores_everything_after():
    d = _greeted()
    d.send(ev.FinalTranscript(text="hello"))
    produced = d.send(ev.Hangup(reason="caller_left"))

    assert EndCall(reason="caller_left") in produced
    assert d.state.phase is Phase.CLOSED
    assert d.send(ev.FinalTranscript(text="are you there?")) == []


# --- sentence splitting -------------------------------------------------------------------


@pytest.mark.parametrize(
    "buffer,chunks,remainder",
    [
        ("Hello there. ", ["Hello there."], " "),
        ("Hello there.", [], "Hello there."),          # no trailing space: may still be growing
        ("One. Two! Three? ", ["One.", "Two!", "Three?"], " "),
        ("Partial text", [], "Partial text"),
    ],
)
def test_split_speakable(buffer, chunks, remainder):
    assert _split_speakable(buffer) == (chunks, remainder)


@pytest.mark.parametrize(
    "buffer,chunks,remainder",
    [
        # The bug this guards: every provider is a "Dr.", so this fired on nearly every call,
        # splitting "with Dr. Aisha Patel?" into two utterances mid-name.
        (
            "How about Friday at 10:30 with Dr. Aisha Patel? ",
            ["How about Friday at 10:30 with Dr. Aisha Patel?"],
            " ",
        ),
        ("Dr. Chen is open. ", ["Dr. Chen is open."], " "),
        ("You'll see Aisha B. Patel on Friday. ", ["You'll see Aisha B. Patel on Friday."], " "),
        ("Arrive 15 min. early. ", ["Arrive 15 min. early."], " "),
        # A real boundary immediately after an abbreviation-free word still splits.
        ("See Dr. Chen. Anything else? ", ["See Dr. Chen.", "Anything else?"], " "),
    ],
)
def test_split_speakable_does_not_break_on_abbreviations(buffer, chunks, remainder):
    assert _split_speakable(buffer) == (chunks, remainder)


def test_long_unpunctuated_text_still_gets_flushed():
    """Otherwise a model streaming one long clause produces total silence until it finishes."""
    buffer = "we have an opening with Doctor Chen on Monday morning, " + "x" * 100
    chunks, remainder = _split_speakable(buffer)
    assert chunks == ["we have an opening with Doctor Chen on Monday morning,"]
    assert remainder.strip() == "x" * 100


# --- the agent ends the call (the 16 seconds of dead air on the second booked call) ----------


def test_the_agent_hangs_up_after_the_caller_says_goodbye():
    """Before this, EndCall had one producer — the caller hanging up.

    Live call: the agent said "take care, and we'll see you soon", then the line sat open for
    16 seconds until the caller gave up and hung up themselves.
    """
    d = _greeted()
    d.state = replace(d.state, booked=True)

    produced = d.send(ev.FinalTranscript(text="No. Thank you."))
    assert d.state.closing is True
    assert not [a for a in produced if isinstance(a, EndCall)], (
        "hung up on the farewell itself — the caller never hears the agent's goodbye"
    )

    # The agent speaks its sign-off, and the line drops only once that has finished playing.
    d.send(ev.LLMCompleted(request_id=d.state.request_id, text="Take care!", stop_reason="end_turn"))
    d.send(ev.BotStartedSpeaking(utterance_id="utt-2"))
    produced = d.send(ev.BotStoppedSpeaking(utterance_id="utt-2", completed=True))

    assert [a for a in produced if isinstance(a, EndCall)], "the call never ended"
    assert d.state.phase is Phase.CLOSED


def test_a_farewell_before_a_booking_does_not_end_the_call():
    """"No thanks" while slots are being offered is declining a time, not leaving."""
    d = _greeted()
    assert d.state.booked is False

    d.send(ev.FinalTranscript(text="No. Thank you."))
    assert d.state.closing is False
    assert d.state.phase is not Phase.CLOSED


# --- the hold_id the model must never retype (live double-round-trip fix) --------------------


def _turn(d: Driver, text: str = "go ahead") -> str:
    """Open a caller turn and return the request id the engine is now serving."""
    d.send(ev.FinalTranscript(text=text))
    assert d.state.request_id is not None
    return d.state.request_id


def _held(d: Driver, hold_id: str, slot_id: int = 25, tid: str | None = None) -> None:
    """Run a successful hold_slot through the engine."""
    tid = tid or f"t-hold-{slot_id}"
    rid = _turn(d)
    d.send(ev.LLMToolUse(request_id=rid, tool_call_id=tid,
                         name="hold_slot", arguments={"slot_id": slot_id}))
    d.send(ev.LLMCompleted(request_id=rid, stop_reason="tool_use"))
    d.send(ev.ToolCompleted(tool_call_id=tid, name="hold_slot", ok=True,
                            result={"ok": True, "hold_id": hold_id, "slot_id": slot_id},
                            latency_ms=1.0, http_status=200))


def _confirm(d: Driver, hold_id: str, tid: str = "t-c") -> InvokeTool:
    rid = d.state.request_id or _turn(d)
    produced = d.send(ev.LLMToolUse(
        request_id=rid, tool_call_id=tid, name="confirm_booking",
        arguments={"hold_id": hold_id, "patient_name": "Nick",
                   "date_of_birth": "05/08/2003", "reason": "back pain"},
    ))
    return next(a for a in produced if isinstance(a, InvokeTool))


def test_a_corrupted_hold_id_is_overwritten_with_the_one_the_api_issued():
    """The live defect: one hex digit changed while copying a 36-char UUID.

    hold_slot issued  c1154b66-3d70-4585-908c-2aa92646049f
    the model sent    c1154b66-3d70-4585-908c-2aa92642049f   (6 -> 2)

    The API refused with a 409 and the agent re-held and re-confirmed, burning 30 seconds of
    the call. The model should never have been holding the token in the first place.
    """
    issued = "c1154b66-3d70-4585-908c-2aa92646049f"
    corrupted = "c1154b66-3d70-4585-908c-2aa92642049f"

    d = _greeted()
    _held(d, issued)
    invoke = _confirm(d, corrupted)

    assert invoke.arguments["hold_id"] == issued
    # everything else the model supplied is still its own — this is intake, not a credential
    assert invoke.arguments["patient_name"] == "Nick"
    assert invoke.arguments["date_of_birth"] == "05/08/2003"


def test_a_fabricated_hold_id_is_overwritten_too():
    d = _greeted()
    _held(d, "aaaaaaaa-1111-2222-3333-444444444444")
    assert _confirm(d, "totally-made-up").arguments["hold_id"] == (
        "aaaaaaaa-1111-2222-3333-444444444444"
    )


def test_the_newest_hold_wins_when_a_slot_is_re_held():
    """A caller who changes their mind mid-flow must not confirm the abandoned slot."""
    d = _greeted()
    _held(d, "old-hold-0000", slot_id=25)
    _held(d, "new-hold-1111", slot_id=35)
    assert _confirm(d, "old-hold-0000").arguments["hold_id"] == "new-hold-1111"


def test_a_failed_hold_does_not_replace_the_live_one():
    d = _greeted()
    _held(d, "good-hold-0000", slot_id=25)
    rid = d.state.request_id or _turn(d)
    d.send(ev.LLMToolUse(request_id=rid, tool_call_id="t-bad",
                         name="hold_slot", arguments={"slot_id": 99}))
    d.send(ev.ToolCompleted(tool_call_id="t-bad", name="hold_slot", ok=False,
                            result={"ok": False, "error": "slot taken"},
                            latency_ms=1.0, http_status=409))
    assert _confirm(d, "whatever").arguments["hold_id"] == "good-hold-0000"


def test_a_failed_confirm_keeps_the_hold_so_the_retry_reuses_it():
    """A 409 from something other than the id must not force a redundant re-hold."""
    d = _greeted()
    _held(d, "live-hold-0000")
    _confirm(d, "live-hold-0000", tid="t-c1")
    d.send(ev.ToolCompleted(tool_call_id="t-c1", name="confirm_booking", ok=False,
                            result={"ok": False, "error": "nope"},
                            latency_ms=1.0, http_status=409))
    assert _confirm(d, "garbage", tid="t-c2").arguments["hold_id"] == "live-hold-0000"


def test_a_successful_confirm_spends_the_hold():
    """A second booking in the same call must not silently reuse a consumed hold."""
    d = _greeted()
    _held(d, "spent-hold-0000")
    _confirm(d, "spent-hold-0000", tid="t-c1")
    d.send(ev.ToolCompleted(tool_call_id="t-c1", name="confirm_booking", ok=True,
                            result={"ok": True, "confirmation_id": "ABC123", "slot_id": 25},
                            latency_ms=1.0, http_status=200))
    assert d.state.active_hold_id == ""
    # with no live hold the model's own value passes through and the API decides
    assert _confirm(d, "spent-hold-0000", tid="t-c2").arguments["hold_id"] == "spent-hold-0000"


def test_confirm_without_any_hold_passes_the_models_value_through():
    """No live hold means no opinion — the API's existing error path still speaks."""
    d = _greeted()
    assert _confirm(d, "unheld-1234").arguments["hold_id"] == "unheld-1234"


# --- the nudge must not fire on a turn whose tool already ran -------------------------------


def test_the_follow_up_after_a_tool_call_is_not_nudged():
    """Live dead-air bug, trace 20260903T215242896036Z at t=130.4.

    The caller said "Yeah. Sure.", the model called `hold_slot`, the tool succeeded, and the
    follow-up request narrated the read-back: "I'm holding that for you. So I have you as
    Olivia. Your date of birth is ... Does that all sound right?"

    `committed` only counts tool calls made by THIS request, and `turn_tool_uses` is cleared at
    the end of every request — so the narration looked like "announced an action, took none"
    and the nudge fired. Starting that extra request closed the live TTS context: Cartesia
    reported `Context closed`, `BotStoppedSpeaking` arrived with `completed: false` 1.06 s into
    an ~8 second sentence, and the nudge's own reply was queued to the dying context and never
    spoken at all. The caller heard "I'm holding that for you. So I have you as Ol—" and then
    6.5 seconds of silence, on the confirmation read-back — the single most important sentence
    in the call. They said "Hello?" to get the agent back.

    The rule was always "a turn that announces an action and calls NO TOOL", per caller turn.
    A tool ran. The nudge must not fire.
    """
    d = _greeted()
    rid = _turn(d, "Yeah. Sure.")

    d.send(ev.LLMToolUse(request_id=rid, tool_call_id="tu-hold",
                         name="hold_slot", arguments={"slot_id": 24}))
    d.send(ev.LLMCompleted(request_id=rid, stop_reason="tool_use"))
    d.send(ev.ToolCompleted(tool_call_id="tu-hold", name="hold_slot", ok=True,
                            result={"ok": True, "hold_id": "h-1", "slot_id": 24},
                            latency_ms=1.0, http_status=200))

    follow_up = d.state.request_id
    assert follow_up is not None and follow_up != rid
    d.send(ev.LLMTextDelta(request_id=follow_up, text="I'm holding that for you. "))
    d.send(ev.LLMTextDelta(request_id=follow_up, text="Does that all sound right?"))
    produced = d.send(ev.LLMCompleted(request_id=follow_up, stop_reason="end_turn"))

    assert not [a for a in produced if isinstance(a, StartLLM)], (
        "nudged a turn whose tool had already run — this truncates the agent mid-sentence"
    )
    assert d.state.nudged is False


def test_an_empty_promise_with_no_tool_anywhere_in_the_turn_is_still_nudged():
    """The behaviour the nudge exists for must survive the fix."""
    d = _greeted()
    rid = _turn(d, "can you check Monday?")
    d.send(ev.LLMTextDelta(request_id=rid, text="Let me check that for you."))
    produced = d.send(ev.LLMCompleted(request_id=rid, stop_reason="end_turn"))

    assert [a for a in produced if isinstance(a, StartLLM)], "the follow-through nudge stopped firing"
    assert d.state.nudged is True


def test_a_new_caller_turn_restores_the_nudge_after_a_tool_turn():
    """`turn_had_tool` is per caller turn, like `nudged` — not for the rest of the call."""
    d = _greeted()
    rid = _turn(d, "hold that slot")
    d.send(ev.LLMToolUse(request_id=rid, tool_call_id="tu-1", name="hold_slot",
                         arguments={"slot_id": 1}))
    d.send(ev.LLMCompleted(request_id=rid, stop_reason="tool_use"))
    d.send(ev.ToolCompleted(tool_call_id="tu-1", name="hold_slot", ok=True,
                            result={"ok": True, "hold_id": "h", "slot_id": 1},
                            latency_ms=1.0, http_status=200))
    follow = d.state.request_id
    d.send(ev.LLMTextDelta(request_id=follow, text="Holding that now."))
    d.send(ev.LLMCompleted(request_id=follow, stop_reason="end_turn"))

    rid2 = _turn(d, "and what about Tuesday?")
    assert d.state.turn_had_tool is False, "a tool from the previous turn silenced this one"
    d.send(ev.LLMTextDelta(request_id=rid2, text="Let me check Tuesday for you."))
    produced = d.send(ev.LLMCompleted(request_id=rid2, stop_reason="end_turn"))
    assert [a for a in produced if isinstance(a, StartLLM)]


def test_a_read_back_after_a_successful_tool_is_never_nudged():
    """The 2026-09-03 truncation, and until Phase 16 nothing asserted it.

    `hold_slot` succeeded; the follow-up request narrates the result — "I'm holding that for
    you. So I have you as Olivia..." — and makes no tool call of its own, because there is
    nothing left to call. `turn_tool_uses` is cleared at the end of every request and
    `committed_tool_ids` when results are flushed, so without `turn_had_tool` that read-back
    looks exactly like an empty promise.

    What the nudge costs here is not "one extra request": starting it closes the live TTS
    context. Trace 20260903T215242896036Z at t=130.4 — Cartesia reported `Context closed`,
    `BotStoppedSpeaking` arrived with `completed: false` 1.06 s into an ~8 second sentence, and
    the nudge's own reply was queued to the dying context and never spoken. The caller heard
    "...So I have you as Ol—", then 6.5 s of silence, on the confirmation read-back.
    """
    d = _greeted()
    rid = _turn(d, "Yeah. Sure.")
    d.send(ev.LLMToolUse(request_id=rid, tool_call_id="tu-1", name="hold_slot",
                         arguments={"slot_id": 5}))
    d.send(ev.LLMCompleted(request_id=rid, stop_reason="tool_use"))
    d.send(ev.ToolCompleted(tool_call_id="tu-1", name="hold_slot", ok=True,
                            result={"ok": True, "hold_id": "h-1", "slot_id": 5},
                            latency_ms=120.0, http_status=200))

    follow = d.state.request_id
    d.send(ev.LLMTextDelta(request_id=follow, text="I'm holding that for you. "))
    d.send(ev.LLMTextDelta(request_id=follow, text="So I have you as Olivia."))
    produced = d.send(ev.LLMCompleted(request_id=follow, stop_reason="end_turn"))

    assert not [a for a in produced if isinstance(a, StartLLM)], (
        "nudged a read-back whose tool had already succeeded — this closes the live TTS "
        "context and truncates the sentence the caller is listening to"
    )
    assert d.state.nudged is False


def test_a_promise_that_ends_by_asking_the_caller_something_is_not_nudged():
    """Live truncation bug, trace 20260904T173146047919Z at t=34.0.

    A returning caller opened with "I need to reschedule". The model replied "Good to hear from
    you — I'm happy to help move that. Let me pull up your appointment first. What's your full
    name?" and called no tool, because it could not: every PHI tool is gated behind
    `verify_identity` and the caller had not verified yet. So the model did the one correct
    thing available to it — it asked for what it needed, which is verbatim what `_ACTION_NUDGE`
    instructs.

    The nudge fired anyway. Starting that request closed the live TTS context: `ProviderDegraded`
    tts "Context closed" and `BotStoppedSpeaking completed: false` 0.8 s into a ~6 second
    sentence. The caller heard "Good to hear from you — I'm happy to h—", then the replacement
    reply, and said "Hello?" — 15 seconds lost on the first turn of the call.

    A reply that asks the caller a question is not silence. It must not be nudged.
    """
    d = _greeted()
    rid = _turn(d, "I need to reschedule my appointment")
    d.send(ev.LLMTextDelta(request_id=rid, text="Let me pull up your appointment first. "))
    d.send(ev.LLMTextDelta(request_id=rid, text="What's your full name?"))
    produced = d.send(ev.LLMCompleted(request_id=rid, stop_reason="end_turn"))

    assert not [a for a in produced if isinstance(a, StartLLM)], (
        "nudged a reply that had already asked the caller for what it needed — "
        "this truncates the agent mid-sentence"
    )
    assert d.state.nudged is False


# --- Phase 14: speak the opening clause rather than waiting for a full sentence --------------


def test_the_opening_clause_is_spoken_before_the_sentence_finishes():
    """Phase 14, finding 1.

    Measured across the two 2026-09-04 calls, the wait from the LLM's first token to its first
    sentence-ending period was 205-255 ms p50 — most of the entire "speech queue" stage, and
    none of it TTS. Cartesia and playback account for only ~60-85 ms.

    This is the real opening of a live reply, cut where the model's first burst of output
    actually ended. There is no period in it yet, so before this change the caller heard
    nothing at all until the next burst arrived.
    """
    assert _split_speakable("Good to hear from you—I'm happy to h", opening=True) == (
        ["Good to hear from you—"], "I'm happy to h",
    )


def test_the_clause_split_applies_only_to_the_opening_of_a_reply():
    """Mid-reply there is already audio playing, so an early split buys silence back from
    nobody and costs prosody. `opening=False` is the reducer saying an utterance is already
    open (`state.utterance_id is not None`)."""
    text = "Good to hear from you—I'm happy to h"
    assert _split_speakable(text, opening=False) == ([], text)


def test_a_clause_too_short_to_outlast_the_next_burst_is_not_spoken():
    """The floor is an underrun guard, not a style rule: the opening chunk must take longer to
    SPEAK than the next burst of model output takes to ARRIVE (79-102 ms p50, ~335 ms p95), or
    the caller hears the reply stutter mid-phrase instead of starting late."""
    assert _split_speakable("Perfect — I'm holding that", opening=True)[0] == []


def test_a_complete_sentence_still_wins_over_the_clause_split():
    """A period is a better place to breathe than a comma, so it must still take precedence —
    the clause path is a fallback for when no sentence exists yet, not a replacement."""
    assert _split_speakable("Thanks, Nikhil. And your date", opening=True) == (
        ["Thanks, Nikhil."], " And your date",
    )


# --- live call 2026-09-04: a refill the agent announced and never filed ----------------------


def test_a_promise_to_SEND_something_is_a_promise_like_any_other():
    """Live fabrication, trace `20260904T183444349772Z` at t=123.5.

    The caller asked for a prescription refill. The intent classifier had already switched to
    `medication_refill` at 0.95, so `request_refill` WAS in the model's tool set (req-13 routed
    as "medication_refill dialogue"). The model called nothing, `stop_reason: end_turn`, and
    said: "Got it — I'll send a refill request for ibuprofen to our staff, and they'll review
    it and follow up with you."

    No `staff_tasks` row was created. The caller thanked the agent and hung up believing a
    refill was queued. The nudge exists for exactly this and stayed silent, because
    `_ACTION_CLAIM` listed `check|look|see|pull|find|book|hold` and not `send` — and
    "sending / filing / passing along to staff" is the entire shape of the non-scheduling
    tools, so those were the most consequential verbs to be missing.
    """
    d = _greeted()
    rid = _turn(d, "Ibuprofen.")
    d.send(ev.LLMTextDelta(request_id=rid, text="Got it — I'll send a refill request for "))
    d.send(ev.LLMTextDelta(request_id=rid, text="ibuprofen to our staff, and they'll review it."))
    produced = d.send(ev.LLMCompleted(request_id=rid, stop_reason="end_turn"))

    assert [a for a in produced if isinstance(a, StartLLM)], (
        "announced a refill request and called no tool, and the nudge did not fire"
    )
    assert d.state.nudged is True


def test_claiming_something_is_already_done_is_nudged_too():
    """Past tense is the worse case: not an unkept promise, a completed action that never
    happened. A turn whose tool actually ran is excluded by `turn_had_tool`, so this cannot
    fire on a legitimate read-back."""
    d = _greeted()
    rid = _turn(d, "did you send that?")
    d.send(ev.LLMTextDelta(request_id=rid, text="Yes, I've sent that over to our staff."))
    produced = d.send(ev.LLMCompleted(request_id=rid, stop_reason="end_turn"))
    assert [a for a in produced if isinstance(a, StartLLM)]


def test_verify_identity_with_no_date_of_birth_never_reaches_the_api():
    """Live, same trace at t=44.8. The caller's opening sentence named them, so the model
    called `verify_identity` straight away with `date_of_birth: ""`. The API answered 422 and
    the caller heard "I apologize — let me ask that differently" for a mistake they could not
    see. A blank credential is answered here, with the instruction, and costs no round trip."""
    d = _greeted()
    rid = _turn(d, "Hi, my name is Nikhil and I want to cancel my appointment.")
    produced = d.send(ev.LLMToolUse(request_id=rid, tool_call_id="tu-v", name="verify_identity",
                                    arguments={"name": "Nikhil", "date_of_birth": ""}))

    assert not [a for a in produced if isinstance(a, InvokeTool)], (
        "sent a blank date of birth to the API"
    )
    result = json.loads(d.state.tool_results[-1]["content"])
    assert result["ok"] is False and "date_of_birth is empty" in result["error"]
    assert d.state.turn_had_tool is True, (
        "a refused tool must still count as a tool for the nudge, or the recovery turn gets "
        "re-prompted on top of the refusal"
    )


def test_a_tool_less_intent_is_never_nudged():
    """Phase 15. The nudge asks why no tool was called; on a tool-less intent there is none.

    Live shape this fixes: the model said "I'm passing you to a staff member", the nudge fired,
    and it talked itself back out of it — "I don't have the ability to transfer calls" — which
    is now false as well as unhelpful. The hand-off is performed by the engine.
    """
    d = _greeted()
    d.send(ev.FinalTranscript(text="are my test results back?"))
    d.state = replace(d.state, intent=Intent.TEST_RESULTS)
    produced = d.send(
        ev.LLMCompleted(
            request_id=d.state.request_id,
            text="Let me pass you to a staff member who can look at that.",
        )
    )

    assert not any(isinstance(a, StartLLM) for a in produced)
    assert d.state.nudged is False

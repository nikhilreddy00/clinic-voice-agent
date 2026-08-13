"""Phase 10 — unit tests for the pure reducer.

These are the tests that replace "run the pipeline and listen to it". Because ``reduce()`` is
pure, a full booking conversation — greeting, slot fill, two tool calls, barge-in, hangup —
is an ordinary list of dataclasses and a list of assertions, running in microseconds with no
audio devices, no network, and no API keys.
"""

from __future__ import annotations

import json

import pytest

from clinic_agent.core import events as ev
from clinic_agent.core.actions import (
    CancelLLM,
    CancelSpeech,
    EndCall,
    InvokeTool,
    Speak,
    StartLLM,
)
from clinic_agent.core.reducer import _split_speakable, reduce
from clinic_agent.core.state import CallState, Phase
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
        )
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

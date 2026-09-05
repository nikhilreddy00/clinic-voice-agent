"""Phase 15 — the degradation ladder: primary → retry → scripted line → a person.

Every rung is pure, so the whole thing is a list of events and a list of assertions. This is
the "failsafe monitor" the phase plan asks for: infra health (`state.degraded`) and
conversation quality (the counters) decided in the one function that already owns both.

The ordering property these tests exist to protect: **the caller is told what is happening
before the line is handed over.** A SIP transfer that fires while the agent is mid-sentence is
a call that sounds like it dropped.
"""

from __future__ import annotations

import pytest

from clinic_agent.core import events as ev
from clinic_agent.core.actions import EndCall, Speak, StartLLM, TransferToHuman
from clinic_agent.core.state import Phase
from clinic_agent.intents import Intent
from clinic_agent.prompts import HANDOFF_LINE, HANDOFF_UNAVAILABLE_LINE, SYSTEM_ERROR_LINE

from test_reducer import Driver, _greeted


def _speaks(produced) -> list[str]:
    return [a.text for a in produced if isinstance(a, Speak)]


def _transfer(produced) -> TransferToHuman | None:
    return next((a for a in produced if isinstance(a, TransferToHuman)), None)


def _finish_speaking(d: Driver) -> list:
    return d.send(ev.BotStoppedSpeaking(utterance_id=d.state.utterance_id))


def _fail_a_turn(d: Driver, text: str = "anything tomorrow?") -> list:
    d.send(ev.FinalTranscript(text=text))
    return d.send(ev.LLMFailed(request_id=d.state.request_id, error="Overloaded"))


# --- the LLM rung ---------------------------------------------------------------------------


def test_the_first_llm_failure_is_a_scripted_apology_not_a_transfer():
    """One failure is a blip the caller barely notices. Handing off on it is an overreaction."""
    d = _greeted()
    produced = _fail_a_turn(d)

    assert _speaks(produced) == [SYSTEM_ERROR_LINE]
    assert _transfer(produced) is None
    assert d.state.llm_failures == 1


def test_the_second_consecutive_llm_failure_hands_off():
    d = _greeted()
    _fail_a_turn(d)
    _finish_speaking(d)
    produced = _fail_a_turn(d, "let me try again")

    assert _speaks(produced) == [HANDOFF_LINE]
    assert _transfer(produced) is None, "the line has not played yet"

    produced = _finish_speaking(d)
    transfer = _transfer(produced)
    assert transfer is not None and transfer.reason == "llm_unavailable"
    assert d.state.escalated is True


def test_a_successful_turn_clears_the_llm_streak():
    """The ladder asks whether the call is failing NOW, not whether it ever failed."""
    d = _greeted()
    _fail_a_turn(d)
    _finish_speaking(d)

    d.send(ev.FinalTranscript(text="hello?"))
    d.send(ev.LLMTextDelta(request_id=d.state.request_id, text="Hi there. "))
    d.send(ev.LLMCompleted(request_id=d.state.request_id, text="Hi there. "))

    assert d.state.llm_failures == 0


# --- the tool rung --------------------------------------------------------------------------


def _tool_turn(d: Driver, *, ok: bool, http_status=None, text="anything tomorrow?") -> list:
    d.send(ev.FinalTranscript(text=text))
    request_id = d.state.request_id
    tool_call_id = f"tu_{d.state.turn_index}"
    d.send(ev.LLMToolUse(request_id=request_id, tool_call_id=tool_call_id,
                         name="check_availability"))
    d.send(ev.LLMCompleted(request_id=request_id, stop_reason="tool_use"))
    return d.send(
        ev.ToolCompleted(
            tool_call_id=tool_call_id,
            name="check_availability",
            result={"ok": ok} if ok else {"ok": False, "error": "could not reach"},
            ok=ok,
            http_status=http_status,
        )
    )


def test_two_unreachable_tool_calls_hand_off():
    d = _greeted()
    produced = _tool_turn(d, ok=False)
    assert any(isinstance(a, StartLLM) for a in produced), "the model gets one chance to recover"

    produced = _tool_turn(d, ok=False, text="try again please")
    assert _speaks(produced) == [HANDOFF_LINE]
    assert _transfer(_finish_speaking(d)).reason == "scheduling_unavailable"


@pytest.mark.parametrize("status", [403, 404, 409])
def test_a_business_answer_is_not_an_outage(status):
    """A wrong date of birth, a missing appointment, a slot just taken — the model handles all
    three in dialogue. Transferring a caller for mistyping their birthday twice is a defect."""
    d = _greeted()
    _tool_turn(d, ok=False, http_status=status)
    produced = _tool_turn(d, ok=False, http_status=status, text="again")

    assert _transfer(produced) is None
    assert d.state.tool_failures == 0
    assert any(isinstance(a, StartLLM) for a in produced)


def test_a_success_between_two_failures_clears_the_streak():
    d = _greeted()
    _tool_turn(d, ok=False)
    _tool_turn(d, ok=True, text="ok")
    produced = _tool_turn(d, ok=False, text="and now")

    assert _transfer(produced) is None
    assert d.state.tool_failures == 1


# --- conversation-quality rungs ---------------------------------------------------------------


def test_three_no_matches_hand_off():
    d = _greeted()
    for _ in range(2):
        assert d.send(ev.FinalTranscript(text="")) == []
    produced = d.send(ev.FinalTranscript(text=""))

    assert _speaks(produced) == [HANDOFF_LINE]
    assert _transfer(_finish_speaking(d)).reason == "no_match"


def test_a_no_match_streak_is_broken_by_one_good_turn():
    d = _greeted()
    d.send(ev.FinalTranscript(text=""))
    d.send(ev.FinalTranscript(text="I'd like an appointment", confidence=0.95))
    d.send(ev.FinalTranscript(text=""))

    assert d.state.no_match_count == 1
    assert d.state.transferring is False


def test_sustained_low_asr_confidence_hands_off():
    """Deepgram answering with a guess three times running is a line this stack cannot serve."""
    d = _greeted()
    for _ in range(2):
        d.send(ev.FinalTranscript(text="mmhm garbled", confidence=0.3))
    produced = d.send(ev.FinalTranscript(text="still garbled", confidence=0.2))

    assert _speaks(produced) == [HANDOFF_LINE]
    assert _transfer(_finish_speaking(d)).reason == "low_asr_confidence"


def test_one_low_confidence_turn_still_gets_answered():
    d = _greeted()
    produced = d.send(ev.FinalTranscript(text="tuesday works", confidence=0.4))

    assert any(isinstance(a, StartLLM) for a in produced)


# --- provider health ------------------------------------------------------------------------


def test_a_transient_degradation_does_not_hand_off_and_can_recover():
    d = _greeted()
    d.send(ev.ProviderDegraded(provider="stt", reason="closed"))
    assert d.state.degraded == ("stt",)
    assert d.state.transferring is False

    d.send(ev.ProviderRecovered(provider="stt"))
    assert d.state.degraded == (), "a recovered provider must not stay degraded all call"


def test_stt_that_is_never_coming_back_hands_off_with_a_spoken_line():
    """The caller can still hear, so they are told. Then the line goes to a person."""
    d = _greeted()
    produced = d.send(ev.ProviderDegraded(provider="stt", reason="closed", fatal=True))

    assert _speaks(produced) == [HANDOFF_LINE]
    assert _transfer(_finish_speaking(d)).reason == "stt_unavailable"


def test_dead_tts_transfers_immediately_because_nothing_can_be_spoken():
    """No line is emitted — there is nothing left to speak it with, and waiting for a
    BotStoppedSpeaking that can never arrive would strand the caller in silence."""
    d = _greeted()
    produced = d.send(ev.ProviderDegraded(provider="tts", reason="closed", fatal=True))

    assert _speaks(produced) == []
    assert _transfer(produced).reason == "tts_unavailable"


# --- the caller asking, and the fallback ------------------------------------------------------


def test_asking_for_a_person_hands_off_and_abandons_the_model_turn():
    d = _greeted()
    d.send(ev.FinalTranscript(text="can I just talk to someone"))
    produced = d.send(
        ev.IntentClassified(intent=Intent.SPEAK_TO_HUMAN.value, confidence=0.95)
    )

    assert _speaks(produced) == [HANDOFF_LINE]
    assert _transfer(_finish_speaking(d)).reason == "caller_requested_human"


def test_a_failed_transfer_still_tells_the_caller_something():
    """A hand-off that ends in a click is worse than never having offered one."""
    d = _greeted()
    d.send(ev.ProviderDegraded(provider="stt", reason="closed", fatal=True))
    _finish_speaking(d)

    produced = d.send(ev.TransferFailed(reason="no destination configured"))
    assert _speaks(produced) == [HANDOFF_UNAVAILABLE_LINE]

    produced = _finish_speaking(d)
    assert any(isinstance(a, EndCall) for a in produced)
    assert d.state.phase is Phase.CLOSED


def test_a_call_transfers_at_most_once():
    d = _greeted()
    d.send(ev.ProviderDegraded(provider="stt", reason="closed", fatal=True))
    first = _finish_speaking(d)
    assert _transfer(first) is not None

    d.send(ev.ProviderDegraded(provider="tts", reason="closed", fatal=True))
    assert _transfer(_finish_speaking(d)) is None


def test_the_summary_carries_routing_context_and_no_identity():
    """It goes to logs, traces and metrics. A name in there is the PHI leak the gate prevents."""
    d = _greeted()
    d.send(ev.FinalTranscript(text="I need to reschedule", confidence=0.9))
    d.send(ev.IntentClassified(intent=Intent.RESCHEDULE_APPOINTMENT.value, confidence=0.95))
    d.send(ev.LLMFailed(request_id=d.state.request_id, error="Overloaded"))
    _finish_speaking(d)
    d.send(ev.FinalTranscript(text="hello?"))
    d.send(ev.LLMFailed(request_id=d.state.request_id, error="Overloaded"))

    summary = _transfer(_finish_speaking(d)).summary
    assert "intent=reschedule_appointment" in summary
    assert "reason=llm_unavailable" in summary
    assert "turns=" in summary and "verified=False" in summary


def test_a_failed_transfer_after_an_emergency_does_not_promise_a_callback():
    """The caller has been told to hang up and call 911.

    "Someone will call you back as soon as they can" is both the wrong thing to say to them and
    the wrong thing to have them wait for. The line closes instead.
    """
    d = _greeted()
    d.send(ev.FinalTranscript(text="I can't breathe"))
    _finish_speaking(d)

    produced = d.send(ev.TransferFailed(reason="no destination configured"))

    assert _speaks(produced) == []
    assert any(isinstance(a, EndCall) for a in produced)
    assert d.state.phase is Phase.CLOSED

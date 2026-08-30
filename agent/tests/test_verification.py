"""Phase 13 — the identity gate, caller memory, and argument scoping, inside the engine.

The API has its own adversarial suite (`scheduling_api/tests/test_verified_flows.py`); this
one asserts the half that runs BEFORE any HTTP request exists. The distinction matters: the
strongest claim here is not "the server refuses", it is that an unverified request never
becomes a request at all — no InvokeTool action, nothing on the wire, nothing to intercept.
"""

from __future__ import annotations

from clinic_agent.core import events as ev
from clinic_agent.core.actions import InvokeTool, LoadCallerMemory, StartLLM
from clinic_agent.scheduling_tools import VERIFICATION_REQUIRED_TOOLS

from test_reducer import Driver, _greeted

CALLER = "+15551230001"
DOB = "03/15/1990"


def _on_a_call(phone: str = CALLER) -> Driver:
    """A greeted call that arrived with an ANI, as a telephony call does."""
    d = Driver()
    d.send(ev.CallStarted(call_id="c1", mode="telephony"))
    d.send(ev.CallerPresent(participant_id="sip_1", phone=phone))
    d.send(ev.BotStartedSpeaking(utterance_id="utt-1"))
    d.send(ev.BotStoppedSpeaking(utterance_id="utt-1"))
    return d


def _tool_call(d: Driver, name: str, **arguments) -> list:
    """Drive one model tool call and return the actions it produced."""
    return d.send(ev.LLMToolUse(
        request_id=d.state.request_id or "req-1",
        tool_call_id=f"tu-{name}",
        name=name,
        arguments=arguments,
    ))


def _verify(d: Driver, dob: str = DOB) -> None:
    _tool_call(d, "verify_identity", date_of_birth=dob)
    d.send(ev.ToolCompleted(
        tool_call_id="tu-verify_identity", name="verify_identity", ok=True,
        result={"ok": True, "patient_id": 7, "name": "Dana Reyes"},
    ))


# --- the gate ----------------------------------------------------------------------------


def test_an_unverified_phi_tool_never_becomes_a_request():
    """The headline: a spoofed ANI does not even reach the network."""
    d = _on_a_call()
    d.send(ev.FinalTranscript(text="what appointments do I have?"))

    for name in sorted(VERIFICATION_REQUIRED_TOOLS):
        produced = _tool_call(d, name, confirmation_id="ABC123", medication="something")
        assert not [a for a in produced if isinstance(a, InvokeTool)], (
            f"{name} was dispatched without verification"
        )


def test_the_refusal_is_handed_back_to_the_model_not_swallowed():
    """A refused tool still owes the model a tool_result, or the call stalls in silence with
    an assistant tool_use nothing ever answers."""
    d = _on_a_call()
    d.send(ev.FinalTranscript(text="what appointments do I have?"))
    _tool_call(d, "list_appointments")

    produced = d.send(ev.LLMCompleted(request_id=d.state.request_id, stop_reason="tool_use"))
    start = next(a for a in produced if isinstance(a, StartLLM))
    results = start.messages[-1]["content"]
    assert results[0]["type"] == "tool_result"
    assert "verify_identity" in results[0]["content"], (
        "the refusal must tell the model what to do next, not just say no"
    )
    assert d.state.identity_verified is False


def test_verifying_opens_the_gate_and_only_the_api_can_open_it():
    d = _on_a_call()
    d.send(ev.FinalTranscript(text="I need to move my appointment"))

    # A failed verification leaves the gate shut.
    _tool_call(d, "verify_identity", date_of_birth="01/01/1900")
    d.send(ev.ToolCompleted(
        tool_call_id="tu-verify_identity", name="verify_identity", ok=False,
        result={"ok": False, "status": 403, "error": "could not verify"},
    ))
    assert d.state.identity_verified is False
    assert not [a for a in _tool_call(d, "list_appointments") if isinstance(a, InvokeTool)]

    _verify(d)
    assert d.state.identity_verified is True
    assert d.state.patient_name == "Dana Reyes"
    assert [a for a in _tool_call(d, "list_appointments") if isinstance(a, InvokeTool)]


def test_booking_a_new_appointment_needs_no_verification():
    """Requiring it would lock out every first-time caller, and a new booking discloses
    nothing — there is no record to read."""
    d = _on_a_call(phone="")
    d.send(ev.FinalTranscript(text="I'd like to book a checkup"))
    for name in ("check_availability", "hold_slot", "confirm_booking"):
        assert [a for a in _tool_call(d, name) if isinstance(a, InvokeTool)], name


# --- argument scoping ---------------------------------------------------------------------


def test_the_model_cannot_choose_whose_chart_to_read():
    """phone and date_of_birth are injected from state. A model that supplies its own — because
    a caller read a number aloud, or a transcript contained one — is overwritten, not merged."""
    d = _on_a_call()
    d.send(ev.FinalTranscript(text="cancel my appointment"))
    _verify(d)

    produced = _tool_call(
        d, "cancel_appointment",
        confirmation_id="ABC123", phone="+19995550000", date_of_birth="01/01/1970",
    )
    invoke = next(a for a in produced if isinstance(a, InvokeTool))
    assert invoke.arguments["phone"] == CALLER
    assert invoke.arguments["date_of_birth"] == DOB


def test_confirm_booking_keeps_the_dob_the_caller_gave():
    """The one exception: on a NEW booking the DOB is intake, not a credential, so the value
    the caller just spoke is the right one — but the phone still comes from the ANI."""
    d = _on_a_call()
    d.send(ev.FinalTranscript(text="book me in"))
    produced = _tool_call(
        d, "confirm_booking", hold_id="h1", patient_name="Dana Reyes",
        reason="checkup", date_of_birth="07/04/1988",
    )
    invoke = next(a for a in produced if isinstance(a, InvokeTool))
    assert invoke.arguments["date_of_birth"] == "07/04/1988"
    assert invoke.arguments["phone"] == CALLER


def test_a_second_verification_attempt_cannot_replace_a_matched_dob():
    """Once the API has matched a date of birth, later attempts must not overwrite the value
    every subsequent PHI call re-sends for server-side re-verification."""
    d = _on_a_call()
    d.send(ev.FinalTranscript(text="reschedule please"))
    _verify(d)
    _tool_call(d, "verify_identity", date_of_birth="12/12/1999")
    assert d.state.verified_dob == DOB


# --- caller memory -------------------------------------------------------------------------


def test_the_ani_triggers_a_memory_lookup_after_the_greeting():
    d = Driver()
    d.send(ev.CallStarted(call_id="c1", mode="telephony"))
    produced = d.send(ev.CallerPresent(participant_id="sip_1", phone=CALLER))
    assert LoadCallerMemory(phone=CALLER) in produced
    assert d.state.caller_phone == CALLER


def test_no_ani_means_no_lookup_and_no_asking_for_a_number():
    d = Driver()
    d.send(ev.CallStarted(call_id="c1", mode="local"))
    produced = d.send(ev.CallerPresent())
    assert not [a for a in produced if isinstance(a, LoadCallerMemory)]


def test_a_recognised_caller_is_never_named_before_verification():
    """The disclosure that would defeat the whole gate one turn early: greeting whoever is
    holding the phone by the account holder's name."""
    d = _on_a_call()
    d.send(ev.CallerMemoryLoaded(known=True, upcoming_appointments=1))
    produced = d.send(ev.FinalTranscript(text="hi"))
    note = next(a for a in produced if isinstance(a, StartLLM)).context_note

    assert "reached the clinic before" in note
    assert "Dana" not in note
    assert "not proof of identity" in note
    assert "until verify_identity has succeeded" in note


def test_the_name_appears_only_after_the_dob_matched():
    d = _on_a_call()
    d.send(ev.CallerMemoryLoaded(known=True, upcoming_appointments=1))
    d.send(ev.FinalTranscript(text="I need to reschedule"))
    _verify(d)

    produced = d.send(ev.FinalTranscript(text="what do I have booked?"))
    note = next(a for a in produced if isinstance(a, StartLLM)).context_note
    assert "IDENTITY VERIFIED" in note and "Dana Reyes" in note


def test_an_unrecognised_number_adds_nothing_to_the_prompt():
    """A first-time caller is the default the booking prompt already describes, so the note is
    empty rather than a line of tokens on every turn saying nothing happened."""
    d = _on_a_call()
    d.send(ev.CallerMemoryLoaded(known=False, upcoming_appointments=0))
    produced = d.send(ev.FinalTranscript(text="hi"))
    assert next(a for a in produced if isinstance(a, StartLLM)).context_note == ""

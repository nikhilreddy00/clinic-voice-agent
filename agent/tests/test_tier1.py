"""Phase 16 — the Tier-1 invariants, proved to actually fire.

`eval/tier1_replay.py` runs on every commit and reports the recorded-call corpus as clean. That
is the regression signal, and it is NOT evidence the checks work: a check that always returns
"no violations" reports the same thing. So each invariant gets a hand-built event stream
containing exactly the defect it exists to catch, and has to go red on it.

Every fixture below is a real failure from CLAUDE.md, reduced to the smallest stream that
reproduces its shape.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "eval"))

import tier1_replay as t1  # noqa: E402

from clinic_agent.core import events as ev  # noqa: E402
from clinic_agent.core.actions import InvokeTool, Speak  # noqa: E402
from clinic_agent.core.reducer import _ACTION_NUDGE  # noqa: E402
from clinic_agent.core.state import CallState  # noqa: E402


def steps(*specs) -> list[t1.Step]:
    """Build a step list directly from (before, after, actions) triples.

    The invariants read state transitions and actions, so a fixture states the transition it
    wants rather than driving a whole call to arrive at it. Testing the CHECK, not the reducer
    — the reducer has its own suite in test_reducer.py.
    """
    out = []
    for i, (before, after, actions) in enumerate(specs, start=1):
        out.append(t1.Step(
            event=ev.LLMCompleted(seq=i, request_id="req-1", stop_reason="end_turn"),
            before=before,
            after=after,
            actions=list(actions),
        ))
    return out


def names(violations) -> set[str]:
    return {v.invariant for v in violations}


# --- the identity gate --------------------------------------------------------------------


def test_ungated_phi_tool_fires_on_an_unverified_lookup():
    """A PHI tool became a request before the caller matched a date of birth."""
    unverified = CallState(identity_verified=False)
    found = t1.ungated_phi_tool(steps(
        (unverified, unverified, [InvokeTool(tool_call_id="t1", name="list_appointments")]),
    ))
    assert names(found) == {"ungated_phi_tool"}


def test_ungated_phi_tool_allows_a_verified_lookup_and_an_ungated_tool():
    verified = CallState(identity_verified=True, verified_dob="12/08/2000", patient_name="Nick")
    unverified = CallState()
    assert not t1.ungated_phi_tool(steps(
        (verified, verified, [InvokeTool(tool_call_id="t1", name="list_appointments")]),
        (unverified, unverified, [InvokeTool(tool_call_id="t2", name="check_availability")]),
    ))


# --- the turn that never drains -----------------------------------------------------------


def _tool_use(seq: int, before: CallState, tool_id: str, name: str, *, accepted: bool,
              answered: bool = False, invoked: bool = False) -> t1.Step:
    """One LLMToolUse step. `accepted` mirrors the reducer appending to turn_tool_uses."""
    block = {"type": "tool_use", "id": tool_id, "name": name, "input": {}}
    after = before
    if accepted:
        after = replace(after, turn_tool_uses=after.turn_tool_uses + (block,))
    if answered:
        after = replace(after, messages=after.messages + ({
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": tool_id, "content": "{}"}],
        },))
    return t1.Step(
        event=ev.LLMToolUse(seq=seq, request_id="req-1", tool_call_id=tool_id, name=name,
                            arguments={}),
        before=before,
        after=after,
        actions=[InvokeTool(tool_call_id=tool_id, name=name)] if invoked else [],
    )


def _completed(seq: int, state: CallState) -> t1.Step:
    return t1.Step(event=ev.LLMCompleted(seq=seq, request_id="req-1", stop_reason="tool_use"),
                   before=state, after=state, actions=[])


def test_orphan_tool_use_fires_when_a_refused_tool_gets_no_result():
    """The silent-call hang: refused, so never invoked, and no result to drain the turn."""
    s = CallState()
    step = _tool_use(1, s, "t1", "list_appointments", accepted=True)
    found = t1.orphan_tool_use([step, _completed(2, step.after)])
    assert names(found) == {"orphan_tool_use"}


def test_orphan_tool_use_accepts_a_refusal_that_was_answered():
    """The gate refuses the tool AND hands back a tool_result — the turn drains."""
    s = CallState()
    step = _tool_use(1, s, "t1", "list_appointments", accepted=True, answered=True)
    assert not t1.orphan_tool_use([step, _completed(2, step.after)])


def test_orphan_tool_use_ignores_a_stale_tool_use():
    """Replay divergence is not a defect.

    A tool use belonging to a superseded request is dropped by `_stale` and changes no state,
    so there is no turn to drain. Counting it would make every trace recorded by an older
    engine look broken — which is exactly the trap this whole tier had to avoid.
    """
    s = CallState()
    step = _tool_use(1, s, "t1", "hold_slot", accepted=False)
    assert not t1.orphan_tool_use([step, _completed(2, step.after)])


def test_orphan_tool_use_ignores_a_turn_the_caller_hung_up_on():
    """No LLMCompleted for the request — there was no turn left to drain."""
    s = CallState()
    assert not t1.orphan_tool_use([_tool_use(1, s, "t1", "hold_slot", accepted=True)])


# --- what the caller hears ----------------------------------------------------------------


@pytest.mark.parametrize("text", [
    "<thinking>The hold id is c1154b66-3d70-4585-908c-2aa92646049f</thinking> All set.",
    "Let me check. </thinking> You're booked.",
    "Your hold is c1154b66-3d70-4585-908c-2aa92646049f, one moment.",
])
def test_thinking_spoken_fires_on_a_scratchpad_or_a_uuid(text):
    """A live call read its own thinking block, hold UUID included, to the caller."""
    s = CallState()
    found = t1.thinking_spoken(steps((s, s, [Speak(utterance_id="u1", text=text)])))
    assert names(found) == {"thinking_spoken"}


def test_thinking_spoken_allows_ordinary_speech():
    s = CallState()
    assert not t1.thinking_spoken(steps(
        (s, s, [Speak(utterance_id="u1", text="You're all set for Monday at 1 PM.")]),
    ))


# --- the worst failure in the system ------------------------------------------------------


def _confirmed(seq: int, state: CallState, code: str) -> t1.Step:
    return t1.Step(
        event=ev.ToolCompleted(seq=seq, tool_call_id="t1", name="confirm_booking", ok=True,
                               latency_ms=120.0, http_status=200,
                               result={"ok": True, "confirmation_id": code}),
        before=state, after=state, actions=[],
    )


def test_unbacked_confirmation_fires_on_a_fabricated_code():
    """The caller hangs up believing they have an appointment. Nothing was booked."""
    s = CallState()
    found = t1.unbacked_confirmation(steps((s, s, [
        Speak(utterance_id="u1", text="Your appointment confirmation code is **GFC-082826-1015**."),
    ])))
    assert names(found) == {"unbacked_confirmation"}


def test_unbacked_confirmation_accepts_a_code_a_tool_returned():
    s = CallState()
    found = t1.unbacked_confirmation([
        _confirmed(1, s, "A754E2BC"),
        *steps((s, s, [Speak(utterance_id="u1",
                             text="Your confirmation number is A754E2BC. Please arrive early.")])),
    ])
    assert not found


def test_unbacked_confirmation_accepts_a_hyphenated_read_back():
    """Live: the agent read 88566088 back as '88-56-60-88'. Same number, different punctuation."""
    s = CallState()
    found = t1.unbacked_confirmation([
        _confirmed(1, s, "88566088"),
        *steps((s, s, [Speak(utterance_id="u1", text="Your confirmation number is 88-56-60-88.")])),
    ])
    assert not found


def test_unbacked_confirmation_fires_when_the_spoken_code_is_a_different_booking():
    """One digit off is still a code the caller cannot use."""
    s = CallState()
    found = t1.unbacked_confirmation([
        _confirmed(1, s, "A754E2BC"),
        *steps((s, s, [Speak(utterance_id="u1", text="Your confirmation number is A754E2BD.")])),
    ])
    assert names(found) == {"unbacked_confirmation"}


# --- the follow-through nudge -------------------------------------------------------------


def _nudge_step(seq: int, before: CallState) -> t1.Step:
    after = replace(before, nudged=True,
                    messages=before.messages + ({"role": "user", "content": _ACTION_NUDGE},))
    return t1.Step(event=ev.LLMCompleted(seq=seq, request_id="req-1", stop_reason="end_turn"),
                   before=before, after=after, actions=[])


def test_nudge_discipline_fires_when_the_turn_already_ran_a_tool():
    """Trace 20260903T215242896036Z: hold_slot succeeded, the read-back was nudged anyway, and
    starting that request closed the live TTS context 1.06 s into an ~8 s sentence."""
    found = t1.nudge_discipline([_nudge_step(1, CallState(turn_had_tool=True))])
    assert names(found) == {"nudge_discipline"}


def test_nudge_discipline_fires_on_a_second_nudge_in_one_turn():
    found = t1.nudge_discipline([_nudge_step(1, CallState(nudged=True))])
    assert names(found) == {"nudge_discipline"}


def test_nudge_discipline_allows_the_one_legitimate_nudge():
    assert not t1.nudge_discipline([_nudge_step(1, CallState())])


# --- one caller, one credential -----------------------------------------------------------


def test_identity_pairing_fires_when_a_name_arrives_without_its_date_of_birth():
    """Verify as Joe, then as Nick: the engine reported Nick while every PHI call carried Joe's
    date of birth. On a shared handset that is the likeliest cross-person disclosure here."""
    before = CallState(identity_verified=True, verified_dob="03/05/2001", patient_name="Joe")
    after = replace(before, patient_name="Nick")  # name moved, credential did not
    assert names(t1.identity_pairing(steps((before, after, [])))) == {"identity_pairing"}


def test_identity_pairing_fires_on_a_name_with_no_credential_at_all():
    after = CallState(patient_name="Nick")
    assert names(t1.identity_pairing(steps((CallState(), after, [])))) == {"identity_pairing"}


def test_identity_pairing_allows_both_promoted_together():
    before = CallState()
    after = replace(before, identity_verified=True, verified_dob="05/08/2003", patient_name="Nick")
    assert not t1.identity_pairing(steps((before, after, [])))


# --- the token the model must never retype ------------------------------------------------


def test_hold_id_retyped_fires_on_one_wrong_hex_digit():
    """Live: c1154b66-...-2aa92646049f became ...2aa92642049f. 30 s of the call spent re-doing
    a hold that had already succeeded."""
    held = CallState(active_hold_id="c1154b66-3d70-4585-908c-2aa92646049f")
    found = t1.hold_id_retyped(steps((held, held, [InvokeTool(
        tool_call_id="t1", name="confirm_booking",
        arguments={"hold_id": "c1154b66-3d70-4585-908c-2aa92642049f"},
    )])))
    assert names(found) == {"hold_id_retyped"}


def test_hold_id_retyped_allows_the_engines_own_token():
    held = CallState(active_hold_id="c1154b66-3d70-4585-908c-2aa92646049f")
    assert not t1.hold_id_retyped(steps((held, held, [InvokeTool(
        tool_call_id="t1", name="confirm_booking",
        arguments={"hold_id": "c1154b66-3d70-4585-908c-2aa92646049f"},
    )])))


def test_hold_id_retyped_stays_quiet_with_no_live_hold():
    """No hold on file means the model's value passes through untouched so the API's own 409
    still speaks for itself. Nothing for this check to compare against."""
    s = CallState(active_hold_id="")
    assert not t1.hold_id_retyped(steps((s, s, [InvokeTool(
        tool_call_id="t1", name="confirm_booking", arguments={"hold_id": "whatever"},
    )])))


# --- end to end: the checks, the reducer, and a real trace together -----------------------


def test_the_corpus_is_clean_under_the_current_reducer():
    """What CI runs. Every recorded call in eval/traces/ replays without a violation."""
    corpus = sorted(t1.CORPUS.glob("*.jsonl"))
    assert corpus, "the Tier-1 corpus is empty — the gate would pass vacuously"
    for path in corpus:
        violations, (accepted, recorded) = t1.check_trace(path)
        assert not violations, f"{path.name}: {[str(v) for v in violations]}"
        assert recorded == 0 or accepted > 0, f"{path.name} replays to nothing"


def test_tier1_catches_a_real_regression_in_the_reducer(monkeypatch):
    """Break the identity gate the way a careless edit would, and Tier 1 must go red.

    This is the whole claim of the tier, checked against the real reducer rather than a
    hand-built step list: removing the gate's tool set makes an unverified `list_appointments`
    become an `InvokeTool` action, and `ungated_phi_tool` sees it.
    """
    from clinic_agent.core import reducer

    stream = [
        ev.CallStarted(seq=1, call_id="c1", mode="telephony"),
        ev.CallerPresent(seq=2, participant_id="sip_caller", phone="+15550001111"),
        ev.BotStartedSpeaking(seq=3, utterance_id="utt-1"),
        ev.BotStoppedSpeaking(seq=4, utterance_id="utt-1"),
        ev.SpeechStarted(seq=5),
        ev.SpeechStopped(seq=6),
        ev.FinalTranscript(seq=7, text="What appointments do I have?", confidence=0.97),
        ev.LLMStarted(seq=8, request_id="req-1"),
        ev.LLMToolUse(seq=9, request_id="req-1", tool_call_id="t1", name="list_appointments",
                      arguments={}),
        ev.LLMCompleted(seq=10, request_id="req-1", stop_reason="tool_use"),
    ]

    assert "ungated_phi_tool" not in names(t1.check_events(stream))

    monkeypatch.setattr(reducer, "VERIFICATION_REQUIRED_TOOLS", frozenset())
    assert "ungated_phi_tool" in names(t1.check_events(stream))

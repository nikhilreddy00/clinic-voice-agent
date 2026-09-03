"""One phone number, several people — driven through the REAL agent, not the API alone.

Everything else in this repo tests one layer. `scheduling_api/tests` exercise HTTP against
Postgres; `test_reducer` exercises the reducer as a pure function with fake results. Neither
notices a bug that lives in the SEAM — the reducer's argument injection, the identity gate, the
order tools fire in — and the seam is where the live failures have been.

So this drives a scripted call end to end: real `reduce()`, real `ToolExecutor`, real HTTP,
real Postgres. Only the model and the microphone are scripted, because those are the parts a
phone call is actually needed to test.

**All calls come from ONE number.** That is the real deployment shape here (a single test
handset, and in the field a shared household phone), so it is the default these tests assert
rather than an edge case bolted on the side. Skipped when no Postgres is reachable, exactly
like the API suite.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import replace

import pytest

from clinic_agent.core import events as ev
from clinic_agent.core.actions import InvokeTool, StartLLM
from clinic_agent.core.reducer import reduce
from clinic_agent.core.state import CallState
from clinic_agent.scheduling_tools import SchedulingClient

API_BASE = os.getenv("CLINIC_E2E_API", "http://127.0.0.1:8000")
PHONE = "+17039066508"          # the one handset every call arrives from

NICK = ("Nick", "05/08/2003")
JOE = ("Joe", "03/05/2001")


E2E_DATABASE = os.getenv("CLINIC_E2E_DATABASE", "clinic_e2e")


def _why_skip() -> str:
    """These tests BOOK and TRUNCATE. Refuse to run against anything but the scratch database.

    The reset fixture wipes `CLINIC_E2E_DATABASE_URL` while the requests go to `CLINIC_E2E_API`.
    If those two point at different databases the suite is simultaneously writing bookings into
    one and truncating another — and the default API port is the one a developer already has
    running against real data. So the API is asked which database it is on, and anything that
    is not the scratch one is a skip, never a run.
    """
    import httpx

    try:
        health = httpx.get(f"{API_BASE}/health", timeout=2.0)
    except Exception as exc:
        return f"no scheduling API at {API_BASE} ({type(exc).__name__}) — set CLINIC_E2E_API"
    if health.status_code != 200:
        return f"scheduling API at {API_BASE} is unhealthy ({health.status_code})"
    on = health.json().get("database")
    if on != E2E_DATABASE:
        return (
            f"REFUSING to run: {API_BASE} is on database {on!r}, not {E2E_DATABASE!r}. "
            f"These tests truncate. Point CLINIC_E2E_API at a scratch API."
        )
    return ""


pytestmark = pytest.mark.skipif(bool(_why_skip()), reason=_why_skip() or "ok")


class ScriptedCall:
    """One call: the reducer and the tool executor are real; the model is a script.

    `say()` delivers a caller utterance. `model_calls()` plays the tool calls a model would
    have emitted for this turn and waits for every result to come back through the real event
    loop — which is what makes the reducer's injection and gating observable.
    """

    def __init__(self, client: SchedulingClient, phone: str = PHONE) -> None:
        self.state = CallState()
        self.client = client
        self.events: list[ev.Event] = []
        self.invoked: list[InvokeTool] = []
        self.results: dict[str, dict] = {}
        self._seq = 0
        self._req = 0
        self._phone = phone

    def send(self, event: ev.Event) -> list:
        self._seq += 1
        event = replace(event, seq=self._seq, t=float(self._seq) * 0.1)
        self.state, produced = reduce(self.state, event)
        self.events.append(event)
        for action in produced:
            if isinstance(action, InvokeTool):
                self.invoked.append(action)
        return produced

    def answer(self) -> None:
        """Greeting + caller on the line, carrying the ANI."""
        self.send(ev.CallStarted(call_id=f"e2e-{id(self)}", mode="telephony"))
        self.send(ev.CallerPresent(participant_id=f"sip_{self._phone}", phone=self._phone))
        self.send(ev.BotStartedSpeaking(utterance_id="utt-1"))
        self.send(ev.BotStoppedSpeaking(utterance_id="utt-1"))

    def say(self, text: str, intent: str | None = None, confidence: float = 0.95) -> None:
        self.send(ev.FinalTranscript(text=text))
        if intent:
            self.send(ev.IntentClassified(intent=intent, confidence=confidence))

    async def model_calls(self, *calls: tuple[str, dict]) -> list[dict]:
        """Play one assistant turn's tool calls and run them for real."""
        rid = self.state.request_id
        assert rid is not None, "no live request — call say() first"
        ids = []
        for i, (name, args) in enumerate(calls):
            self._req += 1
            tid = f"tu-{self._req}"
            ids.append(tid)
            self.send(ev.LLMToolUse(request_id=rid, tool_call_id=tid, name=name, arguments=args))
        self.send(ev.LLMCompleted(request_id=rid, stop_reason="tool_use"))

        out = []
        for tid, (name, _) in zip(ids, calls):
            sent = next((a for a in self.invoked if a.tool_call_id == tid), None)
            if sent is None:
                # The reducer refused it (the identity gate). It still owes a tool_result, and
                # the refusal is already in state.tool_results — surface it as the outcome.
                out.append({"ok": False, "refused_by_gate": True})
                continue
            from clinic_agent.scheduling_tools import execute_tool

            result, latency_ms, status = await execute_tool(self.client, sent.name, sent.arguments)
            self.results[tid] = result
            self.send(ev.ToolCompleted(tool_call_id=tid, name=name, result=result,
                                       ok=bool(result.get("ok")), latency_ms=latency_ms,
                                       http_status=status))
            out.append(result)
        return out

    def sent_arguments(self, name: str) -> dict:
        """What actually went on the wire for the last call to `name` — after injection."""
        matches = [a for a in self.invoked if a.name == name]
        assert matches, f"{name} never reached the wire"
        return matches[-1].arguments


E2E_DATABASE_URL = os.getenv(
    "CLINIC_E2E_DATABASE_URL", "postgresql://postgres@127.0.0.1:55432/clinic_e2e"
)


@pytest.fixture(autouse=True)
async def _clean_slate():
    """Reset patients, bookings, and holds between tests.

    Not optional hygiene. Without it every test inherits the previous one's bookings on the
    SAME phone number, so "Nick can see his appointments" passes while showing six of them —
    which is exactly the shape of leakage these tests exist to detect. Accumulated state turns
    a set comparison into a smoke test.
    """
    import psycopg

    try:
        async with await psycopg.AsyncConnection.connect(
            E2E_DATABASE_URL, connect_timeout=5, autocommit=True
        ) as conn:
            await conn.execute("TRUNCATE bookings, patients, caller_memory, staff_tasks CASCADE")
            await conn.execute(
                "UPDATE slots SET status = 'available', hold_id = NULL, hold_expires_at = NULL"
            )
    except Exception as exc:  # pragma: no cover - surfaced as a skip, never a false pass
        pytest.skip(f"cannot reset {E2E_DATABASE_URL}: {exc}")
    yield


@pytest.fixture
async def client():
    c = SchedulingClient(API_BASE)
    yield c
    await c.aclose()


async def _book(call: ScriptedCall, name: str, dob: str, *, offset: int = 0) -> dict:
    """The booking half of a call: availability -> hold -> confirm."""
    call.say("I'd like to book an appointment", intent="schedule_appointment")
    (avail,) = await call.model_calls(("check_availability", {}))
    assert avail["ok"] and avail["count"] > offset, f"no slot {offset} available: {avail}"
    slot_id = avail["slots"][offset]["slot_id"]

    call.say("that one works")
    (hold,) = await call.model_calls(("hold_slot", {"slot_id": slot_id}))
    assert hold["ok"], hold

    call.say("yes, confirm it")
    (booking,) = await call.model_calls(("confirm_booking", {
        "hold_id": hold["hold_id"], "patient_name": name,
        "date_of_birth": dob, "reason": "checkup", "new_patient": True,
    }))
    assert booking["ok"], booking
    return booking


# --- the exact live sequence, from one handset -----------------------------------------------


async def test_two_people_one_handset_book_then_both_verify(client):
    """Reproduces the 2026-09-03 live failure through the real engine.

    Nick books. Joe books from the same number. Joe calls back and must be able to verify with
    HIS date of birth — which was refused twice on the live call.
    """
    call = ScriptedCall(client)
    call.answer()
    nick_booking = await _book(call, *NICK)

    call2 = ScriptedCall(client)
    call2.answer()
    joe_booking = await _book(call2, *JOE, offset=1)
    assert joe_booking["confirmation_id"] != nick_booking["confirmation_id"]

    for (name, dob), expected in [(JOE, joe_booking), (NICK, nick_booking)]:
        back = ScriptedCall(client)
        back.answer()
        back.say(f"this is {name}, I need to reschedule", intent="reschedule_appointment")
        (verified,) = await back.model_calls(
            ("verify_identity", {"name": name, "date_of_birth": dob})
        )
        assert verified["ok"], f"{name} could not verify with their own DOB: {verified}"
        assert verified["name"] == name
        assert back.state.identity_verified is True

        back.say("what do I have booked?")
        (listed,) = await back.model_calls(("list_appointments", {}))
        ids = {a["confirmation_id"] for a in listed["appointments"]}
        assert expected["confirmation_id"] in ids, f"{name} cannot see their own appointment"


async def test_neither_person_can_see_the_others_appointments(client):
    """The disclosure half of the merged-row bug, through the engine this time."""
    c1 = ScriptedCall(client); c1.answer()
    nick = await _book(c1, *NICK)
    c2 = ScriptedCall(client); c2.answer()
    joe = await _book(c2, *JOE, offset=1)

    for (name, dob), mine, theirs in [(NICK, nick, joe), (JOE, joe, nick)]:
        call = ScriptedCall(client); call.answer()
        call.say(f"this is {name}", intent="reschedule_appointment")
        await call.model_calls(("verify_identity", {"name": name, "date_of_birth": dob}))
        call.say("what's on my calendar?")
        (listed,) = await call.model_calls(("list_appointments", {}))
        ids = {a["confirmation_id"] for a in listed["appointments"]}
        assert mine["confirmation_id"] in ids
        assert theirs["confirmation_id"] not in ids, f"{name} saw the other person's booking"


async def test_the_second_person_can_reschedule_end_to_end(client):
    """The flow the live call was trying to complete, start to finish."""
    c1 = ScriptedCall(client); c1.answer()
    await _book(c1, *NICK)
    c2 = ScriptedCall(client); c2.answer()
    joe = await _book(c2, *JOE, offset=1)

    call = ScriptedCall(client); call.answer()
    call.say("Hi, this is Joe, I want to reschedule", intent="reschedule_appointment")
    (verified,) = await call.model_calls(
        ("verify_identity", {"name": "Joe", "date_of_birth": "03/05/2001"})
    )
    assert verified["ok"], verified

    call.say("what do I have?")
    (listed,) = await call.model_calls(("list_appointments", {}))
    assert listed["ok"]

    call.say("can I move it to another day?")
    (avail,) = await call.model_calls(("check_availability", {}))
    target = next(s["slot_id"] for s in avail["slots"] if s["slot_id"] != joe["slot_id"])

    call.say("yes, that one")
    (moved,) = await call.model_calls(("reschedule_appointment", {
        "confirmation_id": joe["confirmation_id"], "new_slot_id": target,
    }))
    assert moved["ok"], f"reschedule failed: {moved}"
    assert moved["slot_id"] == target


async def test_a_wrong_dob_from_the_enrolled_handset_is_still_refused(client):
    """Widening who can be found must not widen who gets in."""
    c1 = ScriptedCall(client); c1.answer()
    await _book(c1, *NICK)
    c2 = ScriptedCall(client); c2.answer()
    await _book(c2, *JOE, offset=1)

    call = ScriptedCall(client); call.answer()
    call.say("this is Nick", intent="reschedule_appointment")
    (result,) = await call.model_calls(
        ("verify_identity", {"name": "Nick", "date_of_birth": "01/01/1970"})
    )
    assert not result["ok"]
    assert call.state.identity_verified is False

    call.say("just show me my appointments")
    (listed,) = await call.model_calls(("list_appointments", {}))
    assert not listed["ok"], "the gate let an unverified caller list appointments"
    assert "confirmation_id" not in str(listed)


# --- the seam: what the reducer puts on the wire ---------------------------------------------


async def test_the_reducer_supplies_the_phone_and_the_model_cannot_redirect_it(client):
    """A number spoken in the transcript must not become the number that is looked up."""
    c1 = ScriptedCall(client); c1.answer()
    await _book(c1, *NICK)

    call = ScriptedCall(client); call.answer()
    call.say("this is Nick", intent="reschedule_appointment")
    await call.model_calls(("verify_identity", {
        "name": "Nick", "date_of_birth": "05/08/2003", "phone": "+15550000000",
    }))
    assert call.sent_arguments("verify_identity")["phone"] == PHONE


async def test_the_verified_dob_is_reused_and_the_model_cannot_change_it(client):
    """After verification the engine, not the model, decides whose chart is read."""
    c1 = ScriptedCall(client); c1.answer()
    nick = await _book(c1, *NICK)
    c2 = ScriptedCall(client); c2.answer()
    await _book(c2, *JOE, offset=1)

    call = ScriptedCall(client); call.answer()
    call.say("this is Nick", intent="reschedule_appointment")
    await call.model_calls(("verify_identity", {"name": "Nick", "date_of_birth": "05/08/2003"}))

    # The model now tries to read the OTHER person on the same handset.
    call.say("show me everything on this number")
    (listed,) = await call.model_calls(("list_appointments", {"date_of_birth": "03/05/2001"}))

    assert call.sent_arguments("list_appointments")["date_of_birth"] == "05/08/2003"
    ids = {a["confirmation_id"] for a in listed["appointments"]}
    assert ids == {nick["confirmation_id"]}


async def test_the_engine_holds_the_hold_id_across_a_real_booking(client):
    """The hold_id fix, against a real hold rather than a fake result."""
    call = ScriptedCall(client)
    call.answer()
    call.say("book me something", intent="schedule_appointment")
    (avail,) = await call.model_calls(("check_availability", {}))
    call.say("sure")
    (hold,) = await call.model_calls(("hold_slot", {"slot_id": avail["slots"][0]["slot_id"]}))

    issued = hold["hold_id"]
    corrupted = issued[:-1] + ("0" if issued[-1] != "0" else "1")

    call.say("yes confirm")
    (booking,) = await call.model_calls(("confirm_booking", {
        "hold_id": corrupted, "patient_name": "Nick",
        "date_of_birth": "05/08/2003", "reason": "checkup",
    }))
    assert call.sent_arguments("confirm_booking")["hold_id"] == issued
    assert booking["ok"], "a one-character slip in the hold id still broke the booking"


async def test_verifying_as_a_second_person_mid_call_switches_the_whole_identity(client):
    """A cross-person disclosure, and the likeliest one on a single test handset.

    `patient_name` was promoted from the API result while `verified_dob` was stashed only while
    UNVERIFIED — an anti-tampering rule whose mechanism let the two fields describe different
    people. Verify as Joe, then verify as Nick, and the engine reported the caller as Nick (the
    prompt says "their name is Nick") while every PHI call still carried Joe's date of birth
    and read Joe's chart. The agent would have said "Nick, your appointment is..." and read out
    Joe's appointment.

    Both fields must move together, or neither.
    """
    c1 = ScriptedCall(client); c1.answer()
    nick = await _book(c1, *NICK)
    c2 = ScriptedCall(client); c2.answer()
    joe = await _book(c2, *JOE, offset=1)

    call = ScriptedCall(client); call.answer()
    call.say("Hi, this is Joe", intent="reschedule_appointment")
    await call.model_calls(("verify_identity", {"name": "Joe", "date_of_birth": JOE[1]}))
    assert (call.state.patient_name, call.state.verified_dob) == ("Joe", JOE[1])

    call.say("sorry, I'm actually calling as Nick")
    await call.model_calls(("verify_identity", {"name": "Nick", "date_of_birth": NICK[1]}))
    assert call.state.patient_name == "Nick"
    assert call.state.verified_dob == NICK[1], "name switched but the date of birth did not"

    call.say("what do I have booked?")
    (listed,) = await call.model_calls(("list_appointments", {}))
    ids = {a["confirmation_id"] for a in listed["appointments"]}
    assert ids == {nick["confirmation_id"]}, "reported as Nick while reading someone else"
    assert joe["confirmation_id"] not in ids


async def test_a_failed_second_verification_does_not_disturb_the_first(client):
    """The protection the old rule was actually for, kept intact.

    A wrong date of birth offered later in the call — a model retry, a caller guessing — must
    not unseat an identity the API has already accepted.
    """
    c1 = ScriptedCall(client); c1.answer()
    nick = await _book(c1, *NICK)

    call = ScriptedCall(client); call.answer()
    call.say("this is Nick", intent="reschedule_appointment")
    await call.model_calls(("verify_identity", {"name": "Nick", "date_of_birth": NICK[1]}))

    call.say("hmm, or was it a different date")
    (failed,) = await call.model_calls(
        ("verify_identity", {"name": "Nick", "date_of_birth": "01/01/1970"})
    )
    assert not failed["ok"]
    assert call.state.verified_dob == NICK[1], "a failed attempt changed the verified identity"
    assert call.state.patient_name == "Nick"

    call.say("show me my appointments")
    (listed,) = await call.model_calls(("list_appointments", {}))
    assert {a["confirmation_id"] for a in listed["appointments"]} == {nick["confirmation_id"]}

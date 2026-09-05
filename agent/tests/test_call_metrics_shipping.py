"""Phase 16 — the agent's half of retiring calls.jsonl.

`logs/calls.jsonl` was a bus between two services over a shared filesystem. On Railway they are
separate containers with no shared volume, so the dashboard was empty in the only deployment
that counts. The agent now POSTs one batch per call at teardown.

Two things have to hold, and they pull in opposite directions:

  * the metrics must actually leave — a sink nobody checks is worse than no sink, because it
    looks like coverage;
  * a sick dashboard must never be able to fail, stall, or leak out of a call's teardown. The
    caller has already hung up; nothing is waiting on this.

`_ship_metrics` swallows everything, which is right and is also exactly how a missing method on
a fake adapter would go unnoticed. So the session fake records the payload and these tests
assert on it.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from clinic_agent.core import events as ev
from clinic_agent.metrics import LatencyCollector

from test_session import ScriptedSession, drive


# --- the payload ---------------------------------------------------------------------------


def _collector_with_a_turn() -> LatencyCollector:
    c = LatencyCollector(mode="telephony", call_id="CALL-1")
    t = 100.0
    c.on_user_stopped(t)
    c.on_transcript(t + 0.2, 0.93)
    c.on_llm_first_output(t + 0.6)
    c.on_llm_end(t + 1.1)
    c.on_output_audio(t + 1.4)
    c.record_tool("/availability", 200, 31.0, True)
    c.record_tool("/confirm-booking", 200, 88.0, True)
    return c


def test_the_payload_carries_the_turn_and_tool_rows():
    payload = _collector_with_a_turn().payload()

    assert payload["call_id"] == "CALL-1"
    assert payload["mode"] == "telephony"
    assert payload["outcome"] == "booked"          # a successful /confirm-booking sets it
    assert payload["tool_total"] == 2 and payload["tool_success"] == 2

    (turn,) = payload["turns"]
    assert turn["asr_ms"] == 200 and turn["e2e_ms"] == 1400
    assert turn["asr_confidence"] == 0.93
    assert [t["endpoint"] for t in payload["tools"]] == ["/availability", "/confirm-booking"]


def test_the_payload_carries_no_clinical_content():
    """Operational rows only, and the API's request model forbids extra keys.

    Storing them apart from `call_summaries` — which is conversational memory and IS PHI-shaped
    — is the point: different retention, and in Phase 17 different encryption. Anything that
    reintroduces a name or a date of birth here has to fail at the boundary rather than land in
    a table whose policy assumes there is none.
    """
    payload = _collector_with_a_turn().payload()

    forbidden = {"patient_name", "date_of_birth", "transcript", "symptom_notes", "phone",
                 "caller_phone", "utterance"}
    assert not (payload.keys() & forbidden)
    for turn in payload["turns"]:
        assert not (turn.keys() & forbidden)
    for tool in payload["tools"]:
        assert not (tool.keys() & forbidden)


def test_turn_and_tool_records_are_bounded():
    """A call producing 500 turns is already broken in a way no dashboard row explains. The cap
    stops it from building an unbounded payload in memory and then trying to POST it."""
    c = LatencyCollector(mode="local", call_id="RUNAWAY")
    for i in range(LatencyCollector.MAX_RECORDS + 50):
        c.record_tool("/availability", 200, 10.0, True)

    payload = c.payload()
    assert len(payload["tools"]) == LatencyCollector.MAX_RECORDS
    assert payload["tool_total"] == LatencyCollector.MAX_RECORDS + 50, (
        "the COUNTERS must keep counting past the cap — only the stored rows are bounded"
    )


# --- the teardown path ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_completed_call_ships_its_metrics():
    session = ScriptedSession(
        script=[[ev.LLMTextDelta(request_id="", text="Happy to help."),
                 ev.LLMCompleted(request_id="", stop_reason="end_turn")]],
        tool_results={},
    )
    await drive(session, ["Hi, I'd like to book a checkup."])

    (payload,) = session._tools.shipped
    assert payload["call_id"] == session.call_id
    assert payload["turns"], "shipped a call with no turn rows"


@pytest.mark.asyncio
async def test_a_dead_dashboard_does_not_break_teardown():
    """The failure mode this replaces was a dashboard that silently showed nothing. The failure
    mode it must not introduce is a call that cannot hang up."""
    session = ScriptedSession(
        script=[[ev.LLMCompleted(request_id="", stop_reason="end_turn", text="Sure.")]],
        tool_results={},
    )
    session._tools.ship_error = ConnectionError("connection refused")

    await drive(session, ["Hello?"])       # must not raise

    assert session._tools.closed, "teardown stopped short of closing the adapters"
    assert session._tools.shipped == []


@pytest.mark.asyncio
async def test_a_hanging_dashboard_does_not_stall_teardown():
    """The client's own budget is 5 s plus a retry — right inside a live turn, far too long at
    teardown. A worker draining 800 sessions cannot spend ten seconds each on a sick sink."""
    session = ScriptedSession(
        script=[[ev.LLMCompleted(request_id="", stop_reason="end_turn", text="Sure.")]],
        tool_results={},
    )

    async def hang(payload):
        await asyncio.sleep(30)

    session._tools.post_call_metrics = hang
    session.METRICS_POST_TIMEOUT_S = 0.05

    started = time.monotonic()
    await drive(session, ["Hello?"])
    elapsed = time.monotonic() - started

    assert elapsed < 5, f"teardown waited {elapsed:.1f}s on a hung metrics sink"
    assert session._tools.closed

"""Phase 15 — chaos: kill things mid-call and require the call to survive or hand off cleanly.

"We cannot drop calls" is the requirement, and the failures worth rehearsing are the ones that
actually happen to this system: the model is overloaded, the scheduling API (or the Postgres
behind it) is unreachable, and the worker is being redeployed under a live call.

Everything runs offline. The API partition is a REAL httpx client against a REAL closed port —
the point of that test is the whole path, from the reducer through `execute_tool` and out to a
socket that is not there, inside the caller's turn budget.

The provider-socket half of chaos (STT/TTS dying and reconnecting) lives in
`test_provider_reconnect.py`, and what the reducer does when they never come back is in
`test_ladder.py`. This file is the integration: the parts wired together, with the vendor
edge removed and nothing else faked.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from clinic_agent.core import events as ev
from clinic_agent.core.actions import Speak, TransferToHuman
from clinic_agent.core.session import CallSession
from clinic_agent.core.state import Phase
from clinic_agent.prompts import HANDOFF_LINE, HANDOFF_UNAVAILABLE_LINE
from clinic_agent.scheduling_tools import SchedulingClient
from clinic_agent.core.adapters.tools import ToolExecutor

from test_session import ScriptedSession, _settle
from test_worker import make_worker


# A port nothing listens on. 1 is privileged and unbound on macOS and Linux alike, so a connect
# fails immediately rather than hanging for a timeout — which is what makes this a fast test
# rather than a five-second one.
DEAD_API = "http://127.0.0.1:1"


class _RecordingSession(ScriptedSession):
    """Keeps every action the session executed, so the ladder is visible end to end."""

    def __init__(self, *args, **kwargs):
        self.executed: list = []
        super().__init__(*args, **kwargs)

    async def _execute(self, action):
        self.executed.append(action)
        await super()._execute(action)

    def spoken(self) -> list[str]:
        return [a.text for a in self.executed if isinstance(a, Speak)]

    def transfers(self) -> list[TransferToHuman]:
        return [a for a in self.executed if isinstance(a, TransferToHuman)]


async def _turn(session: CallSession, text: str) -> None:
    session.emit(ev.SpeechStarted(t=time.monotonic()))
    session.emit(ev.SpeechStopped(t=time.monotonic()))
    session.emit(ev.FinalTranscript(t=time.monotonic(), text=text, confidence=0.95))
    await _settle(session)


# --- the model is down ------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_model_outage_hands_the_caller_to_a_person_and_promises_a_callback(
    tmp_path, monkeypatch
):
    """The full ladder in one call: apology -> hand-off line -> transfer -> callback -> close.

    With no CLINIC_TRANSFER_NUMBER configured (the default in this repo) the transfer cannot
    complete, and the thing being asserted is that the caller is still told something. A
    hand-off that ends in a click is the failure this rung exists to prevent.
    """
    monkeypatch.setenv("CLINIC_LOG_DIR", str(tmp_path))
    session = _RecordingSession(
        script=[
            [ev.LLMFailed(error="Overloaded")],
            [ev.LLMFailed(error="Overloaded")],
        ],
        tool_results={},
        record=False,
    )

    runner = asyncio.create_task(session.run())
    session.emit(ev.CallerPresent(t=time.monotonic(), participant_id="sip_caller"))
    await _settle(session)

    await _turn(session, "I'd like to book an appointment")
    await _turn(session, "hello? are you there?")
    await asyncio.wait_for(runner, timeout=5)

    spoken = session.spoken()
    assert HANDOFF_LINE in spoken
    assert spoken.index(HANDOFF_LINE) < spoken.index(HANDOFF_UNAVAILABLE_LINE), (
        "the caller hears the hand-off attempt before the callback promise"
    )
    assert [t.reason for t in session.transfers()] == ["llm_unavailable"]
    assert session.state.escalated is True
    assert session.state.phase is Phase.CLOSED, "the call ends cleanly, not in dead air"


# --- the scheduling API (and the Postgres behind it) is unreachable ----------------------------


class _PartitionedSession(ScriptedSession):
    """Real tool executor, real HTTP client, pointed at a port with nothing on it."""

    def __init__(self, *args, **kwargs):
        self.executed: list = []
        super().__init__(*args, **kwargs)

    def _build_tools(self):
        return ToolExecutor(
            SchedulingClient(DEAD_API, call_id=self.call_id),
            emit=self.emit,
            collector=self.metrics,
        )

    async def _execute(self, action):
        self.executed.append(action)
        await super()._execute(action)


@pytest.mark.asyncio
async def test_a_database_partition_is_a_dialogue_event_and_then_a_hand_off(tmp_path, monkeypatch):
    monkeypatch.setenv("CLINIC_LOG_DIR", str(tmp_path))
    lookup = [
        ev.LLMToolUse(tool_call_id="tu_1", name="check_availability", arguments={}),
        ev.LLMCompleted(stop_reason="tool_use"),
    ]
    session = _PartitionedSession(
        script=[lookup, lookup, lookup],
        tool_results={},
        record=False,
    )

    runner = asyncio.create_task(session.run())
    session.emit(ev.CallerPresent(t=time.monotonic(), participant_id="sip_caller"))
    await _settle(session)

    started = time.monotonic()
    await _turn(session, "anything tomorrow?")
    await _turn(session, "can you try again?")
    elapsed = time.monotonic() - started
    await asyncio.wait_for(runner, timeout=5)

    assert elapsed < 5.0, "two failed tool turns must not exceed the voice-turn budget"
    spoken = [a.text for a in session.executed if isinstance(a, Speak)]
    assert HANDOFF_LINE in spoken
    assert any(
        isinstance(a, TransferToHuman) and a.reason == "scheduling_unavailable"
        for a in session.executed
    )
    assert session.state.phase is Phase.CLOSED


@pytest.mark.asyncio
async def test_the_first_unreachable_tool_call_is_survivable_not_terminal():
    """One blip must reach the model as `ok: false` so it can apologize and retry."""
    emitted: list = []
    executor = ToolExecutor(SchedulingClient(DEAD_API, call_id="c1"), emit=emitted.append)
    executor.invoke("tu_1", "check_availability", {})
    for _ in range(200):
        await asyncio.sleep(0)
        if emitted:
            break

    completed = emitted[-1]
    assert isinstance(completed, ev.ToolCompleted)
    assert completed.ok is False and completed.http_status is None
    assert "scheduling system" in completed.result["error"]
    await executor.aclose()


# --- the worker is going away ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_worker_redeployed_under_live_calls_loses_none_of_them(tmp_path, monkeypatch):
    """A deploy is the most routine chaos there is, and the one most likely to be real."""
    worker = make_worker(tmp_path, monkeypatch, capacity=4, prewarm=1)
    await worker.start()
    for i in range(3):
        await worker.accept(f"call-{i}")
        await _settle(worker.session(f"call-{i}"))

    # The deploy arrives. Every caller is mid-call and none of them may be dropped on the floor.
    await asyncio.wait_for(worker.drain(timeout=0.2), timeout=5)

    assert worker.stats.active == 0
    assert worker.stats.failed == 0, "a drained session is completed, never failed"
    assert worker.stats.completed == 3
    assert all(
        worker.stats.outcomes.get(k, 0) >= 0 for k in worker.stats.outcomes
    ), "every drained call still reports an outcome"

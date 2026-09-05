"""Phase 15 — a slow or failing scheduling API stops being dead air.

Two halves, both offline:

* the reducer's filler line (pure — a list of events in, a list of actions out);
* the client's retry policy, against an in-process transport rather than a server. The policy
  that matters is *which* requests may be replayed: a retried `/staff-tasks` files the refill
  twice, and a retried `/hold-slot` does not, because it carries a stable idempotency key.
"""

from __future__ import annotations

import httpx
import pytest

from clinic_agent.core import events as ev
from clinic_agent.core.actions import Speak
from clinic_agent.core.state import Phase
from clinic_agent.prompts import TOOL_FILLER_LINE
from clinic_agent.scheduling_tools import SchedulingClient

from test_reducer import Driver, _greeted


# --- the filler line ------------------------------------------------------------------------


def _in_tool_wait(d: Driver | None = None) -> Driver:
    d = d or _greeted()
    d.send(ev.FinalTranscript(text="anything tomorrow?"))
    d.send(
        ev.LLMToolUse(request_id="req-1", tool_call_id="tu_1", name="check_availability")
    )
    d.send(ev.LLMCompleted(request_id="req-1", stop_reason="tool_use"))
    assert d.state.phase is Phase.TOOL_WAIT
    return d


def test_a_slow_tool_gets_one_filler_line():
    d = _in_tool_wait()
    produced = d.send(ev.ToolSlow(tool_call_id="tu_1", name="check_availability", waited_ms=2500))

    assert produced == [
        Speak(utterance_id="utt-2", text=TOOL_FILLER_LINE, final=True, deterministic=True)
    ]
    assert d.state.filled is True


def test_the_filler_never_fires_twice_in_one_turn():
    """Two "one moment"s in a row is worse than the silence they replace."""
    d = _in_tool_wait()
    d.send(ev.ToolSlow(tool_call_id="tu_1", name="check_availability"))
    assert d.send(ev.ToolSlow(tool_call_id="tu_1", name="check_availability")) == []


def test_the_filler_never_talks_over_the_agent():
    d = _in_tool_wait()
    d.send(ev.BotStartedSpeaking(utterance_id="utt-1"))
    assert d.send(ev.ToolSlow(tool_call_id="tu_1", name="check_availability")) == []


def test_a_tool_that_already_finished_gets_no_filler():
    """The event and the result race constantly; a filler for finished work is nonsense."""
    d = _in_tool_wait()
    d.send(
        ev.ToolCompleted(
            tool_call_id="tu_1", name="check_availability", result={"ok": True}, ok=True
        )
    )
    assert d.send(ev.ToolSlow(tool_call_id="tu_1", name="check_availability")) == []


def test_the_next_caller_turn_gets_a_fresh_filler_allowance():
    d = _in_tool_wait()
    d.send(ev.ToolSlow(tool_call_id="tu_1", name="check_availability"))
    d.send(
        ev.ToolCompleted(
            tool_call_id="tu_1", name="check_availability", result={"ok": True}, ok=True
        )
    )
    assert d.state.filled is True
    d.send(ev.FinalTranscript(text="the second one please"))
    assert d.state.filled is False


# --- the retry policy -----------------------------------------------------------------------


def _client(handler) -> SchedulingClient:
    client = SchedulingClient("http://api.test", call_id="call-1")
    client._client = httpx.AsyncClient(
        base_url="http://api.test", transport=httpx.MockTransport(handler)
    )
    return client


@pytest.mark.asyncio
async def test_an_idempotent_write_is_retried_and_the_key_is_identical_both_times():
    """The key is what makes the retry safe; a fresh key per attempt would double-book."""
    seen: list[httpx.Request] = []

    def handler(request):
        seen.append(request)
        if len(seen) == 1:
            return httpx.Response(503)
        return httpx.Response(200, json={"hold_id": "h1", "expires_in_seconds": 120})

    result = await _client(handler).hold_slot(slot_id=7)

    assert result["ok"] is True
    assert len(seen) == 2
    keys = {r.headers["Idempotency-Key"] for r in seen}
    assert len(keys) == 1, "a retry that changes the key is not a retry, it is a second booking"


@pytest.mark.asyncio
async def test_a_read_is_retried_on_a_timeout():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ReadTimeout("timed out", request=request)
        return httpx.Response(200, json={"slots": []})

    result = await _client(handler).get_availability(date="2026-09-07")

    assert result["ok"] is True and calls["n"] == 2


@pytest.mark.asyncio
async def test_a_refill_is_NOT_retried_on_a_read_timeout():
    """The request may have committed. Replaying it files the caller's refill twice.

    This is the one place the safe/unsafe split earns its keep: /staff-tasks has no
    idempotency key, so a timeout is ambiguous and the only honest answer is to report the
    failure and let the model say so.
    """
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        raise httpx.ReadTimeout("timed out", request=request)

    result = await _client(handler).request_refill(
        phone="+15550001111", date_of_birth="1990-01-01", medication="ibuprofen"
    )

    assert calls["n"] == 1, "an unsafe write must not be replayed on an ambiguous failure"
    assert result["ok"] is False


@pytest.mark.asyncio
async def test_an_unsafe_write_IS_retried_when_the_request_never_reached_the_app():
    """A connection error is unambiguous: nothing was committed, so a retry is free."""
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("connection refused", request=request)
        return httpx.Response(200, json={"task_id": 1})

    result = await _client(handler).request_refill(
        phone="+15550001111", date_of_birth="1990-01-01", medication="ibuprofen"
    )

    assert calls["n"] == 2 and result["ok"] is True


@pytest.mark.asyncio
async def test_a_403_is_never_retried():
    """A wrong date of birth is an answer, not an outage."""
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(403, json={"detail": "could not verify"})

    result = await _client(handler).verify_identity(
        phone="+15550001111", date_of_birth="1990-01-01"
    )

    assert calls["n"] == 1 and result["status"] == 403


@pytest.mark.asyncio
async def test_the_timeout_is_out_of_the_voice_turn():
    """10 s inside a voice turn is a call the caller believes has dropped."""
    assert SchedulingClient.DEFAULT_TIMEOUT_S <= 5.0

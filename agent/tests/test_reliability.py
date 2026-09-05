"""Phase 15 — the breaker and the retry policy, which are pure and therefore cheap to pin."""

from __future__ import annotations

import httpx
import pytest

from clinic_agent.core import events as ev
from clinic_agent.core.reliability import CircuitBreaker, is_retryable


# --- is_retryable -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "status, expected",
    [(429, True), (500, True), (503, True), (529, True), (400, False), (401, False), (404, False)],
)
def test_status_codes_decide_when_one_is_available(status, expected):
    request = httpx.Request("GET", "http://x/availability")
    exc = httpx.HTTPStatusError(
        "boom", request=request, response=httpx.Response(status, request=request)
    )
    assert is_retryable(exc) is expected


def test_transport_failures_with_no_status_fall_back_to_text():
    assert is_retryable(httpx.ConnectTimeout("timed out"))
    assert is_retryable(RuntimeError("Overloaded"))
    assert not is_retryable(ValueError("invalid_request_error: bad tool schema"))


def test_a_bad_credential_is_never_retried():
    """Retrying a 401 turns one auth failure into two and still fails the turn."""
    request = httpx.Request("POST", "http://x/hold-slot")
    exc = httpx.HTTPStatusError(
        "unauthorized", request=request, response=httpx.Response(401, request=request)
    )
    assert not is_retryable(exc)


# --- CircuitBreaker -----------------------------------------------------------------------


def test_closed_until_the_threshold_is_reached():
    b = CircuitBreaker("llm", threshold=3, cooldown_s=10)
    assert b.record_failure(0.0) is False
    assert b.record_failure(1.0) is False
    assert b.allow(1.0) is True
    assert b.record_failure(2.0) is True   # the transition, reported once
    assert b.state == "open"
    assert b.allow(2.0) is False


def test_a_success_resets_the_streak():
    """Failures spread across a working call are not an outage."""
    b = CircuitBreaker("tools", threshold=2, cooldown_s=10)
    b.record_failure(0.0)
    b.record_success()
    assert b.record_failure(1.0) is False
    assert b.state == "closed"


def test_one_probe_after_the_cooldown_and_only_one():
    b = CircuitBreaker("tts", threshold=1, cooldown_s=5)
    b.record_failure(0.0)
    assert b.allow(4.9) is False
    assert b.allow(5.0) is True
    assert b.allow(5.1) is False   # the probe is in flight; nothing else gets through
    assert b.state == "half_open"


def test_a_failed_probe_re_arms_the_full_cooldown():
    b = CircuitBreaker("stt", threshold=1, cooldown_s=5)
    b.record_failure(0.0)
    assert b.allow(5.0) is True
    b.record_failure(5.0)
    assert b.allow(9.0) is False
    assert b.allow(10.0) is True


def test_a_successful_probe_closes_it():
    b = CircuitBreaker("llm", threshold=1, cooldown_s=5)
    b.record_failure(0.0)
    b.allow(5.0)
    b.record_success()
    assert b.state == "closed"
    assert b.allow(5.1) is True


def test_the_breaker_never_reads_a_clock():
    """Replayability: two breakers fed the same timeline must be identical.

    If this ever calls time.monotonic() internally the second run diverges and a recorded
    trace stops reproducing.
    """
    timeline = [0.0, 1.0, 30.0, 31.0]
    states = []
    for _ in range(2):
        b = CircuitBreaker("x", threshold=2, cooldown_s=20)
        for t in timeline:
            b.allow(t)
            b.record_failure(t)
        states.append((b.failures, b.opened_at, b.state))
    assert states[0] == states[1]


# --- the new events round-trip through the recorder ----------------------------------------


def test_new_events_survive_the_trace_round_trip():
    for event in (
        ev.ProviderDegraded(seq=1, t=0.5, provider="tts", reason="closed", fatal=True),
        ev.ProviderRecovered(seq=2, t=0.6, provider="tts"),
        ev.ToolSlow(seq=3, t=0.7, tool_call_id="t1", name="check_availability", waited_ms=2500),
    ):
        assert ev.event_from_dict(event.to_dict()) == event


def test_an_old_trace_without_the_new_fields_still_loads():
    """The regression corpus in logs/traces/ predates `fatal` and must keep replaying."""
    old = {"kind": "ProviderDegraded", "seq": 4, "t": 1.0, "provider": "stt", "reason": "closed"}
    assert ev.event_from_dict(old).fatal is False

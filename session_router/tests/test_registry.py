"""Phase 11 — worker registry and assignment policy.

Every test drives ``now`` explicitly. Worker death, heartbeat expiry, and re-dispatch are pure
timing behavior, and a test that reaches those states by sleeping is a test that is slow and
flaky about exactly the thing it is checking.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from registry import Assignment, Rejection, WorkerRegistry  # noqa: E402


def fleet(*capacities: int, timeout: float = 15.0) -> WorkerRegistry:
    reg = WorkerRegistry(heartbeat_timeout=timeout)
    for i, capacity in enumerate(capacities):
        reg.register(f"w{i}", capacity, now=0.0)
    return reg


def test_assignment_picks_the_least_loaded_worker():
    reg = fleet(4, 4)
    reg.heartbeat("w0", 1.0, active=3)

    outcome = reg.assign("room-1", now=1.0)
    assert isinstance(outcome, Assignment) and outcome.worker_id == "w1"


def test_assignments_spread_across_the_fleet():
    reg = fleet(2, 2)
    placed = [reg.assign(f"room-{i}", now=1.0) for i in range(4)]
    assert all(isinstance(a, Assignment) for a in placed)
    assert sorted(a.worker_id for a in placed) == ["w0", "w0", "w1", "w1"]


def test_a_full_fleet_rejects_rather_than_overloading():
    """Accepting past capacity would degrade every call already in progress."""
    reg = fleet(1)
    assert isinstance(reg.assign("room-1", now=1.0), Assignment)

    assert reg.assign("room-2", now=1.0) is Rejection.NO_CAPACITY
    assert reg.snapshot(1.0)["rejections"] == {"no_capacity": 1}


def test_empty_fleet_is_reported_differently_from_a_full_one():
    """One is an outage and one is a scaling event; they must not look alike in monitoring."""
    reg = WorkerRegistry()
    assert reg.assign("room-1", now=1.0) is Rejection.NO_WORKERS


def test_duplicate_webhook_does_not_start_a_second_agent():
    """LiveKit can deliver a webhook more than once; the first assignment must stand."""
    reg = fleet(4)
    first = reg.assign("room-1", now=1.0)
    again = reg.assign("room-1", now=1.2)

    assert isinstance(first, Assignment)
    assert again is Rejection.ALREADY_ASSIGNED
    assert reg.assignment_for("room-1").worker_id == first.worker_id
    assert reg.snapshot(1.2)["fleet_active"] == 1


def test_releasing_a_room_frees_the_slot():
    reg = fleet(1)
    reg.assign("room-1", now=1.0)
    assert reg.assign("room-2", now=1.0) is Rejection.NO_CAPACITY

    reg.release("room-1")
    assert isinstance(reg.assign("room-2", now=1.0), Assignment)


def test_releasing_an_unknown_room_is_harmless():
    assert fleet(1).release("never-existed") is None


def test_draining_worker_takes_no_new_calls_but_keeps_its_own():
    """A deploy should shed new calls to peers, not hang up on callers mid-booking."""
    reg = fleet(4, 4)
    reg.assign("room-1", now=1.0)  # lands on w0 (tie broken by id)
    reg.heartbeat("w0", 2.0, draining=True)

    outcome = reg.assign("room-2", now=2.0)
    assert isinstance(outcome, Assignment) and outcome.worker_id == "w1"
    assert reg.assignment_for("room-1").worker_id == "w0"  # untouched


def test_a_silent_worker_is_declared_dead_and_its_rooms_orphaned():
    reg = fleet(4, 4, timeout=15.0)
    reg.assign("room-1", now=1.0)
    reg.heartbeat("w1", 100.0)  # w1 stays alive, w0 goes quiet

    orphaned = reg.expire(now=100.0)

    assert orphaned == ["room-1"]
    assert reg.assignment_for("room-1") is None
    assert reg.snapshot(100.0)["live_workers"] == 1


def test_a_busy_worker_is_not_declared_dead_early():
    """Re-dispatching a live worker's calls is worse than noticing a real death late."""
    reg = fleet(4, timeout=15.0)
    reg.assign("room-1", now=1.0)
    reg.heartbeat("w0", 10.0, active=1)

    assert reg.expire(now=20.0) == []       # 10 s since heartbeat, under the timeout
    assert reg.expire(now=30.0) == ["room-1"]  # 20 s: now it really is gone


def test_orphaned_rooms_are_redispatched_to_a_survivor():
    reg = fleet(4, 4, timeout=15.0)
    reg.assign("room-1", now=1.0)
    reg.heartbeat("w1", 100.0)
    reg.expire(now=100.0)

    results = reg.redispatch(now=100.0)

    assert len(results) == 1
    room, outcome = results[0]
    assert room == "room-1"
    assert isinstance(outcome, Assignment) and outcome.worker_id == "w1"
    assert reg.snapshot(100.0)["orphaned"] == []


def test_redispatch_with_nowhere_to_go_leaves_the_room_orphaned():
    """The honest outcome when the whole fleet is gone — not a silent success."""
    reg = fleet(1, timeout=15.0)
    reg.assign("room-1", now=1.0)
    reg.expire(now=100.0)

    results = reg.redispatch(now=100.0)

    assert results == [("room-1", Rejection.NO_WORKERS)]
    assert reg.snapshot(100.0)["orphaned"] == ["room-1"]


def test_clean_drain_releases_rooms_and_replaces_them_immediately():
    reg = fleet(4, 4)
    reg.assign("room-1", now=1.0)

    released = reg.deregister("w0")
    results = reg.redispatch(now=1.0)

    assert released == ["room-1"]
    assert [out.worker_id for _, out in results if isinstance(out, Assignment)] == ["w1"]


def test_capacity_accounting_survives_a_worker_under_reporting_itself():
    """Assignments the router knows about count even if the worker reports active=0."""
    reg = fleet(2)
    reg.assign("room-1", now=1.0)
    reg.assign("room-2", now=1.0)
    reg.heartbeat("w0", 2.0, active=0)  # stale or buggy self-report

    assert reg.assign("room-3", now=2.0) is Rejection.NO_CAPACITY


def test_snapshot_reports_the_fleet_view_the_dashboard_needs():
    reg = fleet(4, 4)
    reg.assign("room-1", now=1.0)
    snap = reg.snapshot(now=1.0)

    assert snap["live_workers"] == 2
    assert snap["fleet_capacity"] == 8
    assert snap["fleet_active"] == 1
    assert snap["fleet_free"] == 7
    assert snap["assignments"] == 1

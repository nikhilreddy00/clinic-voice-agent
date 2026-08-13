"""Phase 11 — session router HTTP surface.

``test_registry`` covers the decisions; this covers the wire: that a LiveKit webhook actually
turns into an assignment, that a full fleet answers with a status a monitor will notice, and
that a duplicate webhook does not start a second agent in the same room.

Skipped when FastAPI is not installed — the router is a separate service with its own
environment, and the agent's test venv has no reason to carry a web framework.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

fastapi = pytest.importorskip("fastapi", reason="session_router deps not installed")
from fastapi.testclient import TestClient  # noqa: E402

import app as router_app  # noqa: E402
from registry import WorkerRegistry  # noqa: E402


@pytest.fixture()
def client():
    # Fresh registry per test: the module holds process-global fleet state by design.
    router_app.registry = WorkerRegistry(heartbeat_timeout=router_app.HEARTBEAT_TIMEOUT)
    return TestClient(router_app.app)


def register(client, worker_id: str, capacity: int = 4):
    return client.post(
        "/workers/register", json={"worker_id": worker_id, "capacity": capacity}
    )


def room_started(client, room: str):
    return client.post(
        "/livekit/webhook", json={"event": "room_started", "room": {"name": room}}
    )


def test_a_new_room_is_assigned_to_a_registered_worker(client):
    register(client, "w0")
    resp = room_started(client, "clinic-call_abc")

    assert resp.status_code == 200
    assert resp.json() == {"ok": True, "room": "clinic-call_abc", "worker_id": "w0"}


def test_a_full_fleet_answers_503_so_it_shows_up_in_monitoring(client):
    """A rejected call is a real caller hearing nothing; it must not look like a success."""
    register(client, "w0", capacity=1)
    room_started(client, "clinic-call_1")

    resp = room_started(client, "clinic-call_2")

    assert resp.status_code == 503
    assert resp.json()["reason"] == "no_capacity"


def test_no_registered_workers_is_reported_distinctly_from_a_full_fleet(client):
    resp = room_started(client, "clinic-call_1")
    assert resp.status_code == 503
    assert resp.json()["reason"] == "no_workers"


def test_duplicate_webhook_does_not_start_a_second_agent(client):
    register(client, "w0")
    room_started(client, "clinic-call_1")

    resp = room_started(client, "clinic-call_1")

    assert resp.status_code == 503
    assert resp.json()["reason"] == "already_assigned"
    assert client.get("/status").json()["assignments"] == 1


def test_room_finished_releases_the_slot(client):
    register(client, "w0", capacity=1)
    room_started(client, "clinic-call_1")

    resp = client.post(
        "/livekit/webhook", json={"event": "room_finished", "room": {"name": "clinic-call_1"}}
    )

    assert resp.json()["released"] is True
    assert room_started(client, "clinic-call_2").status_code == 200


def test_unrelated_livekit_events_are_ignored_not_errors(client):
    resp = client.post(
        "/livekit/webhook",
        json={"event": "participant_joined", "room": {"name": "clinic-call_1"}},
    )
    assert resp.status_code == 200 and resp.json()["ignored"] == "participant_joined"


def test_webhook_without_a_room_name_is_rejected(client):
    assert client.post("/livekit/webhook", json={"event": "room_started"}).status_code == 400


def test_heartbeat_from_an_unknown_worker_asks_it_to_reregister(client):
    """Better than silently dropping a healthy worker after a router restart."""
    resp = client.post("/workers/heartbeat", json={"worker_id": "ghost", "active": 0})
    assert resp.status_code == 404
    assert resp.json()["action"] == "reregister"


def test_workers_poll_for_what_they_should_be_running(client):
    register(client, "w0")
    room_started(client, "clinic-call_1")

    resp = client.get("/assignments/w0")
    assert resp.json()["rooms"] == ["clinic-call_1"]


def test_draining_a_worker_moves_its_rooms_to_a_peer(client):
    register(client, "w0")
    register(client, "w1")
    room_started(client, "clinic-call_1")
    owner = client.get("/status").json()["workers"]
    holder = next(w["worker_id"] for w in owner if w["rooms"])

    resp = client.post("/workers/drain", json={"worker_id": holder})

    assert resp.json()["released_rooms"] == ["clinic-call_1"]
    assert resp.json()["redispatched"] == ["clinic-call_1"]


def test_status_exposes_the_fleet_view(client):
    register(client, "w0", capacity=4)
    register(client, "w1", capacity=4)
    room_started(client, "clinic-call_1")

    body = client.get("/status").json()
    assert body["live_workers"] == 2
    assert body["fleet_capacity"] == 8
    assert body["fleet_active"] == 1
    assert body["assignments"] == 1

"""Phase 11 — the session router service.

The control plane for room-per-call telephony. With Individual SIP dispatch, LiveKit creates a
new room per inbound call and fires a ``room_started`` webhook; this service decides which
worker takes it.

    PSTN ─▶ LiveKit SIP (Individual dispatch, room-per-call)
                  │  room_started / room_finished webhook
                  ▼
            Session Router  ──assign──▶  Worker (N sessions each)

Endpoints:

    POST /livekit/webhook   LiveKit room lifecycle. Assigns on room_started, releases on
                            room_finished.
    POST /workers/register  A worker announces itself with its capacity.
    POST /workers/heartbeat A worker reports load. Missing heartbeats are how death is detected.
    POST /workers/drain     Deregister cleanly, e.g. during a deploy.
    GET  /assignments/{id}  What a worker should be running (workers poll this).
    GET  /status            Fleet view, for the dashboard and for load-test assertions.

All decisions live in :mod:`registry`, which has no I/O and no clock of its own. This module is
transport only: parse, call the registry, serialize. That split is what makes worker death and
assignment races testable without sleeping in a test.

**Webhook authenticity is not verified yet.** LiveKit signs webhooks with an Authorization JWT;
this build parses the body without checking it, which is fine for a load-test control plane on
a private network and is NOT fine on the public internet — an unauthenticated caller could
spawn agent sessions. Verifying the token is a prerequisite for deploying this, and it is
called out in the Phase-11 notes rather than left to be discovered.
"""

from __future__ import annotations

import os
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from registry import Assignment, Rejection, WorkerRegistry

HEARTBEAT_TIMEOUT = float(os.getenv("CLINIC_ROUTER_HEARTBEAT_TIMEOUT", "15"))

app = FastAPI(title="Clinic Session Router", version="0.1.0")
registry = WorkerRegistry(heartbeat_timeout=HEARTBEAT_TIMEOUT)


def _now() -> float:
    return time.monotonic()


def _sweep(now: float) -> list[str]:
    """Drop dead workers and try to re-place whatever they were carrying."""
    orphaned = registry.expire(now)
    if orphaned:
        registry.redispatch(now)
    return orphaned


@app.post("/workers/register")
async def register_worker(request: Request) -> JSONResponse:
    body = await request.json()
    record = registry.register(
        body["worker_id"], int(body.get("capacity", 1)), _now(), url=body.get("url", "")
    )
    return JSONResponse({"ok": True, "worker": record.as_dict()})


@app.post("/workers/heartbeat")
async def worker_heartbeat(request: Request) -> JSONResponse:
    body = await request.json()
    now = _now()
    _sweep(now)
    record = registry.heartbeat(
        body["worker_id"],
        now,
        active=body.get("active"),
        capacity=body.get("capacity"),
        draining=body.get("draining"),
    )
    if record is None:
        # The worker outlived its registration (router restart, or it was expired while busy).
        # Telling it to re-register is better than silently dropping a healthy worker.
        return JSONResponse({"ok": False, "action": "reregister"}, status_code=404)
    return JSONResponse({"ok": True, "worker": record.as_dict()})


@app.post("/workers/drain")
async def drain_worker(request: Request) -> JSONResponse:
    body = await request.json()
    rooms = registry.deregister(body["worker_id"])
    results = registry.redispatch(_now())
    return JSONResponse(
        {
            "ok": True,
            "released_rooms": rooms,
            "redispatched": [r for r, out in results if isinstance(out, Assignment)],
        }
    )


@app.post("/livekit/webhook")
async def livekit_webhook(request: Request) -> JSONResponse:
    """Handle LiveKit room lifecycle events.

    Only ``room_started`` and ``room_finished`` matter here. With Individual dispatch each
    inbound call gets its own room, so room lifecycle *is* call lifecycle.
    """
    body = await request.json()
    event = body.get("event", "")
    room = (body.get("room") or {}).get("name", "")
    now = _now()
    _sweep(now)

    if not room:
        return JSONResponse({"ok": False, "error": "missing room name"}, status_code=400)

    if event == "room_started":
        outcome = registry.assign(room, now, call_id=room)
        if isinstance(outcome, Rejection):
            # A rejected call is a real caller hearing nothing. Surfaced with a distinct
            # status so it shows up in monitoring instead of looking like a success.
            return JSONResponse(
                {"ok": False, "room": room, "reason": outcome.value}, status_code=503
            )
        return JSONResponse(
            {"ok": True, "room": room, "worker_id": outcome.worker_id}
        )

    if event == "room_finished":
        released = registry.release(room)
        return JSONResponse({"ok": True, "room": room, "released": released is not None})

    return JSONResponse({"ok": True, "ignored": event})


@app.get("/assignments/{worker_id}")
async def assignments_for(worker_id: str) -> JSONResponse:
    now = _now()
    _sweep(now)
    snapshot = registry.snapshot(now)
    worker = next((w for w in snapshot["workers"] if w["worker_id"] == worker_id), None)
    if worker is None:
        return JSONResponse({"ok": False, "error": "unknown worker"}, status_code=404)
    return JSONResponse({"ok": True, "rooms": worker["rooms"]})


@app.get("/status")
async def status() -> JSONResponse:
    now = _now()
    _sweep(now)
    return JSONResponse(registry.snapshot(now))


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"ok": True})

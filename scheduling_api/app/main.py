"""Clinic scheduling API.

Endpoints:
  GET  /health             liveness
  GET  /availability       list open appointment slots (optional filters)
  POST /hold-slot          place a short-lived hold on a slot
  POST /confirm-booking    turn a valid hold into a confirmed booking
  GET  /metrics            aggregate the agent's call log for the dashboard

Storage is Postgres (see app/db.py and app/schema.sql). All data is synthetic — no real PHI.

Phase-9 changes visible here:
  * Handlers are `async def`. They were sync, so FastAPI ran each one in a threadpool worker and
    concurrency was capped by that pool while every request blocked a thread on I/O.
  * Writes are idempotent when the caller supplies an Idempotency-Key. A voice turn can time out
    *after* the booking commits; without this the agent's retry books a second appointment.
  * GET /availability performs no writes. Releasing expired holds and rolling the seeded window
    forward now run in a background sweeper.
  * Endpoints require a service token when CLINIC_API_TOKEN is set.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import secrets
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Response
from fastapi.staticfiles import StaticFiles

from . import db
from .metrics import aggregate_metrics
from .models import (
    AvailabilityResponse,
    ConfirmRequest,
    ConfirmResponse,
    HoldRequest,
    HoldResponse,
    SlotOut,
)
from .seed_data import CLINIC_NAME

logger = logging.getLogger("clinic.api")

# Service-to-service token. Unset = open (local dev and the existing test suite). Set in any
# deployed environment: these endpoints mutate appointment state and, once Phase 13 lands
# patient lookup, return PHI.
API_TOKEN = os.getenv("CLINIC_API_TOKEN")


async def require_token(authorization: str | None = Header(None)) -> None:
    if not API_TOKEN:
        return
    expected = f"Bearer {API_TOKEN}"
    # Constant-time compare: a plain == leaks the token prefix through timing.
    if not authorization or not secrets.compare_digest(authorization, expected):
        raise HTTPException(status_code=401, detail="Missing or invalid service token")


async def _sweeper() -> None:
    """Background maintenance, moved off the /availability read path.

    Releasing expired holds and re-stamping the seeded window are writes. Doing them inside every
    availability request meant reads took the write lock — the single worst thing you can do to a
    read path you intend to scale. Correctness does not depend on this loop: hold_slot's
    compare-and-swap reclaims expired holds on its own.
    """
    while True:
        try:
            await asyncio.sleep(db.SWEEP_INTERVAL_SECONDS)
            result = await db.sweep()
            if result["released_holds"]:
                logger.info("sweeper released %d expired hold(s)", result["released_holds"])
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — a sweep failure must not kill the loop
            logger.exception("sweeper pass failed; continuing")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.init_db()
    task = asyncio.create_task(_sweeper())
    try:
        yield
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        await db.close_pool()


app = FastAPI(
    title=f"{CLINIC_NAME} Scheduling API",
    version="0.2.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "clinic": CLINIC_NAME}


@app.get("/metrics")
async def metrics() -> dict:
    """Aggregate the agent's call log (logs/calls.jsonl) for the dashboard (Phase 6)."""
    return aggregate_metrics()


# Serve the observability dashboard same-origin with /metrics so the page needs no CORS and no
# configured API base — it just fetches "/metrics".
_DASHBOARD_DIR = Path(
    os.getenv("CLINIC_DASHBOARD_DIR", str(Path(__file__).resolve().parents[2] / "dashboard"))
)
if _DASHBOARD_DIR.is_dir():
    app.mount("/dashboard", StaticFiles(directory=_DASHBOARD_DIR, html=True), name="dashboard")


def _iso(value: datetime | str) -> str:
    """Serialise a timestamptz to ISO 8601. The API's wire format is strings."""
    return value.isoformat() if isinstance(value, datetime) else str(value)


@app.get("/availability", response_model=AvailabilityResponse,
         dependencies=[Depends(require_token)])
async def get_availability(
    provider_id: int | None = Query(None, description="Filter to one provider"),
    date: str | None = Query(None, description="Filter to a YYYY-MM-DD clinic-local day"),
    reason: str | None = Query(None, description="Filter by reason_category"),
) -> AvailabilityResponse:
    """Return currently-available future slots, optionally filtered. Read-only."""
    rows = await db.list_available_slots(provider_id=provider_id, date=date, reason=reason)
    return AvailabilityResponse(
        slots=[SlotOut(**{**r, "start_time": _iso(r["start_time"])}) for r in rows]
    )


@app.post("/hold-slot", response_model=HoldResponse, dependencies=[Depends(require_token)])
async def hold_slot(
    req: HoldRequest,
    response: Response,
    idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
) -> HoldResponse:
    """Place a short-lived hold on an available slot."""
    endpoint = "hold-slot"
    fingerprint = db.request_fingerprint(req.model_dump())

    if idempotency_key:
        replay = await _replay_or_claim(idempotency_key, endpoint, fingerprint)
        if replay is not None:
            response.headers["Idempotent-Replay"] = "true"
            return HoldResponse(**replay)

    try:
        row = await db.hold_slot(req.slot_id)
    except db.SlotNotFound as exc:
        await _release_claim(idempotency_key, endpoint)
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except db.SlotUnavailable as exc:
        await _release_claim(idempotency_key, endpoint)
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception:
        await _release_claim(idempotency_key, endpoint)
        raise

    payload = {
        "hold_id": str(row["hold_id"]),
        "slot_id": row["slot_id"],
        "expires_at": _iso(row["hold_expires_at"]),
    }
    if idempotency_key:
        await db.store_idempotent_response(idempotency_key, endpoint, payload)
    return HoldResponse(**payload)


@app.post("/confirm-booking", response_model=ConfirmResponse,
          dependencies=[Depends(require_token)])
async def confirm_booking(
    req: ConfirmRequest,
    response: Response,
    idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
) -> ConfirmResponse:
    """Convert a valid, unexpired hold into a confirmed booking.

    This is the endpoint idempotency matters most for: a retry after a timed-out voice turn must
    return the original confirmation number, not book a second appointment.
    """
    endpoint = "confirm-booking"
    fingerprint = db.request_fingerprint(req.model_dump())

    if idempotency_key:
        replay = await _replay_or_claim(idempotency_key, endpoint, fingerprint)
        if replay is not None:
            response.headers["Idempotent-Replay"] = "true"
            return ConfirmResponse(**replay)

    try:
        booking = await db.confirm_booking(
            hold_id=req.hold_id,
            patient_name=req.patient_name,
            reason=req.reason,
            date_of_birth=req.date_of_birth,
            new_patient=req.new_patient,
            symptom_notes=req.symptom_notes,
        )
    except db.HoldInvalid as exc:
        await _release_claim(idempotency_key, endpoint)
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception:
        await _release_claim(idempotency_key, endpoint)
        raise

    payload = {**booking, "start_time": _iso(booking["start_time"])}
    if idempotency_key:
        await db.store_idempotent_response(idempotency_key, endpoint, payload)
    return ConfirmResponse(**payload)


# =========================================================================================
# Idempotency helpers
# =========================================================================================


async def _replay_or_claim(key: str, endpoint: str, fingerprint: str) -> dict | None:
    """Return a stored response to replay, or None if this request now owns the key."""
    try:
        stored = await db.claim_idempotency_key(key, endpoint, fingerprint)
    except db.IdempotencyConflict as exc:
        # 422: the key is valid but was used for a different body — a client bug, not a conflict
        # over resource state.
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except db.IdempotencyInFlight as exc:
        # 409 + Retry-After: the original is still running. Retrying is correct; executing the
        # write a second time is not.
        raise HTTPException(
            status_code=409, detail=str(exc), headers={"Retry-After": "1"}
        ) from exc
    return stored["response"] if stored else None


async def _release_claim(key: str | None, endpoint: str) -> None:
    """Drop an unfinished claim after a failure so the same key can be retried."""
    if key:
        await db.release_idempotency_key(key, endpoint)

"""Clinic scheduling API.

Endpoints:
  GET  /health             liveness
  GET  /availability       list open appointment slots (optional filters)
  POST /hold-slot          place a short-lived hold on a slot
  POST /confirm-booking    turn a valid hold into a confirmed booking
  GET  /metrics            aggregate recent calls' metrics for the dashboard
  POST /call-metrics       the agent posts one call's metrics at teardown

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
    AppointmentOut,
    AppointmentsResponse,
    AvailabilityResponse,
    CallMetricsRequest,
    CallMetricsResponse,
    CallerMemoryResponse,
    CancelRequest,
    CancelResponse,
    ClinicInfoResponse,
    ClinicResponse,
    ConfirmRequest,
    ConfirmResponse,
    HoldRequest,
    HoldResponse,
    RescheduleRequest,
    RescheduleResponse,
    SlotOut,
    StaffTaskRequest,
    StaffTaskResponse,
    VerifiedRequest,
    VerifyResponse,
)
from .seed_data import CLINIC_NAME

logger = logging.getLogger("clinic.api")

# Service-to-service token. Unset = open (local dev and the existing test suite). Set in any
# deployed environment: these endpoints mutate appointment state and, once Phase 13 lands
# patient lookup, return PHI.
API_TOKEN = os.getenv("CLINIC_API_TOKEN")

# One message for every verification failure. See the Phase-13 section below.
_NOT_VERIFIED = "Identity could not be verified"


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
    logger.info("connected to %s", db.describe_target())
    # Say the PHI posture out loud on every boot. An operator who believes the database is
    # encrypted and is wrong should find that out here, not in an incident review.
    if db.phi_at_rest_enabled():
        logger.info("PHI at rest: dates of birth digested, clinical notes encrypted")
    else:
        logger.warning(
            "PHI at rest: DISABLED (CLINIC_PHI_KEY unset) — dates of birth and clinical notes "
            "are stored in the clear. Correct for synthetic data; set a key before real PHI."
        )
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


@app.middleware("http")
async def bind_request_context(request, call_next):
    """Bind the caller's call id and actor for the audit trail (Phase 17).

    Headers rather than a body field: every endpoint would otherwise need a new required field,
    and this is metadata about the request, not about the appointment. `X-Call-Id` is the
    agent's own call id, so an audit row joins straight to `logs/traces/<call_id>.jsonl` and to
    the OTel trace for the same call.

    `X-Clinic-Slug` is the tenant (Phase 17). Absent means the default clinic, which is what
    every pre-Phase-17 caller, script and test sends — so tenancy arrived without a flag day.
    It is NOT an authorization boundary on its own: today the service token is shared, so a
    holder of it can name any tenant. Per-tenant credentials are named in docs/compliance.md as
    an open item; the isolation this header selects is real, the authentication in front of it
    is not yet per-tenant.
    """
    db.set_request_context(
        call_id=request.headers.get("X-Call-Id"),
        actor=request.headers.get("X-Actor"),
        clinic_slug=request.headers.get("X-Clinic-Slug"),
    )
    return await call_next(request)


@app.get("/health")
async def health() -> dict:
    """Liveness — and the NAME of the database this process is writing to.

    The name is here for the same reason the full target is logged at startup: an agent booking
    into the wrong database looks completely healthy from every other angle. The name alone
    ("clinic_dev" vs "postgres") answers that, and this route is deliberately the only one with
    no service token — start_demo.sh publishes it through ngrok. The host, the username, and
    the Supabase project ref stay in the startup log, where they are not world-readable.
    """
    return {"status": "ok", "clinic": CLINIC_NAME, "database": db.target_name()}


@app.get("/metrics")
async def metrics() -> dict:
    """Aggregate recent calls' metrics for the dashboard.

    Phase 16: the source is Postgres, not `logs/calls.jsonl`. The old reader re-read the whole
    file per request and, on Railway, read a file the agent's container had never written to.
    Bounded to the most recent `CLINIC_METRICS_WINDOW` calls.
    """
    return aggregate_metrics(await db.fetch_call_events())


@app.post("/call-metrics", response_model=CallMetricsResponse,
          dependencies=[Depends(require_token)])
async def post_call_metrics(req: CallMetricsRequest) -> CallMetricsResponse:
    """One call's operational metrics, posted by the agent at teardown.

    Idempotent on call_id: a re-post replaces the call's rows. The request model forbids extra
    fields, so PHI cannot be attached to an operational record even by accident.
    """
    result = await db.record_call_metrics(req.model_dump(mode="json"))
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("error", "bad request"))
    return CallMetricsResponse(**result)


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
            phone=req.phone,
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
# Phase 13 — caller memory, verification, and the expanded tool surface
# =========================================================================================
#
# Every PHI endpoint here re-verifies (phone, date_of_birth) inside the database call, and a
# failure is a 403 with a fixed message. It is deliberately the SAME message whether the number
# is unknown, the DOB is wrong, or the appointment belongs to someone else: an attacker probing
# with a spoofed caller ID must not be able to tell "not a patient" from "wrong birthday".


@app.get("/caller-memory", response_model=CallerMemoryResponse,
         dependencies=[Depends(require_token)])
async def get_caller_memory(phone: str = Query(..., description="Caller ANI, E.164")):
    """Pre-greeting lookup. Returns no identity — recognising a number is not authentication."""
    return CallerMemoryResponse(**await db.caller_memory(phone))


@app.get("/clinic", response_model=ClinicResponse, dependencies=[Depends(require_token)])
async def get_clinic(did: str = Query(..., description="The number the caller dialed, E.164")):
    """Resolve a DIALED number to its tenant — the agent's entry point into multi-tenancy.

    One worker process answers for several clinics, so the first thing a call needs is which
    clinic it is. The answer is not PHI: a name, a timezone, and the number this clinic's calls
    escalate to.
    """
    row = await db.clinic_by_did(did)
    if row is None:
        raise HTTPException(status_code=404, detail="No clinic is configured for that number")
    return ClinicResponse(**row)


@app.get("/clinic-info", response_model=ClinicInfoResponse,
         dependencies=[Depends(require_token)])
async def get_clinic_info(topic: str | None = Query(None)):
    """Curated facts (hours, location, parking, providers, prep). Open — not PHI."""
    return ClinicInfoResponse(**await db.clinic_info(topic))


@app.post("/verify-identity", response_model=VerifyResponse,
          dependencies=[Depends(require_token)])
async def verify_identity(req: VerifiedRequest) -> VerifyResponse:
    try:
        return VerifyResponse(**await db.verify_identity(
            phone=req.phone, date_of_birth=req.date_of_birth, name=req.name
        ))
    except db.NotVerified as exc:
        raise HTTPException(status_code=403, detail=_NOT_VERIFIED) from exc


@app.post("/appointments", response_model=AppointmentsResponse,
          dependencies=[Depends(require_token)])
async def list_appointments(req: VerifiedRequest) -> AppointmentsResponse:
    """POST, not GET: the request body carries a date of birth, and a DOB does not belong in a
    URL where it lands in access logs, proxy logs, and browser history."""
    try:
        rows = await db.list_appointments(
            phone=req.phone, date_of_birth=req.date_of_birth, name=req.name
        )
    except db.NotVerified as exc:
        raise HTTPException(status_code=403, detail=_NOT_VERIFIED) from exc
    return AppointmentsResponse(
        appointments=[
            AppointmentOut(**{**r, "start_time": _iso(r["start_time"])}) for r in rows
        ]
    )


@app.post("/reschedule", response_model=RescheduleResponse,
          dependencies=[Depends(require_token)])
async def reschedule(req: RescheduleRequest) -> RescheduleResponse:
    try:
        row = await db.reschedule_appointment(
            confirmation_id=req.confirmation_id,
            new_slot_id=req.new_slot_id,
            phone=req.phone,
            date_of_birth=req.date_of_birth,
            name=req.name,
        )
    except db.NotVerified as exc:
        raise HTTPException(status_code=403, detail=_NOT_VERIFIED) from exc
    except db.BookingNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except db.SlotUnavailable as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return RescheduleResponse(**{**row, "start_time": _iso(row["start_time"])})


@app.post("/cancel", response_model=CancelResponse, dependencies=[Depends(require_token)])
async def cancel(req: CancelRequest) -> CancelResponse:
    try:
        row = await db.cancel_appointment(
            confirmation_id=req.confirmation_id,
            phone=req.phone,
            date_of_birth=req.date_of_birth,
            reason=req.reason,
            name=req.name,
        )
    except db.NotVerified as exc:
        raise HTTPException(status_code=403, detail=_NOT_VERIFIED) from exc
    except db.BookingNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return CancelResponse(**row)


@app.post("/staff-tasks", response_model=StaffTaskResponse,
          dependencies=[Depends(require_token)])
async def create_staff_task(req: StaffTaskRequest) -> StaffTaskResponse:
    """Queue work for a human — refills above all. Never an approval."""
    try:
        row = await db.create_staff_task(
            kind=req.kind, phone=req.phone, date_of_birth=req.date_of_birth,
            payload=req.payload, name=req.name,
        )
    except db.NotVerified as exc:
        raise HTTPException(status_code=403, detail=_NOT_VERIFIED) from exc
    return StaffTaskResponse(**row)


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

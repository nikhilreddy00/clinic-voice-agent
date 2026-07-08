"""Mock clinic scheduling API.

Endpoints:
  GET  /availability       list open appointment slots (optional filters)
  POST /hold-slot          place a short-lived hold on a slot
  POST /confirm-booking    turn a valid hold into a confirmed booking

Storage is SQLite (see app/db.py). All data is synthetic — no real PHI.
"""

from __future__ import annotations

import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    yield


app = FastAPI(
    title=f"{CLINIC_NAME} Scheduling API (mock)",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "clinic": CLINIC_NAME}


@app.get("/metrics")
def metrics() -> dict:
    """Aggregate the agent's call log (logs/calls.jsonl) for the dashboard (Phase 6).

    Latency P50/P95/P99 per stage, ASR confidence, tool-call outcomes, call outcomes, and the
    last five calls. Returns a zeroed-but-valid payload when no calls have been logged yet.
    """
    return aggregate_metrics()


# Serve the observability dashboard same-origin with /metrics so the page needs no CORS and no
# configured API base — it just fetches "/metrics". dashboard/ lives at the repo root in local
# dev; CLINIC_DASHBOARD_DIR overrides the location inside the container image.
_DASHBOARD_DIR = Path(
    os.getenv("CLINIC_DASHBOARD_DIR", str(Path(__file__).resolve().parents[2] / "dashboard"))
)
if _DASHBOARD_DIR.is_dir():
    app.mount("/dashboard", StaticFiles(directory=_DASHBOARD_DIR, html=True), name="dashboard")


@app.get("/availability", response_model=AvailabilityResponse)
def get_availability(
    provider_id: int | None = Query(None, description="Filter to one provider"),
    date: str | None = Query(None, description="Filter to a YYYY-MM-DD day (UTC)"),
    reason: str | None = Query(None, description="Filter by reason_category"),
) -> AvailabilityResponse:
    """Return currently-available slots, optionally filtered."""
    conn = db.get_connection()
    try:
        db.release_expired_holds(conn)
        # Roll seeded slots forward so an always-on server never runs out of upcoming slots as
        # the originally-seeded window ages into the past (see db.refresh_available_slots).
        db.refresh_available_slots(conn)
        conn.commit()

        # Only future slots: never offer a time that has already passed (Phase-4 fix #2).
        # timezone: US/Eastern (America/New_York) — agent operates in clinic local time. "Now"
        # is taken in clinic-local time (so "already passed" means passed in the clinic's day,
        # not in UTC), then normalized back to UTC for the comparison: start_time is stored as
        # UTC (+00:00), and this lexical string comparison is only valid when BOTH operands
        # share that same offset. An instant compared in either tz is the same instant; taking
        # it in CLINIC_TZ first is what keeps the reasoning about "today"/"passed" in local time.
        sql = """
            SELECT s.id AS slot_id, s.provider_id, p.name AS provider_name,
                   p.specialty, s.start_time, s.reason_category
              FROM slots s
              JOIN providers p ON p.id = s.provider_id
             WHERE s.status = 'available'
               AND s.start_time >= ?
        """
        now_clinic = datetime.now(db.CLINIC_TZ)
        params: list = [now_clinic.astimezone(timezone.utc).isoformat()]
        if provider_id is not None:
            sql += " AND s.provider_id = ?"
            params.append(provider_id)
        if reason is not None:
            sql += " AND s.reason_category = ?"
            params.append(reason)
        if date is not None:
            sql += " AND substr(s.start_time, 1, 10) = ?"
            params.append(date)
        sql += " ORDER BY s.start_time, s.provider_id"

        rows = conn.execute(sql, params).fetchall()
        return AvailabilityResponse(slots=[SlotOut(**dict(r)) for r in rows])
    finally:
        conn.close()


@app.post("/hold-slot", response_model=HoldResponse)
def hold_slot(req: HoldRequest) -> HoldResponse:
    """Place a short-lived hold on an available slot."""
    conn = db.get_connection()
    try:
        db.release_expired_holds(conn)

        row = conn.execute("SELECT status FROM slots WHERE id = ?", (req.slot_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail=f"Slot {req.slot_id} not found")
        if row["status"] != "available":
            raise HTTPException(
                status_code=409,
                detail=f"Slot {req.slot_id} is not available (status: {row['status']})",
            )

        hold_id = uuid.uuid4().hex
        expires_at = db._now() + timedelta(seconds=db.HOLD_TTL_SECONDS)
        conn.execute(
            """
            UPDATE slots
               SET status = 'held', hold_id = ?, hold_expires_at = ?
             WHERE id = ?
            """,
            (hold_id, expires_at.isoformat(), req.slot_id),
        )
        conn.commit()
        return HoldResponse(
            hold_id=hold_id, slot_id=req.slot_id, expires_at=expires_at.isoformat()
        )
    finally:
        conn.close()


@app.post("/confirm-booking", response_model=ConfirmResponse)
def confirm_booking(req: ConfirmRequest) -> ConfirmResponse:
    """Convert a valid, unexpired hold into a confirmed booking."""
    conn = db.get_connection()
    try:
        db.release_expired_holds(conn)
        conn.commit()

        row = conn.execute(
            """
            SELECT s.id AS slot_id, s.status, s.start_time, p.name AS provider_name
              FROM slots s
              JOIN providers p ON p.id = s.provider_id
             WHERE s.hold_id = ?
            """,
            (req.hold_id,),
        ).fetchone()

        if row is None or row["status"] != "held":
            # Either the hold_id was never valid, or the hold expired and was released.
            raise HTTPException(
                status_code=409,
                detail="Hold is invalid or has expired. Please select a slot again.",
            )

        confirmation_id = uuid.uuid4().hex[:8].upper()
        conn.execute(
            "UPDATE slots SET status = 'booked', hold_id = NULL, hold_expires_at = NULL WHERE id = ?",
            (row["slot_id"],),
        )
        conn.execute(
            """
            INSERT INTO bookings (confirmation_id, slot_id, patient_name, reason, created_at,
                                  date_of_birth, new_patient, symptom_notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                confirmation_id,
                row["slot_id"],
                req.patient_name,
                req.reason,
                db._now().isoformat(),
                req.date_of_birth,
                req.new_patient,
                req.symptom_notes,
            ),
        )
        conn.commit()
        return ConfirmResponse(
            confirmation_id=confirmation_id,
            slot_id=row["slot_id"],
            provider_name=row["provider_name"],
            start_time=row["start_time"],
            patient_name=req.patient_name,
            date_of_birth=req.date_of_birth,
            new_patient=req.new_patient,
            symptom_notes=req.symptom_notes,
        )
    finally:
        conn.close()

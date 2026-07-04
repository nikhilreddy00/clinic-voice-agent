"""Mock clinic scheduling API.

Endpoints:
  GET  /availability       list open appointment slots (optional filters)
  POST /hold-slot          place a short-lived hold on a slot
  POST /confirm-booking    turn a valid hold into a confirmed booking

Storage is SQLite (see app/db.py). All data is synthetic — no real PHI.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from datetime import timedelta

from fastapi import FastAPI, HTTPException, Query

from . import db
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
        conn.commit()

        sql = """
            SELECT s.id AS slot_id, s.provider_id, p.name AS provider_name,
                   p.specialty, s.start_time, s.reason_category
              FROM slots s
              JOIN providers p ON p.id = s.provider_id
             WHERE s.status = 'available'
        """
        params: list = []
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
            INSERT INTO bookings (confirmation_id, slot_id, patient_name, reason, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                confirmation_id,
                row["slot_id"],
                req.patient_name,
                req.reason,
                db._now().isoformat(),
            ),
        )
        conn.commit()
        return ConfirmResponse(
            confirmation_id=confirmation_id,
            slot_id=row["slot_id"],
            provider_name=row["provider_name"],
            start_time=row["start_time"],
            patient_name=req.patient_name,
        )
    finally:
        conn.close()

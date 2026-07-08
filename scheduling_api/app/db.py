"""SQLite storage for the mock scheduling API.

Tables:
  providers(id, name, specialty)
  slots(id, provider_id, start_time, reason_category, status, hold_id, hold_expires_at)
        status in ('available', 'held', 'booked')
  bookings(confirmation_id, slot_id, patient_name, reason, created_at,
           date_of_birth, new_patient, symptom_notes)

Holds are short-lived: an expired hold is lazily released (slot returns to 'available')
whenever we read or mutate slot state.
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from . import seed_data

# timezone: US/Eastern (America/New_York) — agent operates in clinic local time.
# Slots' start_time and hold_expires_at are STORED as UTC (+00:00) so lexical string
# comparisons stay valid year-round (no DST offset flip mid-column); clinic-local time is
# used only when we need to know the current wall-clock day/time in the clinic.
CLINIC_TZ = ZoneInfo("America/New_York")

# How long a hold lasts before it expires and the slot frees up again.
HOLD_TTL_SECONDS = 120

# DB path is configurable so tests can point at a throwaway file.
DB_PATH = os.getenv("CLINIC_DB_PATH", os.path.join(os.path.dirname(__file__), "..", "clinic.db"))


def _now() -> datetime:
    return datetime.now(timezone.utc)


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    """Create tables if they don't exist, then seed synthetic data (idempotent)."""
    conn = get_connection()
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS providers (
                id        INTEGER PRIMARY KEY,
                name      TEXT NOT NULL,
                specialty TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS slots (
                id              INTEGER PRIMARY KEY,
                provider_id     INTEGER NOT NULL REFERENCES providers(id),
                start_time      TEXT NOT NULL,
                reason_category TEXT NOT NULL,
                status          TEXT NOT NULL DEFAULT 'available',
                hold_id         TEXT,
                hold_expires_at TEXT
            );
            CREATE TABLE IF NOT EXISTS bookings (
                confirmation_id TEXT PRIMARY KEY,
                slot_id         INTEGER NOT NULL REFERENCES slots(id),
                patient_name    TEXT NOT NULL,
                reason          TEXT NOT NULL,
                created_at      TEXT NOT NULL,
                date_of_birth   TEXT,
                new_patient     INTEGER,
                symptom_notes   TEXT
            );
            """
        )
        _migrate_bookings_intake_columns(conn)
        _seed(conn)
        refresh_available_slots(conn)
        conn.commit()
    finally:
        conn.close()


# Intake columns added after the bookings table already shipped. CREATE TABLE IF NOT EXISTS
# won't alter an existing table, so add any missing column in place. Idempotent: only ALTERs
# columns that aren't already present. Synthetic data only.
_BOOKINGS_INTAKE_COLUMNS = (
    ("date_of_birth", "TEXT"),
    ("new_patient", "INTEGER"),
    ("symptom_notes", "TEXT"),
)


def _migrate_bookings_intake_columns(conn: sqlite3.Connection) -> None:
    existing = {row["name"] for row in conn.execute("PRAGMA table_info(bookings)")}
    for name, sql_type in _BOOKINGS_INTAKE_COLUMNS:
        if name not in existing:
            conn.execute(f"ALTER TABLE bookings ADD COLUMN {name} {sql_type}")


def _seed(conn: sqlite3.Connection) -> None:
    """Insert synthetic providers and slots. Idempotent via fixed primary keys."""
    conn.executemany(
        "INSERT OR IGNORE INTO providers (id, name, specialty) VALUES (?, ?, ?)",
        seed_data.PROVIDERS,
    )
    for slot_id, provider_id, start_iso, reason in seed_data.generate_slots():
        conn.execute(
            """
            INSERT OR IGNORE INTO slots (id, provider_id, start_time, reason_category, status)
            VALUES (?, ?, ?, ?, 'available')
            """,
            (slot_id, provider_id, start_iso, reason),
        )


def refresh_available_slots(conn: sqlite3.Connection) -> None:
    """Roll still-'available' slots forward to the upcoming next-few-working-days window.

    Seeded slots carry fixed ids but dates computed relative to seed time (see
    seed_data.generate_slots), and seeding is INSERT-OR-IGNORE by id — so nothing ever moves a
    slot's date after it's first written. On an always-on deploy (or a persisted DB) the seeded
    window therefore ages into the past within a few days and /availability goes empty. This
    re-stamps every slot that is still 'available' with the freshly-computed window, so a
    long-running server keeps offering upcoming slots without a restart.

    Only 'available' slots are touched: 'held' (an in-flight caller) and 'booked' (a confirmed
    appointment, referenced by a bookings row) keep their original date. Idempotent — on a
    freshly-seeded DB the dates already match, so this is a no-op.
    """
    for slot_id, _provider_id, start_iso, reason in seed_data.generate_slots():
        conn.execute(
            """
            UPDATE slots
               SET start_time = ?, reason_category = ?
             WHERE id = ? AND status = 'available'
            """,
            (start_iso, reason, slot_id),
        )


def release_expired_holds(conn: sqlite3.Connection) -> None:
    """Free any slot whose hold has expired (status 'held' -> 'available')."""
    conn.execute(
        """
        UPDATE slots
           SET status = 'available', hold_id = NULL, hold_expires_at = NULL
         WHERE status = 'held' AND hold_expires_at IS NOT NULL AND hold_expires_at < ?
        """,
        (_now().isoformat(),),
    )

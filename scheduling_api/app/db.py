"""Async Postgres storage for the clinic scheduling service (Phase 9).

Replaces the Phase-0 SQLite layer. The schema and the reasoning behind the migration are in
app/schema.sql; this module holds the connection pool and the queries.

THE CENTRAL IDEA: EVERY STATE TRANSITION IS A COMPARE-AND-SWAP
--------------------------------------------------------------
The SQLite version read a slot's status, decided in Python whether the transition was legal, then
wrote it -- three steps with nothing holding the row still in between. Two concurrent callers
could both read 'available' and both write 'held', and the second silently overwrote the first's
hold_id. The caller whose hold was clobbered then got a confusing 409 at confirm time, or worse,
both callers confirmed and the clinic double-booked.

Every mutation here is instead a single `UPDATE ... WHERE <expected state> RETURNING ...`. The
WHERE clause *is* the precondition, evaluated and applied atomically by Postgres under row-level
locking. Either the row matched and you get a row back, or it didn't and you get nothing -- there
is no window between the check and the write for anyone to slip into. No explicit transaction, no
SELECT ... FOR UPDATE, no advisory locks.

Expiry folds into the same statement. `hold_slot` accepts a slot that is 'available' *or* whose
hold has already expired, so reclaiming an abandoned hold is part of the same atomic swap rather
than a separate release-then-acquire that could interleave.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from . import seed_data

# The clinic's wall clock. Slot times are stored as timestamptz (absolute instants), so this is
# used for *reasoning* about the clinic's day -- "has this time passed today?" -- not storage.
CLINIC_TZ = ZoneInfo("America/New_York")

# How long a hold lasts before the slot frees up again.
HOLD_TTL_SECONDS = int(os.getenv("CLINIC_HOLD_TTL_SECONDS", "120"))

# How often the background sweeper runs (see main.py lifespan).
SWEEP_INTERVAL_SECONDS = int(os.getenv("CLINIC_SWEEP_INTERVAL_SECONDS", "30"))

# Single-tenant today; every table carries clinic_id so Phase 17 multi-tenancy is a routing
# change rather than a migration.
DEFAULT_CLINIC_SLUG = "grove-family"

DATABASE_URL = os.getenv(
    "CLINIC_DATABASE_URL", "postgresql://postgres@127.0.0.1:5432/clinic_dev"
)

# Supavisor's transaction-pooler port. Supabase exposes three endpoints and the choice matters:
#
#   direct    db.<ref>.supabase.co:5432            IPv6 only (IPv4 is a paid add-on)
#   session   aws-<region>.pooler.supabase.com:5432 IPv4 on every tier  <-- use this one
#   transaction aws-<region>.pooler.supabase.com:6543 IPv4, serverless-oriented
#
# The session pooler is the right endpoint for this service: a long-lived process holding its own
# connection pool, on a network that is very often IPv4-only (home ISPs, most CI runners). A
# direct connection simply fails to resolve there, which looks like a hang rather than a
# configuration error.
#
# The transaction pooler works too, but DOES NOT SUPPORT PREPARED STATEMENTS -- and psycopg3
# silently starts preparing a statement once it has been executed `prepare_threshold` times
# (default 5). So the app would work fine, then start failing on the sixth execution of a hot
# query. That is a genuinely nasty delayed failure, so it is detected and disabled here rather
# than left as a deployment note nobody reads.
_TRANSACTION_POOLER_PORT = "6543"


def _prepare_threshold() -> int | None:
    """psycopg3's prepared-statement threshold; None disables preparation entirely.

    Auto-disabled on the transaction pooler (see above). Override with
    CLINIC_DB_PREPARE_THRESHOLD ("none" to disable, an integer to set it).
    """
    override = os.getenv("CLINIC_DB_PREPARE_THRESHOLD")
    if override is not None:
        return None if override.strip().lower() in {"none", "off", ""} else int(override)
    if f":{_TRANSACTION_POOLER_PORT}/" in DATABASE_URL or DATABASE_URL.endswith(
        f":{_TRANSACTION_POOLER_PORT}"
    ):
        return None
    return 5  # psycopg3's default

_SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"

_pool: AsyncConnectionPool | None = None


def _now() -> datetime:
    return datetime.now(timezone.utc)


# =========================================================================================
# Pool lifecycle
# =========================================================================================


async def open_pool() -> AsyncConnectionPool:
    """Open the shared connection pool. Idempotent."""
    global _pool
    if _pool is None:
        _pool = AsyncConnectionPool(
            DATABASE_URL,
            min_size=1,
            # Keep this well under the database's connection limit. Supabase's free tier allows
            # far fewer direct connections than a self-hosted server, and the pool is per
            # process — N API replicas multiply it.
            max_size=int(os.getenv("CLINIC_DB_POOL_MAX", "10")),
            open=False,
            kwargs={
                "row_factory": dict_row,
                "prepare_threshold": _prepare_threshold(),
            },
        )
        await _pool.open(wait=True, timeout=15)
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


def pool() -> AsyncConnectionPool:
    if _pool is None:
        raise RuntimeError("connection pool is not open — call open_pool() first")
    return _pool


# =========================================================================================
# Schema + seed
# =========================================================================================


async def init_db() -> None:
    """Apply the schema and seed synthetic data. Idempotent — safe on every boot."""
    p = await open_pool()
    async with p.connection() as conn:
        await conn.execute(_SCHEMA_PATH.read_text())
        await _seed(conn)


async def _clinic_id(conn: AsyncConnection, slug: str = DEFAULT_CLINIC_SLUG) -> int:
    row = await (await conn.execute(
        "SELECT id FROM clinics WHERE slug = %s", (slug,)
    )).fetchone()
    if row is None:
        raise RuntimeError(f"clinic {slug!r} is not seeded")
    return row["id"]


async def _seed(conn: AsyncConnection) -> None:
    """Insert the synthetic clinic, providers, and slots. Idempotent via fixed keys."""
    await conn.execute(
        """
        INSERT INTO clinics (slug, name, timezone) VALUES (%s, %s, %s)
        ON CONFLICT (slug) DO NOTHING
        """,
        (DEFAULT_CLINIC_SLUG, seed_data.CLINIC_NAME, str(CLINIC_TZ)),
    )
    clinic_id = await _clinic_id(conn)

    for provider_id, name, specialty in seed_data.PROVIDERS:
        await conn.execute(
            """
            INSERT INTO providers (id, clinic_id, name, specialty) VALUES (%s, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING
            """,
            (provider_id, clinic_id, name, specialty),
        )

    for slot_id, provider_id, start, reason in seed_data.generate_slots():
        await conn.execute(
            """
            INSERT INTO slots (id, clinic_id, provider_id, start_time, reason_category, status)
            VALUES (%s, %s, %s, %s, %s, 'available')
            ON CONFLICT (id) DO NOTHING
            """,
            (slot_id, clinic_id, provider_id, start, reason),
        )

    await refresh_available_slots(conn)


# =========================================================================================
# Background maintenance (moved OFF the read path)
# =========================================================================================


"""Every table, children before parents, so a TRUNCATE ... CASCADE is ordered correctly.

Defined here rather than in a test fixture so the API tests and the eval harness share one
authoritative list. A table added to schema.sql and forgotten here would silently survive a
reset and leak rows between test cases.
"""
TABLES_IN_DEPENDENCY_ORDER = (
    "audit_log",
    "staff_tasks",
    "clinic_facts",
    "call_summaries",
    "caller_memory",
    "idempotency_keys",
    "bookings",
    "slots",
    "patients",
    "providers",
    "clinics",
)


async def truncate_all(conn: AsyncConnection) -> None:
    """Wipe every table. For tests and the eval harness — never called in normal operation."""
    existing = {
        r["tablename"]
        for r in await (await conn.execute(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
        )).fetchall()
    }
    targets = [t for t in TABLES_IN_DEPENDENCY_ORDER if t in existing]
    if targets:
        await conn.execute(f"TRUNCATE {', '.join(targets)} RESTART IDENTITY CASCADE")


async def reset_and_seed(conn: AsyncConnection) -> None:
    """Truncate then re-seed — one case's bookings must never bleed into the next."""
    await truncate_all(conn)
    await _seed(conn)


async def release_expired_holds(conn: AsyncConnection) -> int:
    """Free slots whose hold has expired. Returns how many were released.

    This used to run inside GET /availability, which meant every read took the write lock. It is
    now a background sweep. Correctness does not depend on it: hold_slot's compare-and-swap
    reclaims an expired hold on its own, so the sweeper is housekeeping (keeping /availability
    accurate between calls), not a safety mechanism.
    """
    cur = await conn.execute(
        """
        UPDATE slots
           SET status = 'available', hold_id = NULL, hold_expires_at = NULL
         WHERE status = 'held' AND hold_expires_at < now()
        """
    )
    return cur.rowcount


async def refresh_available_slots(conn: AsyncConnection) -> None:
    """Roll still-available seeded slots forward into the upcoming window.

    Seeded slots have fixed ids but dates computed at seed time, and seeding is
    INSERT ... ON CONFLICT DO NOTHING, so nothing moves a slot's date once written. On an
    always-on deploy the seeded window ages into the past within days and /availability empties
    out. Re-stamping the still-'available' slots keeps a long-running server offering appointments
    without a restart. Held and booked slots keep their original time.
    """
    clinic_id = await _clinic_id(conn)
    for slot_id, _provider_id, start, reason in seed_data.generate_slots():
        await conn.execute(
            """
            UPDATE slots
               SET start_time = %s, reason_category = %s
             WHERE id = %s AND clinic_id = %s AND status = 'available'
            """,
            (start, reason, slot_id, clinic_id),
        )


async def sweep() -> dict[str, int]:
    """One maintenance pass, called on an interval by the sweeper task in main.py."""
    async with pool().connection() as conn:
        released = await release_expired_holds(conn)
        await refresh_available_slots(conn)
    return {"released_holds": released}


# =========================================================================================
# Reads
# =========================================================================================


async def list_available_slots(
    *,
    provider_id: int | None = None,
    date: str | None = None,
    reason: str | None = None,
) -> list[dict[str, Any]]:
    """Available, still-in-the-future slots. Pure read — no writes on this path.

    "In the future" is evaluated against now() in the clinic's timezone. Because start_time is
    timestamptz, this is a real instant comparison rather than the old lexical string compare,
    which is what made the previous version quietly sensitive to stored-offset drift.
    """
    sql = """
        SELECT s.id AS slot_id, s.provider_id, p.name AS provider_name,
               p.specialty, s.start_time, s.reason_category
          FROM slots s
          JOIN providers p ON p.id = s.provider_id
         WHERE s.status = 'available'
           AND s.start_time >= now()
    """
    params: list[Any] = []
    if provider_id is not None:
        sql += " AND s.provider_id = %s"
        params.append(provider_id)
    if reason is not None:
        sql += " AND s.reason_category = %s"
        params.append(reason)
    if date is not None:
        # Compare the clinic-local calendar day: a caller asking for "Tuesday" means Tuesday in
        # the clinic's zone, not in UTC. Late-evening clinic slots fall on the next UTC day, so
        # casting in UTC would silently drop them from the requested date.
        sql += " AND (s.start_time AT TIME ZONE %s)::date = %s::date"
        params.extend([str(CLINIC_TZ), date])
    sql += " ORDER BY s.start_time, s.provider_id"

    async with pool().connection() as conn:
        rows = await (await conn.execute(sql, params)).fetchall()
    return rows


# =========================================================================================
# Writes — each one a single-statement compare-and-swap
# =========================================================================================


class SlotNotFound(Exception):
    pass


class SlotUnavailable(Exception):
    pass


class HoldInvalid(Exception):
    pass


async def hold_slot(slot_id: int) -> dict[str, Any]:
    """Place a short-lived hold. Atomic: no read-then-write window.

    The WHERE clause carries the whole precondition. A slot qualifies if it is available, or if
    it is held but that hold has already expired — folding reclaim into the same statement, so
    two callers racing for an abandoned slot cannot both win.

    Raises SlotUnavailable if the swap matched nothing, distinguishing "no such slot" (404) from
    "someone else has it" (409) with a follow-up read. That read is only for the error message;
    it cannot affect the outcome, which is already decided.
    """
    hold_id = uuid.uuid4()
    expires_at = _now() + timedelta(seconds=HOLD_TTL_SECONDS)

    async with pool().connection() as conn:
        row = await (await conn.execute(
            """
            UPDATE slots
               SET status = 'held', hold_id = %s, hold_expires_at = %s
             WHERE id = %s
               AND (status = 'available'
                    OR (status = 'held' AND hold_expires_at < now()))
            RETURNING id AS slot_id, hold_id, hold_expires_at
            """,
            (hold_id, expires_at, slot_id),
        )).fetchone()

        if row is not None:
            return row

        # The swap failed. Read once more only to produce the right error code.
        existing = await (await conn.execute(
            "SELECT status FROM slots WHERE id = %s", (slot_id,)
        )).fetchone()

    if existing is None:
        raise SlotNotFound(f"Slot {slot_id} not found")
    raise SlotUnavailable(f"Slot {slot_id} is not available (status: {existing['status']})")


async def confirm_booking(
    *,
    hold_id: str,
    patient_name: str,
    reason: str,
    date_of_birth: str | None = None,
    new_patient: bool | None = None,
    symptom_notes: str | None = None,
) -> dict[str, Any]:
    """Turn a valid, unexpired hold into a booking. Atomic across both writes.

    Two writes have to agree — the slot flips to 'booked' and a bookings row appears — so unlike
    hold_slot this needs a transaction around the compare-and-swap and the insert. The CAS still
    does the deciding: `WHERE hold_id = ... AND status = 'held' AND hold_expires_at > now()`
    matches at most one row, and the transaction guarantees the booking row lands with it or not
    at all.

    `bookings.slot_id` is additionally UNIQUE, so even if this logic were later broken the
    database refuses the second booking rather than accepting it.
    """
    try:
        parsed_hold = uuid.UUID(str(hold_id))
    except (ValueError, AttributeError, TypeError):
        # A malformed hold_id can never match a real hold; treat it as an expired/unknown hold
        # rather than letting a cast error surface as a 500.
        raise HoldInvalid("Hold is invalid or has expired. Please select a slot again.") from None

    confirmation_id = uuid.uuid4().hex[:8].upper()

    async with pool().connection() as conn:
        async with conn.transaction():
            slot = await (await conn.execute(
                """
                UPDATE slots
                   SET status = 'booked', hold_id = NULL, hold_expires_at = NULL
                 WHERE hold_id = %s
                   AND status = 'held'
                   AND hold_expires_at > now()
                RETURNING id AS slot_id, clinic_id, start_time, provider_id
                """,
                (parsed_hold,),
            )).fetchone()

            if slot is None:
                raise HoldInvalid("Hold is invalid or has expired. Please select a slot again.")

            provider = await (await conn.execute(
                "SELECT name FROM providers WHERE id = %s", (slot["provider_id"],)
            )).fetchone()

            await conn.execute(
                """
                INSERT INTO bookings (confirmation_id, clinic_id, slot_id, patient_name, reason,
                                      date_of_birth, new_patient, symptom_notes)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    confirmation_id,
                    slot["clinic_id"],
                    slot["slot_id"],
                    patient_name,
                    reason,
                    date_of_birth,
                    new_patient,
                    symptom_notes,
                ),
            )

    return {
        "confirmation_id": confirmation_id,
        "slot_id": slot["slot_id"],
        "provider_name": provider["name"] if provider else "",
        "start_time": slot["start_time"],
        "patient_name": patient_name,
        "date_of_birth": date_of_birth,
        "new_patient": new_patient,
        "symptom_notes": symptom_notes,
    }


# =========================================================================================
# Idempotency
# =========================================================================================


def request_fingerprint(payload: dict[str, Any]) -> str:
    """Stable hash of a request body, for detecting key reuse with a different payload.

    sort_keys matters: without it, two dicts with identical content but different insertion order
    hash differently and a legitimate retry looks like key reuse.
    """
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()


class IdempotencyConflict(Exception):
    """Same key, different request body."""


class IdempotencyInFlight(Exception):
    """The original request holding this key hasn't finished yet."""


async def claim_idempotency_key(key: str, endpoint: str, fingerprint: str) -> dict | None:
    """Try to claim a key. Returns the stored response on a replay, None if we now own it.

    The claim is an INSERT ... ON CONFLICT DO NOTHING, so claiming is itself atomic: exactly one
    concurrent caller inserts the row and proceeds to do the work. Everyone else conflicts and
    either replays the stored response or, if the first request is still running, gets told to
    retry rather than executing the write a second time.
    """
    async with pool().connection() as conn:
        cur = await conn.execute(
            """
            INSERT INTO idempotency_keys (key, endpoint, request_hash)
            VALUES (%s, %s, %s)
            ON CONFLICT (key, endpoint) DO NOTHING
            """,
            (key, endpoint, fingerprint),
        )
        if cur.rowcount == 1:
            return None  # claimed; caller performs the work

        existing = await (await conn.execute(
            """
            SELECT request_hash, response_json, status_code
              FROM idempotency_keys
             WHERE key = %s AND endpoint = %s
            """,
            (key, endpoint),
        )).fetchone()

    if existing is None:  # row vanished between statements; treat as claimable
        return None
    if existing["request_hash"] != fingerprint:
        raise IdempotencyConflict(
            f"Idempotency key {key!r} was already used for a different request body."
        )
    if existing["response_json"] is None:
        raise IdempotencyInFlight(
            f"A request with idempotency key {key!r} is still in flight."
        )
    return {"response": existing["response_json"], "status_code": existing["status_code"]}


async def store_idempotent_response(
    key: str, endpoint: str, response: dict, status_code: int = 200
) -> None:
    async with pool().connection() as conn:
        await conn.execute(
            """
            UPDATE idempotency_keys
               SET response_json = %s, status_code = %s
             WHERE key = %s AND endpoint = %s
            """,
            (Jsonb(response), status_code, key, endpoint),
        )


async def release_idempotency_key(key: str, endpoint: str) -> None:
    """Drop an unfinished claim so a failed request can be retried with the same key.

    Without this, a request that errored after claiming would leave the key permanently
    in-flight, and every retry would be rejected instead of being allowed to try again.
    """
    async with pool().connection() as conn:
        await conn.execute(
            """
            DELETE FROM idempotency_keys
             WHERE key = %s AND endpoint = %s AND response_json IS NULL
            """,
            (key, endpoint),
        )

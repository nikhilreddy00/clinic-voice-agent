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
import math
import os
import re
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


def target_name() -> str:
    """Just the database name — safe for an unauthenticated response.

    This is the half of describe_target() that answers "local or cloud?" ("clinic_dev" vs
    "postgres") while carrying no host, no username, and no project identifier. /health is the
    one route with no service token, and start_demo.sh puts it behind a public ngrok tunnel.
    """
    path = DATABASE_URL.rsplit("/", 1)[-1]
    return path.split("?", 1)[0] or "unknown"


def describe_target() -> str:
    """Where this process is actually writing, with the password removed.

    Exists because "which database is it storing in?" was a real question during live testing,
    and the honest answer needed `ps eww` on the running process. A service that holds
    appointments should say what it is connected to when it starts, every time — the failure
    this prevents is booking happily into a throwaway local database while someone refreshes a
    cloud dashboard and sees nothing.
    """
    url = DATABASE_URL
    if "@" in url:
        scheme, _, rest = url.partition("://")
        creds, _, hostpart = rest.rpartition("@")
        user, sep, _password = creds.partition(":")
        # Only print a mask where a password actually exists — "postgres:****@localhost" for a
        # trust-auth local socket is a small lie, and this line exists to answer questions.
        return f"{scheme}://{user}{':****' if sep else ''}@{hostpart}"
    return url


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
        await migrate_patient_identity(conn)
        await _seed(conn)


async def migrate_patient_identity(conn: AsyncConnection) -> dict[str, int]:
    """Re-key `patients` from (clinic, phone) to (clinic, phone, DOB). Idempotent.

    `CREATE TABLE IF NOT EXISTS` cannot change a constraint on a table that already exists, so
    a database created before this change keeps the old one-person-per-phone key and every
    `ON CONFLICT (clinic_id, phone, date_of_birth)` fails outright. This runs on every boot and
    does nothing once applied.

    Order matters and is the reason this is Python rather than SQL in schema.sql: the stored
    dates have to be canonicalized BEFORE the new key exists (normalizing afterwards could
    collide two rows the constraint had already accepted), and canonicalizing can itself create
    duplicates that must be merged before the key can be added at all.
    """
    counts = {"normalized": 0, "merged": 0}

    existing = await (await conn.execute(
        """
        SELECT conname FROM pg_constraint
         WHERE conrelid = 'patients'::regclass AND contype = 'u'
        """
    )).fetchall()
    names = {r["conname"] for r in existing}
    if "patients_clinic_id_phone_date_of_birth_key" in names:
        return counts  # already migrated

    await conn.execute(
        "ALTER TABLE patients DROP CONSTRAINT IF EXISTS patients_clinic_id_phone_key"
    )

    rows = await (await conn.execute(
        "SELECT id, date_of_birth FROM patients WHERE date_of_birth IS NOT NULL"
    )).fetchall()
    for row in rows:
        canonical = normalize_dob(row["date_of_birth"])
        if canonical != row["date_of_birth"]:
            await conn.execute(
                "UPDATE patients SET date_of_birth = %s WHERE id = %s", (canonical, row["id"])
            )
            counts["normalized"] += 1

    # Canonicalization can turn "3/5/2001" and "03/05/2001" into two rows for one person.
    # Keep the oldest (its id is referenced elsewhere) and repoint the rest onto it.
    dupes = await (await conn.execute(
        """
        SELECT min(id) AS keep, array_agg(id) AS ids
          FROM patients
         WHERE date_of_birth IS NOT NULL
         GROUP BY clinic_id, phone, date_of_birth
        HAVING count(*) > 1
        """
    )).fetchall()
    for group in dupes:
        drop = [i for i in group["ids"] if i != group["keep"]]
        await conn.execute(
            "UPDATE bookings SET patient_id = %s WHERE patient_id = ANY(%s)",
            (group["keep"], drop),
        )
        await conn.execute(
            "UPDATE caller_memory SET patient_id = %s WHERE patient_id = ANY(%s)",
            (group["keep"], drop),
        )
        await conn.execute("DELETE FROM patients WHERE id = ANY(%s)", (drop,))
        counts["merged"] += len(drop)

    await conn.execute(
        """
        ALTER TABLE patients
          ADD CONSTRAINT patients_clinic_id_phone_date_of_birth_key
          UNIQUE (clinic_id, phone, date_of_birth)
        """
    )
    return counts


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

    for topic, content in seed_data.CLINIC_FACTS.items():
        await conn.execute(
            """
            INSERT INTO clinic_facts (clinic_id, topic, content) VALUES (%s, %s, %s)
            ON CONFLICT (clinic_id, topic) DO UPDATE SET content = EXCLUDED.content
            """,
            (clinic_id, topic, content),
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
    "call_tool_metrics",
    "call_turn_metrics",
    "call_metrics",
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
    phone: str | None = None,
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

            # Phase 13: the booking is what creates the patient record, so the NEXT call from
            # this number is a returning caller. Only with a phone AND a FULL DOB -- a patient
            # row without a DOB can never be verified, so it would be memory that is
            # permanently useless for anything except a name we are not allowed to speak; and a
            # row enrolled with a PARTIAL one is worse than useless, because that fragment
            # becomes the credential (see is_full_dob). The booking itself still records
            # whatever the caller said, so nothing is lost from the appointment.
            patient_id = None
            if phone and is_full_dob(date_of_birth):
                patient_id = await _upsert_patient(
                    conn, slot["clinic_id"], phone, patient_name, date_of_birth
                )

            await conn.execute(
                """
                INSERT INTO bookings (confirmation_id, clinic_id, slot_id, patient_id,
                                      patient_name, reason,
                                      date_of_birth, new_patient, symptom_notes)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    confirmation_id,
                    slot["clinic_id"],
                    slot["slot_id"],
                    patient_id,
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


# =========================================================================================
# PHASE 13 — caller memory, identity verification, and the expanded tool surface
# =========================================================================================
#
# One rule governs everything below: **recognising a phone number is not authentication.**
# Caller ID is trivially spoofable, so the ANI alone unlocks nothing. Every function that
# returns or mutates PHI takes `phone` AND `date_of_birth` and re-verifies the pair here, in
# the database layer, on every single call. The agent also holds an `identity_verified` flag
# in CallState and refuses to invoke these tools without it — but that flag is a UX gate in a
# process that talks to a language model, and a language model is not a security boundary.
# This is.
#
# The failure mode is deliberately "no rows", not "error": a wrong DOB looks exactly like a
# number the clinic has never seen, so an attacker learns nothing about who is a patient here.


class NotVerified(Exception):
    """The (phone, date_of_birth) pair does not match a patient on file. Fails closed."""


class BookingNotFound(Exception):
    """No confirmed booking with that confirmation id belongs to this verified patient."""


def normalize_dob(value: str | None) -> str:
    """Reduce a date of birth to a comparable form.

    The agent normalizes speech to MM/DD/YYYY, but "03/15/1990", "3/15/1990", and "3-15-1990"
    are the same birthday, and a caller must not fail verification over a missing leading zero
    or a dash. Digits-only is not enough for exactly that reason ("3151990" != "03151990"), so
    a three-part value is zero-padded to MM DD YYYY; anything else falls back to its digits.

    Not a date parser on purpose. It never has to decide whether 03/04 is March or April — both
    sides of the comparison come from the same clinic in the same format.
    """
    parts = [g for g in re.split(r"\D+", (value or "").strip()) if g]
    if len(parts) == 3:
        m, d, y = parts
        return f"{m.zfill(2)}{d.zfill(2)}{y.zfill(4)}"
    return "".join(parts)


def is_full_dob(value: str | None) -> bool:
    """Whether this is a whole date of birth rather than a fragment of one.

    THIS IS A CREDENTIAL CHECK, NOT A FORMAT PREFERENCE. `normalize_dob` reduces to digits, so
    "March 1990" and "1990" both normalize to "1990" — and until this existed, a booking made
    with a partial date ENROLLED that fragment as the patient's verification secret. Measured
    against a live scratch database: a patient booked with date_of_birth "1990" could then be
    verified by saying "1990", or "March 1990", or any other phrasing containing that year.
    A four-digit credential shared by everyone born that year is not a second factor.

    Eight digits is MMDDYYYY, which is what the agent normalizes speech to and what the
    scheduling API stores. Anything shorter is a fragment; anything longer is not a date.
    """
    return len(normalize_dob(value)) == 8


async def _upsert_patient(
    conn: AsyncConnection, clinic_id: int, phone: str | None, name: str, date_of_birth: str
) -> int:
    """Create or refresh the patient record for ONE PERSON on a phone. Returns patients.id.

    Keyed on ``(clinic_id, phone, date_of_birth)``. Keying on the phone alone was a live
    defect, and the two rules that produced it were each individually right:

      * this function refused to overwrite a date of birth already on file, because the DOB is
        the verification secret and a later booking that mistyped it would lock the real
        patient out; and
      * ``_verify`` stops dead when an enrolled number presents a wrong DOB, because falling
        through to a name+DOB search would let anyone holding an enrolled patient's handset
        reach a stranger's chart.

    Together they meant the FIRST person to book from a number owned it permanently. A second
    household member booking from the same handset overwrote ``name`` — the row was not
    protected — but kept the first person's DOB, so they could book and could never verify.
    Measured 2026-09-03: "Joe" (DOB 03/05/2001) booked from a number already enrolled to
    "Nick" (DOB 05/08/2003), got confirmation E938C8F6, called back, and was refused twice on
    the very date of birth he had just booked with. Worse than the lockout, the row was then a
    merge of two people — named Joe, carrying Nick's DOB, owning Nick's bookings — so anyone
    verifying with Nick's DOB would have been greeted as Joe and shown Nick's appointments.

    With the DOB in the key, a different person on the same number is simply a different row,
    and no COALESCE is needed: nothing can reach another person's record to overwrite.

    The stored DOB is NORMALIZED (``normalize_dob``). The column used to hold whatever the
    caller said, with normalization applied on every comparison; that is fine for comparing and
    useless for a key, where "3/5/2001" and "03/05/2001" would file one person twice.
    """
    row = await (await conn.execute(
        """
        INSERT INTO patients (clinic_id, phone, name, date_of_birth)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (clinic_id, phone, date_of_birth) DO UPDATE
            SET name = EXCLUDED.name
        RETURNING id
        """,
        (clinic_id, phone, name, normalize_dob(date_of_birth)),
    )).fetchone()
    return row["id"]


def _same_name(a: str | None, b: str | None) -> bool:
    """Case- and spacing-insensitive name comparison. Deliberately NOT fuzzy.

    A name is half of a two-factor check here, and fuzzy matching weakens a credential — the
    point of accepting "  dana   reyes " is a transcription artifact, not accepting "D. Reyes".
    """
    return " ".join((a or "").lower().split()) == " ".join((b or "").lower().split())


async def _verify(
    conn: AsyncConnection, phone: str, date_of_birth: str, name: str | None = None
) -> dict[str, Any]:
    """Resolve a caller to a patient row, or raise NotVerified. The single gate.

    TWO paths, both two-factor, tried in this order:

    1. **Phone + DOB.** The fast path for a caller the clinic has already enrolled.
    2. **Name + DOB against confirmed bookings**, but ONLY when this number has no patient
       record yet. This exists because keying on the ANI alone made whole classes of
       appointment permanently unreachable: anything booked at the front desk or on the web,
       anything booked before this number was ever seen, and any caller phoning from a
       different handset. The agent would ask for a date of birth it could never match and
       then dead-end at a hand-off — which is exactly what happened on a live call.
       Name + DOB is the check a real front desk runs, and it is still two factors.

    On a path-2 match the caller is **enrolled**: a patient record is created for this number
    and the matching orphan bookings are attached to it, so the next call is recognised by the
    number alone and never has to ask for a name again.

    Path 2 is skipped when the number ALREADY belongs to an enrolled patient whose DOB did not
    match. That is not a convenience gap, it is the hijack this ordering prevents: without it,
    anyone holding an enrolled patient's handset could verify as somebody else by name and DOB
    and have that person's bookings re-pointed at the phone they are holding.
    """
    # A FRAGMENT IS NOT A CREDENTIAL. "1990" normalizes to four digits and would match any
    # patient enrolled with a partial date — see is_full_dob. Refused with the IDENTICAL body
    # a wrong date gets: saying "that is not a full date of birth" only when the number is
    # enrolled would turn the malformed-input path into the patient-enumeration oracle the
    # 403 exists to prevent. The agent answers a partial date in the reducer instead, before
    # it ever becomes a request (reducer._MISSING_DOB_RESULT).
    if not is_full_dob(date_of_birth):
        raise NotVerified("Identity not verified")

    if phone:
        # EVERY person enrolled on this number, not just one. A shared handset holds a whole
        # household, and fetching a single row here is what let the first caller own the number
        # forever — see _upsert_patient. The DOB still decides WHICH of them is calling.
        rows = await (await conn.execute(
            "SELECT id, name, date_of_birth FROM patients WHERE phone = %s", (phone,)
        )).fetchall()
        if rows:
            for row in rows:
                if normalize_dob(row["date_of_birth"]) == normalize_dob(date_of_birth):
                    return row
            # An enrolled number presenting a DOB that matches NOBODY on it. Still a hard stop,
            # and still for the original reason: falling through to the name+DOB search below
            # would let anyone holding this handset reach a stranger's chart by naming them.
            raise NotVerified("Identity not verified")

    if not name:
        raise NotVerified("Identity not verified")

    # An already-enrolled patient, reached from a number the clinic does not know — a caller on
    # their partner's phone, a new handset, or a withheld ANI. Same two factors; no rebinding,
    # because a patient row holds one number and silently moving it would be a surprise.
    existing = await (await conn.execute(
        "SELECT id, name, date_of_birth FROM patients WHERE date_of_birth IS NOT NULL"
    )).fetchall()
    for row in existing:
        if _same_name(row["name"], name) and normalize_dob(row["date_of_birth"]) == normalize_dob(
            date_of_birth
        ):
            return row

    # DOB is compared in Python, not SQL: the column stores what the caller spoke, and
    # normalize_dob is the only thing that knows "3-15-1990" and "03/15/1990" are one date.
    candidates = await (await conn.execute(
        """
        SELECT confirmation_id, clinic_id, patient_name, date_of_birth
          FROM bookings
         WHERE status = 'confirmed'
           AND patient_id IS NULL
           AND date_of_birth IS NOT NULL
        """
    )).fetchall()
    matched = [
        b for b in candidates
        if _same_name(b["patient_name"], name)
        and normalize_dob(b["date_of_birth"]) == normalize_dob(date_of_birth)
    ]
    if not matched:
        raise NotVerified("Identity not verified")

    clinic_id = matched[0]["clinic_id"]
    # NULL, not "": phone is UNIQUE per clinic, and an empty string would collide the moment
    # a second caller verified with no ANI (the local path). NULLs do not collide.
    patient_id = await _upsert_patient(
        conn, clinic_id, phone or None, matched[0]["patient_name"], date_of_birth
    )
    await conn.execute(
        "UPDATE bookings SET patient_id = %s WHERE confirmation_id = ANY(%s)",
        (patient_id, [b["confirmation_id"] for b in matched]),
    )
    return {"id": patient_id, "name": matched[0]["patient_name"],
            "date_of_birth": date_of_birth}


async def caller_memory(phone: str) -> dict[str, Any]:
    """What is known about a number BEFORE verification. Deliberately almost nothing.

    Returns whether this number has called before and how many confirmed appointments it has —
    enough for the agent to say "welcome back" and offer to look something up, and nothing that
    identifies anybody. The name is NOT returned: speaking a patient's name to whoever happens
    to be holding their phone is a disclosure, and this function runs before any verification.
    """
    if not phone:
        return {"known": False, "upcoming_appointments": 0}
    async with pool().connection() as conn:
        # Aggregated across EVERY patient on the number, deliberately. A phone is a household:
        # since `patients` became (clinic, phone, DOB)-keyed, one number can hold several
        # people, and this used to `GROUP BY p.id` and take the first row — reporting one
        # arbitrary household member's count, non-deterministically. Measured: three confirmed
        # appointments on a number reported as two.
        #
        # The number, not the person, is the right unit here: this runs BEFORE verification and
        # the caller has not said who they are yet. It still discloses nothing identifying —
        # no name, no times, no confirmation ids — which is the property that matters.
        row = await (await conn.execute(
            """
            SELECT COUNT(*) FILTER (
                       WHERE b.status = 'confirmed' AND s.start_time >= now()
                   ) AS upcoming,
                   COUNT(DISTINCT p.id) AS people
              FROM patients p
              LEFT JOIN bookings b ON b.patient_id = p.id
              LEFT JOIN slots s ON s.id = b.slot_id
             WHERE p.phone = %s
            """,
            (phone,),
        )).fetchone()
    if row is None or not row["people"]:
        return {"known": False, "upcoming_appointments": 0}
    return {"known": True, "upcoming_appointments": int(row["upcoming"] or 0)}


async def verify_identity(
    *, phone: str, date_of_birth: str, name: str | None = None
) -> dict[str, Any]:
    """Check a caller against the records: phone + DOB, or name + DOB. See _verify."""
    async with pool().connection() as conn:
        async with conn.transaction():  # path 2 enrolls, so this can write
            patient = await _verify(conn, phone, date_of_birth, name)
    return {"patient_id": patient["id"], "name": patient["name"]}


async def list_appointments(
    *, phone: str, date_of_birth: str, name: str | None = None
) -> list[dict[str, Any]]:
    """Upcoming confirmed appointments for a verified caller."""
    async with pool().connection() as conn:
        async with conn.transaction():
            patient = await _verify(conn, phone, date_of_birth, name)
            rows = await (await conn.execute(
                """
                SELECT b.confirmation_id, b.slot_id, b.reason, s.start_time,
                       pr.name AS provider_name
                  FROM bookings b
                  JOIN slots s ON s.id = b.slot_id
                  JOIN providers pr ON pr.id = s.provider_id
                 WHERE b.patient_id = %s
                   AND b.status = 'confirmed'
                   AND s.start_time >= now()
                 ORDER BY s.start_time
                """,
                (patient["id"],),
            )).fetchall()
    return rows


async def cancel_appointment(
    *, confirmation_id: str, phone: str, date_of_birth: str, reason: str | None = None,
    name: str | None = None,
) -> dict[str, Any]:
    """Cancel a verified caller's appointment and return the slot to the pool.

    Idempotent by construction rather than by an idempotency key: cancelling an
    already-cancelled booking returns the same success payload instead of a 404, because a
    retried voice turn must not tell the caller their cancellation failed when it did not.
    """
    async with pool().connection() as conn:
        async with conn.transaction():
            patient = await _verify(conn, phone, date_of_birth, name)
            booking = await (await conn.execute(
                """
                SELECT confirmation_id, slot_id, status
                  FROM bookings
                 WHERE confirmation_id = %s AND patient_id = %s
                """,
                (confirmation_id, patient["id"]),
            )).fetchone()
            if booking is None:
                raise BookingNotFound(f"No appointment {confirmation_id} for this caller")
            if booking["status"] == "cancelled":
                return {"confirmation_id": confirmation_id, "status": "cancelled"}

            await conn.execute(
                "UPDATE bookings SET status = 'cancelled' WHERE confirmation_id = %s",
                (confirmation_id,),
            )
            await conn.execute(
                """
                UPDATE slots SET status = 'available', hold_id = NULL, hold_expires_at = NULL
                 WHERE id = %s
                """,
                (booking["slot_id"],),
            )
            if reason:
                await conn.execute(
                    """
                    INSERT INTO staff_tasks (clinic_id, patient_id, kind, payload)
                    SELECT clinic_id, %s, 'cancellation_note', %s FROM bookings
                     WHERE confirmation_id = %s
                    """,
                    (patient["id"], Jsonb({"confirmation_id": confirmation_id, "reason": reason}),
                     confirmation_id),
                )
    return {"confirmation_id": confirmation_id, "status": "cancelled"}


async def reschedule_appointment(
    *, confirmation_id: str, new_slot_id: int, phone: str, date_of_birth: str,
    name: str | None = None,
) -> dict[str, Any]:
    """Move a verified caller's appointment to a different open slot.

    One transaction, and the new slot is taken by the same compare-and-swap that `hold_slot`
    uses, so a concurrent caller cannot win the same slot. There is no hold-then-book dance
    here: the caller already has an appointment, so the failure mode a hold protects against
    (losing the slot while reading it back) is bounded — if the swap fails, they still have
    their original time and the agent offers another.

    Idempotent: rescheduling to the slot the booking already occupies returns success.
    """
    async with pool().connection() as conn:
        async with conn.transaction():
            patient = await _verify(conn, phone, date_of_birth, name)
            booking = await (await conn.execute(
                """
                SELECT confirmation_id, slot_id, status
                  FROM bookings
                 WHERE confirmation_id = %s AND patient_id = %s
                """,
                (confirmation_id, patient["id"]),
            )).fetchone()
            if booking is None or booking["status"] != "confirmed":
                raise BookingNotFound(
                    f"No confirmed appointment {confirmation_id} for this caller"
                )
            old_slot_id = booking["slot_id"]
            if old_slot_id != new_slot_id:
                taken = await (await conn.execute(
                    """
                    UPDATE slots
                       SET status = 'booked', hold_id = NULL, hold_expires_at = NULL
                     WHERE id = %s
                       AND start_time >= now()
                       AND (status = 'available'
                            OR (status = 'held' AND hold_expires_at < now()))
                    RETURNING id
                    """,
                    (new_slot_id,),
                )).fetchone()
                if taken is None:
                    raise SlotUnavailable(f"Slot {new_slot_id} is not available")

                await conn.execute(
                    "UPDATE bookings SET slot_id = %s WHERE confirmation_id = %s",
                    (new_slot_id, confirmation_id),
                )
                await conn.execute(
                    """
                    UPDATE slots SET status = 'available', hold_id = NULL, hold_expires_at = NULL
                     WHERE id = %s
                    """,
                    (old_slot_id,),
                )

            row = await (await conn.execute(
                """
                SELECT s.id AS slot_id, s.start_time, pr.name AS provider_name
                  FROM slots s JOIN providers pr ON pr.id = s.provider_id
                 WHERE s.id = %s
                """,
                (new_slot_id,),
            )).fetchone()

    return {
        "confirmation_id": confirmation_id,
        "slot_id": row["slot_id"],
        "start_time": row["start_time"],
        "provider_name": row["provider_name"],
    }


async def create_staff_task(
    *, kind: str, phone: str, date_of_birth: str, payload: dict[str, Any],
    name: str | None = None,
) -> dict[str, Any]:
    """Queue work for a human. The agent creates the task; it never completes it.

    Refills above all: an automated system that says "your refill is approved" has practised
    medicine. This returns a task id and nothing that resembles an approval, and the tool
    description tells the model to say only that a staff member will follow up.
    """
    async with pool().connection() as conn:
        async with conn.transaction():
            patient = await _verify(conn, phone, date_of_birth, name)
            row = await (await conn.execute(
                """
                INSERT INTO staff_tasks (clinic_id, patient_id, kind, payload)
                SELECT clinic_id, id, %s, %s FROM patients WHERE id = %s
                RETURNING id
                """,
                (kind, Jsonb(payload), patient["id"]),
            )).fetchone()
    return {"task_id": row["id"], "kind": kind, "status": "open"}


async def clinic_info(topic: str | None = None) -> dict[str, Any]:
    """Curated clinic facts. Not PHI, no verification — the one open tool.

    An unknown topic returns the list of topics that DO exist rather than an error, so the
    model's next move is to pick a real one instead of inventing an address.
    """
    async with pool().connection() as conn:
        rows = await (await conn.execute(
            "SELECT topic, content FROM clinic_facts ORDER BY topic"
        )).fetchall()
    facts = {r["topic"]: r["content"] for r in rows}
    if topic and topic in facts:
        return {"topic": topic, "content": facts[topic], "topics": sorted(facts)}
    return {"topic": topic, "content": None, "topics": sorted(facts)}


# =========================================================================================
# CALL METRICS (Phase 16)
# =========================================================================================
#
# `logs/calls.jsonl` was a shared-filesystem bus between two services. On Railway they are
# separate containers with no shared volume, so the dashboard was empty in the only deployment
# that counts. The agent now POSTs one batch per call here at teardown.
#
# Nothing in these rows is clinical: no name, no date of birth, no transcript. That is a
# property of what the agent sends, and `main.py` is where it is enforced.

# How many recent calls /metrics aggregates over. Unbounded was the old behaviour and it meant
# one page load re-read all of history; a fixed window keeps the endpoint O(window).
METRICS_WINDOW_CALLS = int(os.getenv("CLINIC_METRICS_WINDOW", "200"))


async def record_call_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    """Store one call's operational metrics. Idempotent on (clinic, call_id).

    A re-post REPLACES the call's rows rather than appending a second copy: the agent posts once
    at teardown, but "once" is a property of a network nobody controls, and a duplicated call
    would double-count in every percentile on the dashboard.
    """
    call_id = str(payload.get("call_id") or "").strip()
    if not call_id:
        return {"ok": False, "error": "call_id is required"}

    turns = payload.get("turns") or []
    tools = payload.get("tools") or []

    async with pool().connection() as conn:
        async with conn.transaction():
            clinic_id = await _clinic_id(conn)
            row = await (await conn.execute(
                """
                INSERT INTO call_metrics
                    (clinic_id, call_id, mode, outcome, turns, tool_total, tool_success,
                     started_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (clinic_id, call_id) DO UPDATE SET
                    mode = EXCLUDED.mode,
                    outcome = EXCLUDED.outcome,
                    turns = EXCLUDED.turns,
                    tool_total = EXCLUDED.tool_total,
                    tool_success = EXCLUDED.tool_success,
                    started_at = EXCLUDED.started_at,
                    created_at = now()
                RETURNING id
                """,
                (
                    clinic_id,
                    call_id,
                    payload.get("mode"),
                    payload.get("outcome"),
                    len(turns),
                    int(payload.get("tool_total") or 0),
                    int(payload.get("tool_success") or 0),
                    payload.get("started_at"),
                ),
            )).fetchone()
            metrics_id = row["id"]

            # Children are replaced wholesale. ON CONFLICT cannot express "this call's turns are
            # now exactly these" — a shorter re-post would leave the old tail behind.
            await conn.execute(
                "DELETE FROM call_turn_metrics WHERE call_metrics_id = %s", (metrics_id,)
            )
            await conn.execute(
                "DELETE FROM call_tool_metrics WHERE call_metrics_id = %s", (metrics_id,)
            )
            for i, turn in enumerate(turns):
                await conn.execute(
                    """
                    INSERT INTO call_turn_metrics
                        (call_metrics_id, turn_index, asr_ms, llm_ms, tts_ms, e2e_ms,
                         asr_confidence, had_tool_call)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (metrics_id, i, turn.get("asr_ms"), turn.get("llm_ms"), turn.get("tts_ms"),
                     turn.get("e2e_ms"), turn.get("asr_confidence"),
                     bool(turn.get("had_tool_call"))),
                )
            for tool in tools:
                await conn.execute(
                    """
                    INSERT INTO call_tool_metrics
                        (call_metrics_id, endpoint, http_status, latency_ms, success)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (metrics_id, str(tool.get("endpoint") or "unknown"), tool.get("http_status"),
                     tool.get("latency_ms"), bool(tool.get("success"))),
                )

    return {"ok": True, "call_id": call_id, "turns": len(turns), "tools": len(tools)}


async def fetch_call_events(limit_calls: int | None = None) -> list[dict[str, Any]]:
    """The most recent calls' metrics, in the event shape `metrics.aggregate_metrics` expects.

    Returning events rather than a finished aggregate is deliberate: the aggregation logic and
    its tests predate this change and are unaffected by where the rows come from. Only the
    transport moved.
    """
    limit = limit_calls or METRICS_WINDOW_CALLS
    async with pool().connection() as conn:
        calls = await (await conn.execute(
            """
            SELECT id, call_id, mode, outcome, turns, created_at
            FROM call_metrics
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (limit,),
        )).fetchall()
        if not calls:
            return []
        ids = [c["id"] for c in calls]
        turn_rows = await (await conn.execute(
            """
            SELECT call_metrics_id, asr_ms, llm_ms, tts_ms, e2e_ms, asr_confidence, had_tool_call
            FROM call_turn_metrics WHERE call_metrics_id = ANY(%s) ORDER BY turn_index
            """,
            (ids,),
        )).fetchall()
        tool_rows = await (await conn.execute(
            """
            SELECT call_metrics_id, endpoint, http_status, latency_ms, success
            FROM call_tool_metrics WHERE call_metrics_id = ANY(%s) ORDER BY id
            """,
            (ids,),
        )).fetchall()

    by_id = {c["id"]: c for c in calls}
    events: list[dict[str, Any]] = []
    for row in turn_rows:
        call = by_id[row["call_metrics_id"]]
        events.append({
            "event": "turn",
            "call_id": call["call_id"],
            "asr_ms": row["asr_ms"],
            "llm_ms": row["llm_ms"],
            "tts_ms": row["tts_ms"],
            "e2e_ms": row["e2e_ms"],
            "asr_confidence": row["asr_confidence"],
            "had_tool_call": row["had_tool_call"],
        })
    for row in tool_rows:
        call = by_id[row["call_metrics_id"]]
        events.append({
            "event": "tool",
            "call_id": call["call_id"],
            "endpoint": row["endpoint"],
            "http_status": row["http_status"],
            "latency_ms": row["latency_ms"],
            "success": row["success"],
        })
    for call in calls:
        e2e = [r["e2e_ms"] for r in turn_rows
               if r["call_metrics_id"] == call["id"] and r["e2e_ms"] is not None]
        events.append({
            "event": "call_summary",
            "call_id": call["call_id"],
            "ts": call["created_at"].isoformat(),
            "mode": call["mode"],
            "outcome": call["outcome"],
            "turns": call["turns"],
            "latency_ms": {"e2e": {"p50": _p50(e2e)}},
        })
    return events


def _p50(values: list[float]) -> int | None:
    """Nearest-rank median, matching the agent-side collector and app/metrics.py."""
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[max(0, math.ceil(0.5 * len(ordered)) - 1)])

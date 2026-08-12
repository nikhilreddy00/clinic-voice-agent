"""Concurrency guarantees — the reason Phase 9 exists.

The SQLite implementation read a slot's status, decided in Python, then wrote it. Two callers
could both read 'available' and both write 'held'; the second silently clobbered the first's
hold_id. The old suite's `test_double_hold_is_rejected` only ever issued the two requests
SEQUENTIALLY, so it passed against racy code — the bug was real and invisible.

These tests fire genuinely concurrent coroutines at the same slot and assert exactly one winner.
They drive db.* directly rather than going through TestClient, which serialises requests and
therefore cannot reproduce a race at all.

`asyncio.gather` on a pool-backed coroutine gives real overlap: each call checks out its own
connection, so the statements interleave inside Postgres rather than queueing behind one another
in Python.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from app import db

CONCURRENCY = 20


async def _first_available_slot_id() -> int:
    slots = await db.list_available_slots()
    assert slots, "expected seeded availability"
    return slots[0]["slot_id"]


async def _settle(coros):
    """Run coroutines concurrently, splitting successes from raised exceptions."""
    results = await asyncio.gather(*coros, return_exceptions=True)
    ok = [r for r in results if not isinstance(r, BaseException)]
    errs = [r for r in results if isinstance(r, BaseException)]
    return ok, errs


# =========================================================================================
# hold-slot
# =========================================================================================


async def test_concurrent_holds_on_one_slot_produce_exactly_one_winner(pool):
    """The core race. Under the old read-then-write, several callers could all 'win'."""
    slot_id = await _first_available_slot_id()

    ok, errs = await _settle([db.hold_slot(slot_id) for _ in range(CONCURRENCY)])

    assert len(ok) == 1, f"expected exactly 1 successful hold, got {len(ok)}"
    assert len(errs) == CONCURRENCY - 1
    assert all(isinstance(e, db.SlotUnavailable) for e in errs), (
        f"losers must fail with SlotUnavailable, got {[type(e).__name__ for e in errs]}"
    )


async def test_the_winning_hold_id_is_the_one_actually_stored(pool):
    """A clobbered write is the subtle half of the bug: the winner's hold_id must survive.

    Under the old code a loser's UPDATE could land last and overwrite the winner's hold_id, so
    the caller holding a 'successful' hold could no longer confirm it.
    """
    slot_id = await _first_available_slot_id()

    ok, _ = await _settle([db.hold_slot(slot_id) for _ in range(CONCURRENCY)])
    winner = ok[0]

    async with pool.connection() as conn:
        row = await (await conn.execute(
            "SELECT status, hold_id FROM slots WHERE id = %s", (slot_id,)
        )).fetchone()

    assert row["status"] == "held"
    assert row["hold_id"] == winner["hold_id"], "the stored hold_id must be the winner's"


async def test_an_expired_hold_is_reclaimable_by_exactly_one_caller(pool):
    """Reclaim is folded into the same compare-and-swap, so it cannot double-grant either."""
    slot_id = await _first_available_slot_id()
    first = await db.hold_slot(slot_id)

    # Expire the hold without going through the sweeper — the CAS must handle this itself.
    async with pool.connection() as conn:
        await conn.execute(
            "UPDATE slots SET hold_expires_at = %s WHERE id = %s",
            (datetime.now(timezone.utc) - timedelta(seconds=1), slot_id),
        )

    ok, errs = await _settle([db.hold_slot(slot_id) for _ in range(CONCURRENCY)])

    assert len(ok) == 1, "an expired hold must be reclaimable by exactly one caller"
    assert ok[0]["hold_id"] != first["hold_id"], "reclaim must mint a fresh hold_id"
    assert all(isinstance(e, db.SlotUnavailable) for e in errs)


async def test_holding_a_booked_slot_always_fails(pool):
    slot_id = await _first_available_slot_id()
    hold = await db.hold_slot(slot_id)
    await db.confirm_booking(
        hold_id=str(hold["hold_id"]), patient_name="Jane Doe", reason="checkup"
    )

    with pytest.raises(db.SlotUnavailable):
        await db.hold_slot(slot_id)


async def test_holds_on_distinct_slots_all_succeed(pool):
    """Guard against over-correcting into a global lock that serialises unrelated bookings."""
    slots = await db.list_available_slots()
    ids = [s["slot_id"] for s in slots[:10]]
    assert len(ids) == 10

    ok, errs = await _settle([db.hold_slot(i) for i in ids])

    assert len(ok) == 10, f"independent slots must not contend: {errs}"
    assert len({str(r["hold_id"]) for r in ok}) == 10, "each hold must be unique"


# =========================================================================================
# confirm-booking
# =========================================================================================


async def test_concurrent_confirms_of_one_hold_book_exactly_once(pool):
    """Retries racing on the same hold must not produce two bookings for one slot."""
    slot_id = await _first_available_slot_id()
    hold = await db.hold_slot(slot_id)
    hold_id = str(hold["hold_id"])

    ok, errs = await _settle([
        db.confirm_booking(hold_id=hold_id, patient_name="Jane Doe", reason="checkup")
        for _ in range(CONCURRENCY)
    ])

    assert len(ok) == 1, f"expected exactly 1 booking, got {len(ok)}"
    assert all(isinstance(e, db.HoldInvalid) for e in errs), (
        f"losers must fail with HoldInvalid, got {[type(e).__name__ for e in errs]}"
    )

    async with pool.connection() as conn:
        count = await (await conn.execute(
            "SELECT count(*) AS n FROM bookings WHERE slot_id = %s", (slot_id,)
        )).fetchone()
    assert count["n"] == 1, "the slot must have exactly one booking row"


async def test_two_callers_racing_from_hold_to_booking_yield_one_appointment(pool):
    """End-to-end race: two callers, one slot, full hold->confirm each. One appointment."""
    slot_id = await _first_available_slot_id()

    async def book(name: str):
        held = await db.hold_slot(slot_id)
        return await db.confirm_booking(
            hold_id=str(held["hold_id"]), patient_name=name, reason="checkup"
        )

    ok, errs = await _settle([book("Caller A"), book("Caller B")])

    assert len(ok) == 1, "two callers must not both book the same slot"
    assert len(errs) == 1

    async with pool.connection() as conn:
        rows = await (await conn.execute(
            "SELECT patient_name FROM bookings WHERE slot_id = %s", (slot_id,)
        )).fetchall()
    assert len(rows) == 1
    assert rows[0]["patient_name"] == ok[0]["patient_name"]


async def test_expired_hold_cannot_be_confirmed(pool):
    slot_id = await _first_available_slot_id()
    hold = await db.hold_slot(slot_id)

    async with pool.connection() as conn:
        await conn.execute(
            "UPDATE slots SET hold_expires_at = %s WHERE id = %s",
            (datetime.now(timezone.utc) - timedelta(seconds=1), slot_id),
        )

    with pytest.raises(db.HoldInvalid):
        await db.confirm_booking(
            hold_id=str(hold["hold_id"]), patient_name="Too Late", reason="checkup"
        )


async def test_unknown_and_malformed_hold_ids_are_rejected_not_crashed(pool):
    """A non-UUID hold_id must be a clean 409-shaped error, not a cast failure -> 500."""
    import uuid as _uuid

    with pytest.raises(db.HoldInvalid):
        await db.confirm_booking(
            hold_id="not-a-uuid-at-all", patient_name="X", reason="checkup"
        )
    with pytest.raises(db.HoldInvalid):
        await db.confirm_booking(
            hold_id=str(_uuid.uuid4()), patient_name="X", reason="checkup"
        )


# =========================================================================================
# Database-level backstop
# =========================================================================================


async def test_unique_constraint_blocks_a_second_booking_for_one_slot(pool):
    """Defence in depth: if the CAS were ever broken, the schema must still refuse.

    Inserts a duplicate bookings row directly, bypassing the application logic entirely, and
    asserts the database rejects it.
    """
    import psycopg

    slot_id = await _first_available_slot_id()
    hold = await db.hold_slot(slot_id)
    await db.confirm_booking(
        hold_id=str(hold["hold_id"]), patient_name="First", reason="checkup"
    )

    async with pool.connection() as conn:
        clinic = await (await conn.execute(
            "SELECT id FROM clinics WHERE slug = %s", (db.DEFAULT_CLINIC_SLUG,)
        )).fetchone()

        with pytest.raises(psycopg.errors.UniqueViolation):
            await conn.execute(
                """
                INSERT INTO bookings (confirmation_id, clinic_id, slot_id, patient_name, reason)
                VALUES ('DUPE0001', %s, %s, 'Second', 'checkup')
                """,
                (clinic["id"], slot_id),
            )


async def test_hold_field_consistency_constraint_rejects_a_partial_hold(pool):
    """A 'held' row with no expiry would be un-bookable forever and invisible to the sweeper."""
    import psycopg

    slot_id = await _first_available_slot_id()

    async with pool.connection() as conn:
        with pytest.raises(psycopg.errors.CheckViolation):
            await conn.execute(
                "UPDATE slots SET status = 'held', hold_id = NULL, hold_expires_at = NULL"
                " WHERE id = %s",
                (slot_id,),
            )

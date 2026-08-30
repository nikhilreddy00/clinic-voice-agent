"""Tests for the scheduling API: happy path + hold edge cases.

Concurrency guarantees live in test_concurrency.py; idempotency in test_idempotency.py.
"""

import psycopg

from app import db


def _first_slot_id(client) -> int:
    resp = client.get("/availability")
    assert resp.status_code == 200
    slots = resp.json()["slots"]
    assert slots, "expected seeded availability"
    return slots[0]["slot_id"]


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_availability_returns_seeded_slots(client):
    resp = client.get("/availability")
    assert resp.status_code == 200
    slots = resp.json()["slots"]
    assert len(slots) > 0
    slot = slots[0]
    for field in ("slot_id", "provider_id", "provider_name", "specialty",
                  "start_time", "reason_category"):
        assert field in slot


def test_availability_excludes_past_slots(client):
    """Slots whose start_time is already in the past must not be offered (Phase-4 fix #2)."""
    from datetime import datetime, timedelta, timezone

    past = datetime.now(timezone.utc) - timedelta(hours=2)
    future = datetime.now(timezone.utc) + timedelta(hours=2)
    with psycopg.connect(db.DATABASE_URL) as conn:
        clinic_id = conn.execute(
            "SELECT id FROM clinics WHERE slug = %s", (db.DEFAULT_CLINIC_SLUG,)
        ).fetchone()[0]
        # psycopg3 puts executemany on the cursor, not the connection.
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO slots (id, clinic_id, provider_id, start_time, reason_category,"
                " status) VALUES (%s, %s, 1, %s, 'checkup', 'available')",
                [(90001, clinic_id, past), (90002, clinic_id, future)],
            )
        conn.commit()

    ids = [s["slot_id"] for s in client.get("/availability").json()["slots"]]
    assert 90001 not in ids, "past slot should be filtered out"
    assert 90002 in ids, "future slot should still be offered"


async def test_sweeper_refreshes_stale_slots(pool):
    """An always-on server whose seeded slots have aged into the past must self-heal.

    Seeded slots carry fixed ids but dates relative to seed time, so after a few days of uptime
    the whole seeded window falls into the past and availability empties out.

    PHASE-9 BEHAVIOUR CHANGE: this refresh used to happen inside GET /availability, which made
    every read take the write lock. It now runs in the background sweeper, so the trigger under
    test is db.sweep(), and a read is asserted to do nothing on its own.

    Driven through db.* rather than TestClient because the pool is bound to the event loop that
    opened it — reaching into it from a second loop is what a TestClient-based version would do.
    """
    from datetime import datetime, timezone

    async with pool.connection() as conn:
        await conn.execute(
            "UPDATE slots SET start_time = %s WHERE status = 'available'",
            (datetime(2000, 1, 1, 9, 0, tzinfo=timezone.utc),),
        )

    assert await db.list_available_slots() == [], (
        "a pure read must NOT self-heal — that is the sweeper's job now"
    )

    await db.sweep()

    slots = await db.list_available_slots()
    assert slots, "the sweeper should roll stale available slots into the upcoming window"
    now = datetime.now(timezone.utc)
    assert all(s["start_time"] > now for s in slots), "refreshed slots must all be in the future"


def test_availability_filter_by_provider(client):
    resp = client.get("/availability", params={"provider_id": 1})
    assert resp.status_code == 200
    assert all(s["provider_id"] == 1 for s in resp.json()["slots"])


def test_hold_then_confirm_happy_path(client):
    slot_id = _first_slot_id(client)

    hold = client.post("/hold-slot", json={"slot_id": slot_id})
    assert hold.status_code == 200
    hold_id = hold.json()["hold_id"]
    assert hold_id

    # Held slot should no longer show as available.
    available_ids = [s["slot_id"] for s in client.get("/availability").json()["slots"]]
    assert slot_id not in available_ids

    confirm = client.post(
        "/confirm-booking",
        json={"hold_id": hold_id, "patient_name": "Jane Doe", "reason": "annual checkup"},
    )
    assert confirm.status_code == 200
    body = confirm.json()
    assert body["confirmation_id"]
    assert body["slot_id"] == slot_id
    assert body["patient_name"] == "Jane Doe"


def test_double_hold_is_rejected(client):
    slot_id = _first_slot_id(client)
    assert client.post("/hold-slot", json={"slot_id": slot_id}).status_code == 200
    second = client.post("/hold-slot", json={"slot_id": slot_id})
    assert second.status_code == 409


def test_hold_unknown_slot_404(client):
    resp = client.post("/hold-slot", json={"slot_id": 999999})
    assert resp.status_code == 404


def test_confirm_without_valid_hold_is_rejected(client):
    resp = client.post(
        "/confirm-booking",
        json={"hold_id": "not-a-real-hold", "patient_name": "John Roe", "reason": "cough"},
    )
    assert resp.status_code == 409


def test_cannot_book_same_slot_twice(client):
    slot_id = _first_slot_id(client)
    hold_id = client.post("/hold-slot", json={"slot_id": slot_id}).json()["hold_id"]
    client.post(
        "/confirm-booking",
        json={"hold_id": hold_id, "patient_name": "A B", "reason": "checkup"},
    )
    # Slot is now booked; holding it again must fail.
    assert client.post("/hold-slot", json={"slot_id": slot_id}).status_code == 409


def test_health_names_the_database_without_leaking_where_it_is(client):
    """/health is the only route with no service token, and start_demo.sh publishes it through
    an ngrok tunnel. The database NAME answers "local or cloud?"; the host, the username, and
    the Supabase project ref must stay in the startup log."""
    body = client.get("/health").json()
    assert body["database"]
    blob = str(body)
    for leak in ("@", "supabase", "pooler", "postgresql://", "127.0.0.1", ":5432"):
        assert leak not in blob, f"/health leaked {leak!r}"

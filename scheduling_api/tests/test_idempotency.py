"""Idempotency-Key behaviour on the write endpoints.

WHY THIS MATTERS ON A PHONE CALL
--------------------------------
The agent's HTTP client sits inside a voice turn with a timeout. A booking can commit and then
have its response lost — the caller is silent, the agent retries, and without idempotency the
retry books a SECOND appointment (hold_id and confirmation_id are minted fresh per request, so
nothing links the two). The caller hears one confirmation number; the clinic sees two
appointments.

With a key, the retry replays the original response: same confirmation number, one appointment.
"""

from __future__ import annotations

import psycopg

from app import db


def _first_slot_id(client) -> int:
    slots = client.get("/availability").json()["slots"]
    assert slots, "expected seeded availability"
    return slots[0]["slot_id"]


def _booking_count(slot_id: int) -> int:
    with psycopg.connect(db.DATABASE_URL) as conn:
        return conn.execute(
            "SELECT count(*) FROM bookings WHERE slot_id = %s", (slot_id,)
        ).fetchone()[0]


# =========================================================================================
# Replay
# =========================================================================================


def test_repeated_confirm_with_one_key_books_once_and_replays(client):
    """The headline case: a retried confirm returns the original booking, not a second one."""
    slot_id = _first_slot_id(client)
    hold_id = client.post("/hold-slot", json={"slot_id": slot_id}).json()["hold_id"]
    body = {"hold_id": hold_id, "patient_name": "Jane Doe", "reason": "checkup"}
    headers = {"Idempotency-Key": "call-abc-confirm-1"}

    first = client.post("/confirm-booking", json=body, headers=headers)
    second = client.post("/confirm-booking", json=body, headers=headers)

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json() == first.json(), "the retry must replay the original response"
    assert second.headers.get("Idempotent-Replay") == "true"
    assert _booking_count(slot_id) == 1, "a retry must not create a second appointment"


def test_retry_without_a_key_is_rejected_rather_than_double_booking(client):
    """Without a key there is nothing to correlate on — the hold is spent, so the retry 409s.

    Not as good as a replay, but it is the safe failure: the caller gets an error instead of the
    clinic getting two appointments.
    """
    slot_id = _first_slot_id(client)
    hold_id = client.post("/hold-slot", json={"slot_id": slot_id}).json()["hold_id"]
    body = {"hold_id": hold_id, "patient_name": "Jane Doe", "reason": "checkup"}

    assert client.post("/confirm-booking", json=body).status_code == 200
    assert client.post("/confirm-booking", json=body).status_code == 409
    assert _booking_count(slot_id) == 1


def test_repeated_hold_with_one_key_returns_the_same_hold(client):
    slot_id = _first_slot_id(client)
    headers = {"Idempotency-Key": "call-abc-hold-1"}

    first = client.post("/hold-slot", json={"slot_id": slot_id}, headers=headers)
    second = client.post("/hold-slot", json={"slot_id": slot_id}, headers=headers)

    assert first.status_code == second.status_code == 200
    assert first.json()["hold_id"] == second.json()["hold_id"]
    assert second.headers.get("Idempotent-Replay") == "true"


def test_distinct_keys_are_independent(client):
    """Two different callers must not be collapsed into one another's response."""
    slots = client.get("/availability").json()["slots"]
    a, b = slots[0]["slot_id"], slots[1]["slot_id"]

    first = client.post("/hold-slot", json={"slot_id": a}, headers={"Idempotency-Key": "k-a"})
    second = client.post("/hold-slot", json={"slot_id": b}, headers={"Idempotency-Key": "k-b"})

    assert first.json()["slot_id"] == a
    assert second.json()["slot_id"] == b
    assert first.json()["hold_id"] != second.json()["hold_id"]


def test_the_same_key_on_different_endpoints_does_not_collide(client):
    """Keys are scoped per endpoint, so a call id reused across both writes is safe."""
    slot_id = _first_slot_id(client)
    key = {"Idempotency-Key": "call-xyz"}

    hold = client.post("/hold-slot", json={"slot_id": slot_id}, headers=key)
    assert hold.status_code == 200

    confirm = client.post(
        "/confirm-booking",
        json={"hold_id": hold.json()["hold_id"], "patient_name": "A B", "reason": "checkup"},
        headers=key,
    )
    assert confirm.status_code == 200, "the same key on a different endpoint must not replay"
    assert "confirmation_id" in confirm.json()


# =========================================================================================
# Misuse
# =========================================================================================


def test_reusing_a_key_with_a_different_body_is_rejected(client):
    """Silently returning the first call's answer for a different request would be worse."""
    slots = client.get("/availability").json()["slots"]
    headers = {"Idempotency-Key": "reused-key"}

    first = client.post("/hold-slot", json={"slot_id": slots[0]["slot_id"]}, headers=headers)
    assert first.status_code == 200

    second = client.post("/hold-slot", json={"slot_id": slots[1]["slot_id"]}, headers=headers)
    assert second.status_code == 422
    assert "different request" in second.json()["detail"].lower()


def test_a_failed_request_releases_its_key_for_retry(client):
    """A claim left dangling after an error would lock the key out forever.

    Sequence: a hold on an unavailable slot fails (409), then the same key is used for a hold
    that should succeed. If the failed request had kept its claim, the second call would be
    rejected as in-flight rather than being allowed to proceed.
    """
    slot_id = _first_slot_id(client)
    client.post("/hold-slot", json={"slot_id": slot_id})  # occupy it, no key

    headers = {"Idempotency-Key": "retry-after-failure"}
    failed = client.post("/hold-slot", json={"slot_id": slot_id}, headers=headers)
    assert failed.status_code == 409

    with psycopg.connect(db.DATABASE_URL) as conn:
        dangling = conn.execute(
            "SELECT count(*) FROM idempotency_keys WHERE key = %s", ("retry-after-failure",)
        ).fetchone()[0]
    assert dangling == 0, "a failed request must not leave its key claimed"


def test_a_404_also_releases_its_key(client):
    headers = {"Idempotency-Key": "key-on-404"}
    assert client.post("/hold-slot", json={"slot_id": 999999}, headers=headers).status_code == 404

    with psycopg.connect(db.DATABASE_URL) as conn:
        dangling = conn.execute(
            "SELECT count(*) FROM idempotency_keys WHERE key = %s", ("key-on-404",)
        ).fetchone()[0]
    assert dangling == 0


def test_an_in_flight_key_is_told_to_retry_not_executed_twice(client):
    """A claim with no stored response means the original is still running.

    Simulated by claiming the key directly, since the real window is milliseconds wide.
    """
    slot_id = _first_slot_id(client)
    body = {"slot_id": slot_id}
    fingerprint = db.request_fingerprint(body)

    with psycopg.connect(db.DATABASE_URL) as conn:
        conn.execute(
            "INSERT INTO idempotency_keys (key, endpoint, request_hash) VALUES (%s, %s, %s)",
            ("in-flight", "hold-slot", fingerprint),
        )
        conn.commit()

    resp = client.post("/hold-slot", json=body, headers={"Idempotency-Key": "in-flight"})

    assert resp.status_code == 409
    assert resp.headers.get("Retry-After") == "1"
    assert "in flight" in resp.json()["detail"].lower()


# =========================================================================================
# Fingerprinting
# =========================================================================================


def test_fingerprint_is_stable_across_key_order():
    """Without sort_keys, a legitimate retry whose JSON serialised differently would 422."""
    assert db.request_fingerprint({"a": 1, "b": 2}) == db.request_fingerprint({"b": 2, "a": 1})


def test_fingerprint_changes_with_content():
    assert db.request_fingerprint({"slot_id": 1}) != db.request_fingerprint({"slot_id": 2})

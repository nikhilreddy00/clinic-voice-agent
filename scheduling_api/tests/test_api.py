"""Tests for the mock scheduling API: happy path + hold edge cases."""


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

"""Phase 17 — multi-tenancy as a boundary, not a column.

`clinic_id` has been on every row since Phase 9, which made this phase a routing change rather
than a migration. But two lookups in `_verify` carried no tenant at all —
`SELECT ... FROM patients WHERE phone = %s`, and a name+DOB scan across the ENTIRE patients
table — and one of them is reachable through the ordinary verification path. With one clinic
that is invisible. With two it is a cross-tenant PHI disclosure.

So every test here sets up the case that finds it: **the same phone number and the same date of
birth enrolled at both clinics**. That is not contrived — a phone number is a household and a
birthday is one of 36,525 values; two clinics in one deployment will collide on both. Under a
correct boundary each clinic sees exactly its own patient and nothing of the other's.
"""

import pytest

from app import db, seed_data

from conftest import _DB_UP

PHONE = "+15551234321"
DOB = "07/04/1982"

GROVE = {"X-Clinic-Slug": "grove-family"}
BAYSIDE = {"X-Clinic-Slug": "bayside-health"}

pytestmark = pytest.mark.skipif(not _DB_UP, reason="no Postgres")


def _book(client, headers, name) -> dict:
    slots = client.get("/availability", headers=headers).json()["slots"]
    assert slots, f"no slots for {headers} — the tenant was not seeded"
    hold = client.post("/hold-slot", json={"slot_id": slots[0]["slot_id"]},
                       headers=headers).json()
    resp = client.post("/confirm-booking", json={
        "hold_id": hold["hold_id"], "patient_name": name, "reason": "checkup",
        "date_of_birth": DOB, "phone": PHONE,
    }, headers=headers)
    assert resp.status_code == 200, resp.text
    return resp.json()


# --- the seam itself ----------------------------------------------------------------------


def test_availability_is_scoped_to_the_tenant(client):
    grove = {s["slot_id"] for s in client.get("/availability", headers=GROVE).json()["slots"]}
    bayside = {s["slot_id"] for s in client.get("/availability",
                                                headers=BAYSIDE).json()["slots"]}
    assert grove and bayside
    assert not (grove & bayside)


def test_a_tenant_cannot_hold_another_tenants_slot(client):
    other = client.get("/availability", headers=BAYSIDE).json()["slots"][0]["slot_id"]
    resp = client.post("/hold-slot", json={"slot_id": other}, headers=GROVE)
    # 404, not 409: "someone else has it" would confirm the id exists somewhere.
    assert resp.status_code == 404, resp.text


def test_clinic_facts_are_per_tenant(client):
    grove = client.get("/clinic-info", params={"topic": "hours"}, headers=GROVE).json()
    bayside = client.get("/clinic-info", params={"topic": "hours"}, headers=BAYSIDE).json()
    assert "Grove" in grove["content"]
    assert "Bayside" in bayside["content"]


def test_a_dialed_number_resolves_to_its_tenant(client):
    body = client.get("/clinic", params={"did": seed_data.BAYSIDE.did}).json()
    assert body["slug"] == "bayside-health"
    assert body["timezone"] == "America/Chicago"
    assert body["transfer_number"] == seed_data.BAYSIDE.transfer_number

    assert client.get("/clinic", params={"did": "+19998887777"}).status_code == 404


def test_an_unspecified_tenant_is_the_default_clinic(client):
    """Every pre-Phase-17 caller sends no header. They must land where they always did."""
    plain = {s["slot_id"] for s in client.get("/availability").json()["slots"]}
    grove = {s["slot_id"] for s in client.get("/availability", headers=GROVE).json()["slots"]}
    assert plain == grove


# --- the disclosure this phase closes -------------------------------------------------------


def test_one_number_and_one_birthday_at_two_clinics_are_two_patients(client):
    grove_booking = _book(client, GROVE, "Dana Reyes")
    bayside_booking = _book(client, BAYSIDE, "Sam Okafor")

    at_grove = client.post("/verify-identity",
                           json={"phone": PHONE, "date_of_birth": DOB}, headers=GROVE)
    at_bayside = client.post("/verify-identity",
                             json={"phone": PHONE, "date_of_birth": DOB}, headers=BAYSIDE)
    assert at_grove.json()["name"] == "Dana Reyes"
    assert at_bayside.json()["name"] == "Sam Okafor"
    assert at_grove.json()["patient_id"] != at_bayside.json()["patient_id"]

    grove_appts = client.post("/appointments", json={"phone": PHONE, "date_of_birth": DOB},
                              headers=GROVE).json()["appointments"]
    ids = {a["confirmation_id"] for a in grove_appts}
    assert ids == {grove_booking["confirmation_id"]}, "Grove can see Bayside's appointment"
    assert bayside_booking["confirmation_id"] not in ids


def test_a_stranger_at_another_clinic_cannot_be_reached_by_name(client):
    """The name+DOB fallback used to scan every patient row in the database. It is the path a
    caller from an UNKNOWN number takes, so it is reachable without holding anyone's handset."""
    _book(client, BAYSIDE, "Sam Okafor")
    resp = client.post(
        "/verify-identity",
        json={"phone": "", "date_of_birth": DOB, "name": "Sam Okafor"},
        headers=GROVE,
    )
    assert resp.status_code == 403, resp.text


def test_caller_memory_does_not_count_the_other_clinics_appointments(client):
    _book(client, BAYSIDE, "Sam Okafor")
    body = client.get("/caller-memory", params={"phone": PHONE}, headers=GROVE).json()
    assert body == {"known": False, "upcoming_appointments": 0}

    _book(client, GROVE, "Dana Reyes")
    body = client.get("/caller-memory", params={"phone": PHONE}, headers=GROVE).json()
    assert body["upcoming_appointments"] == 1


def test_a_caller_cannot_reschedule_onto_another_tenants_slot(client):
    booking = _book(client, GROVE, "Dana Reyes")
    foreign = client.get("/availability", headers=BAYSIDE).json()["slots"][0]["slot_id"]
    resp = client.post("/reschedule", json={
        "confirmation_id": booking["confirmation_id"], "new_slot_id": foreign,
        "phone": PHONE, "date_of_birth": DOB,
    }, headers=GROVE)
    assert resp.status_code == 409, resp.text


def test_a_tenant_cannot_cancel_the_other_tenants_booking(client):
    _book(client, GROVE, "Dana Reyes")
    bayside_booking = _book(client, BAYSIDE, "Sam Okafor")
    resp = client.post("/cancel", json={
        "confirmation_id": bayside_booking["confirmation_id"],
        "phone": PHONE, "date_of_birth": DOB,
    }, headers=GROVE)
    assert resp.status_code == 404, resp.text


def test_the_date_filter_uses_the_tenants_own_timezone(client):
    """Bayside is Central. "Tuesday" has to mean Tuesday where the CLINIC is, or a caller in one
    zone silently gets another zone's day — which is the Phase-9 seed bug, one level up."""
    from datetime import datetime

    for headers, clinic in ((GROVE, seed_data.GROVE), (BAYSIDE, seed_data.BAYSIDE)):
        slots = client.get("/availability", headers=headers).json()["slots"]
        local_day = datetime.fromisoformat(slots[0]["start_time"]).astimezone(
            clinic.timezone
        ).date().isoformat()
        filtered = client.get("/availability", params={"date": local_day},
                              headers=headers).json()["slots"]
        assert filtered, f"{clinic.slug}: no slots on its own local day {local_day}"
        for slot in filtered:
            assert datetime.fromisoformat(slot["start_time"]).astimezone(
                clinic.timezone
            ).date().isoformat() == local_day


def test_the_audit_row_names_the_tenant(client):
    import psycopg

    _book(client, BAYSIDE, "Sam Okafor")
    with psycopg.connect(db.DATABASE_URL, row_factory=psycopg.rows.dict_row) as conn:
        rows = conn.execute(
            """
            SELECT c.slug FROM audit_log a JOIN clinics c ON c.id = a.clinic_id
             WHERE a.action = 'confirm_booking'
            """
        ).fetchall()
    assert {r["slug"] for r in rows} == {"bayside-health"}

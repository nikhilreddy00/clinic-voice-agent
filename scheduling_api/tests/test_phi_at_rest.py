"""Phase 17 — PHI at rest: a digested date of birth and an encrypted clinical note.

The rest of the suite already proves the verified flows work; running it with CLINIC_PHI_KEY
set proves they still work over digests. What is left, and what this file is for, is the part
that would otherwise be a claim in a document: that the value in the column is genuinely not the
patient's birthday, that the note is genuinely ciphertext, that the boot migration converts an
existing database exactly once, and that a wrong key fails CLOSED rather than open.
"""

import asyncio

import psycopg
import pytest
from psycopg.rows import dict_row
from fastapi.testclient import TestClient

from app import db
from app.main import app

from conftest import _DB_UP, _truncate_all

KEY = "phase17-test-key-not-a-secret"
DOB = "03/15/1990"
PHONE = "+15551230777"
NOTE = "sore throat for two days, no fever"

pytestmark = pytest.mark.skipif(not _DB_UP, reason="no Postgres")


@pytest.fixture
def unkeyed_client(monkeypatch):
    """A client booted with NO key — the pre-Phase-17 posture, pinned explicitly rather than
    inherited from the environment (the suite is also run with CLINIC_PHI_KEY set)."""
    monkeypatch.setattr(db, "PHI_KEY", "")
    _truncate_all()
    with TestClient(app) as c:
        yield c


@pytest.fixture
def keyed_client(monkeypatch):
    """A client whose API boots WITH a PHI key — so the lifespan migration runs under it."""
    monkeypatch.setattr(db, "PHI_KEY", KEY)
    _truncate_all()
    with TestClient(app) as c:
        yield c


def _book(client, *, dob=DOB, notes=NOTE, phone=PHONE, name="Dana Reyes") -> dict:
    slot_id = client.get("/availability").json()["slots"][0]["slot_id"]
    hold = client.post("/hold-slot", json={"slot_id": slot_id}).json()
    resp = client.post("/confirm-booking", json={
        "hold_id": hold["hold_id"], "patient_name": name, "reason": "checkup",
        "date_of_birth": dob, "symptom_notes": notes, "phone": phone,
    })
    assert resp.status_code == 200, resp.text
    return resp.json()


def _stored(sql: str, params=()) -> list[tuple]:
    with psycopg.connect(db.DATABASE_URL) as conn:
        return conn.execute(sql, params).fetchall()


def _run(fn):
    """Run one db.py coroutine on its OWN connection and loop.

    Not the shared pool: TestClient's lifespan opened that on a different event loop, and
    reusing it from `asyncio.run` fails with "attached to a different loop".
    """
    async def go():
        async with await psycopg.AsyncConnection.connect(
            db.DATABASE_URL, row_factory=dict_row, autocommit=True
        ) as conn:
            return await fn(conn)
    return asyncio.run(go())


def test_the_column_does_not_contain_the_birthday(keyed_client):
    _book(keyed_client)
    stored = _stored("SELECT date_of_birth FROM patients") + _stored(
        "SELECT date_of_birth FROM bookings"
    )
    assert stored, "nothing was written — the assertions below would pass vacuously"
    for (value,) in stored:
        assert value.startswith("h1:")
        # Not just "different" — none of the date is recoverable from it.
        assert "1990" not in value and "0315" not in value


def test_a_digested_dob_still_verifies_and_lists(keyed_client):
    _book(keyed_client)
    resp = keyed_client.post("/verify-identity",
                             json={"phone": PHONE, "date_of_birth": "3-15-1990"})
    assert resp.status_code == 200, resp.text
    assert resp.json()["name"] == "Dana Reyes"

    appts = keyed_client.post("/appointments", json={"phone": PHONE, "date_of_birth": DOB})
    assert appts.status_code == 200
    assert len(appts.json()["appointments"]) == 1


def test_a_wrong_dob_still_fails_closed_under_digests(keyed_client):
    _book(keyed_client)
    resp = keyed_client.post("/verify-identity",
                             json={"phone": PHONE, "date_of_birth": "04/16/1991"})
    assert resp.status_code == 403


def test_the_clinical_note_is_ciphertext_and_decrypts(keyed_client):
    booking = _book(keyed_client)
    (stored,), = _stored("SELECT symptom_notes FROM bookings WHERE confirmation_id = %s",
                         (booking["confirmation_id"],))
    assert stored.startswith("pgp:")
    assert "throat" not in stored

    assert _run(lambda c: db.read_clinical_note(booking["confirmation_id"], c)) == NOTE


def test_the_boot_migration_converts_an_existing_database_exactly_once(
    unkeyed_client, monkeypatch
):
    """The client boots WITHOUT a key, so the row lands in the clear — the pre-Phase-17 state."""
    booking = _book(unkeyed_client)
    (dob,), = _stored("SELECT date_of_birth FROM patients")
    assert dob == "03151990"  # canonical, but plainly the birthday

    monkeypatch.setattr(db, "PHI_KEY", KEY)

    first = _run(db.migrate_phi_at_rest)
    assert first["patient_dobs"] == 1 and first["booking_dobs"] == 1 and first["notes"] == 1

    second = _run(db.migrate_phi_at_rest)
    assert second == {"patient_dobs": 0, "booking_dobs": 0, "notes": 0}, (
        "the migration is not idempotent — a second boot would digest the digest"
    )

    (dob,), = _stored("SELECT date_of_birth FROM patients")
    assert dob.startswith("h1:")
    (notes,), = _stored("SELECT symptom_notes FROM bookings WHERE confirmation_id = %s",
                        (booking["confirmation_id"],))
    assert notes.startswith("pgp:") and "throat" not in notes


def test_changing_the_key_makes_every_caller_unverifiable(keyed_client, monkeypatch):
    """The key IS part of the data. Losing it must fail closed, loudly and completely — never
    fall back to comparing something else."""
    _book(keyed_client)
    monkeypatch.setattr(db, "PHI_KEY", "a-different-key")
    resp = keyed_client.post("/verify-identity", json={"phone": PHONE, "date_of_birth": DOB})
    assert resp.status_code == 403


def test_without_a_key_nothing_changes(unkeyed_client):
    """The dev/test default stays byte-for-byte what Phase 16 stored, so the whole existing
    suite is still testing the same thing."""
    booking = _book(unkeyed_client)
    (dob,), = _stored("SELECT date_of_birth FROM bookings WHERE confirmation_id = %s",
                      (booking["confirmation_id"],))
    assert dob == "03151990"
    (notes,), = _stored("SELECT symptom_notes FROM bookings WHERE confirmation_id = %s",
                        (booking["confirmation_id"],))
    assert notes == NOTE

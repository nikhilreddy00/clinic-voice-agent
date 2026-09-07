"""Phase 17 — the audit trail: every PHI access recorded, and the record immutable.

Two properties, and the second is the one that makes the first worth anything:

  * every access writes a row — INCLUDING a refused one, which is the row an investigation
    actually wants ("four wrong dates of birth on one call");
  * no ordinary statement can rewrite history. The database refuses UPDATE and DELETE on
    `audit_log`, so "append-only" is enforced rather than intended.

And the negative property: the rows say THAT a chart was reached, never what was in it.
"""

import psycopg
import pytest

from app import db

from conftest import _DB_UP

CALLER = "+15551230042"
DOB = "03/15/1990"
WRONG_DOB = "04/16/1991"
CALL_ID = "20260907T000000000000Z"

pytestmark = pytest.mark.skipif(not _DB_UP, reason="no Postgres")


def _book(client) -> dict:
    slot_id = client.get("/availability").json()["slots"][0]["slot_id"]
    hold = client.post("/hold-slot", json={"slot_id": slot_id}).json()
    resp = client.post("/confirm-booking", json={
        "hold_id": hold["hold_id"], "patient_name": "Dana Reyes", "reason": "checkup",
        "date_of_birth": DOB, "symptom_notes": "sore throat", "phone": CALLER,
    }, headers={"X-Call-Id": CALL_ID})
    assert resp.status_code == 200, resp.text
    return resp.json()


def _rows(where: str = "TRUE", params=()) -> list[dict]:
    with psycopg.connect(db.DATABASE_URL, row_factory=psycopg.rows.dict_row) as conn:
        return conn.execute(
            f"SELECT * FROM audit_log WHERE {where} ORDER BY id", params
        ).fetchall()


def test_a_verified_access_is_recorded_against_the_call(client):
    _book(client)
    client.post("/verify-identity", json={"phone": CALLER, "date_of_birth": DOB},
                headers={"X-Call-Id": CALL_ID})
    client.post("/appointments", json={"phone": CALLER, "date_of_birth": DOB},
                headers={"X-Call-Id": CALL_ID})

    actions = [r["action"] for r in _rows()]
    assert "confirm_booking" in actions
    assert "verify_identity" in actions
    assert "list_appointments" in actions

    for row in _rows("action = %s", ("list_appointments",)):
        assert row["call_id"] == CALL_ID       # joins to logs/traces/<call_id>.jsonl
        assert row["actor"] == "voice-agent"
        assert row["resource"] == "patient"
        assert row["resource_id"] is not None  # which chart, by id


def test_a_refused_access_is_recorded_too(client):
    """The row that matters. A denial rolls its transaction back, so an audit write inside it
    would disappear — which is why _audit takes its own connection."""
    _book(client)
    resp = client.post("/verify-identity", json={"phone": CALLER, "date_of_birth": WRONG_DOB},
                       headers={"X-Call-Id": CALL_ID})
    assert resp.status_code == 403

    denied = _rows("action = %s", ("verify_identity:denied",))
    assert len(denied) == 1
    assert denied[0]["call_id"] == CALL_ID
    assert denied[0]["resource_id"] is None  # nobody was identified — there is no chart to name


def test_repeated_refusals_are_all_visible(client):
    """The pattern is the signal: one wrong birthday is a caller misremembering, four is not."""
    _book(client)
    for _ in range(4):
        client.post("/appointments", json={"phone": CALLER, "date_of_birth": WRONG_DOB},
                    headers={"X-Call-Id": CALL_ID})
    assert len(_rows("action = %s", ("list_appointments:denied",))) == 4


def test_the_audit_log_holds_no_clinical_content(client):
    """It records that a chart was reached, never what was in it."""
    _book(client)
    client.post("/staff-tasks", json={
        "kind": "refill", "phone": CALLER, "date_of_birth": DOB,
        "payload": {"medication": "quillfeatherazepam"},
    }, headers={"X-Call-Id": CALL_ID})

    blob = " ".join(str(v) for row in _rows() for v in row.values())
    for secret in ("Dana", "Reyes", DOB, "03151990", "quillfeatherazepam", CALLER, "throat"):
        assert secret not in blob, f"audit_log leaked {secret!r}"


def test_the_log_cannot_be_rewritten(client):
    _book(client)
    assert _rows(), "nothing to protect — the assertions below would pass vacuously"

    with psycopg.connect(db.DATABASE_URL) as conn:
        with pytest.raises(psycopg.errors.RaiseException, match="append-only"):
            conn.execute("UPDATE audit_log SET action = 'nothing happened'")
        conn.rollback()
        with pytest.raises(psycopg.errors.RaiseException, match="append-only"):
            conn.execute("DELETE FROM audit_log")
        conn.rollback()


def test_an_actor_can_identify_itself(client):
    """Today the only client is the agent. A staff console would send its own actor, and the
    column is what makes those distinguishable after the fact."""
    _book(client)
    client.post("/verify-identity", json={"phone": CALLER, "date_of_birth": DOB},
                headers={"X-Call-Id": CALL_ID, "X-Actor": "staff-console"})
    assert any(r["actor"] == "staff-console" for r in _rows("action = %s", ("verify_identity",)))

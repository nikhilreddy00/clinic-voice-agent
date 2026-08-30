"""Phase 13 — caller memory, identity verification, and the verified tool surface.

The adversarial half of this file is the point. Caller ID is spoofable, so every test that
matters here asks the same question: with the right phone number and the WRONG date of birth
(or no DOB at all), does the API still refuse? It must fail closed, and it must fail with the
same message every time — an attacker who can tell "unknown number" from "wrong birthday" can
enumerate who is a patient at this clinic, which is itself a disclosure.
"""

CALLER = "+15551230001"
OTHER_CALLER = "+15559998888"
DOB = "03/15/1990"
WRONG_DOB = "04/16/1991"


def _book(client, *, phone=CALLER, dob=DOB, name="Dana Reyes", slot_index=0) -> dict:
    """Run one full booking so there is a patient on file for `phone`."""
    slots = client.get("/availability").json()["slots"]
    slot_id = slots[slot_index]["slot_id"]
    hold = client.post("/hold-slot", json={"slot_id": slot_id}).json()
    resp = client.post(
        "/confirm-booking",
        json={
            "hold_id": hold["hold_id"],
            "patient_name": name,
            "reason": "checkup",
            "date_of_birth": dob,
            "phone": phone,
        },
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


# --- caller memory ----------------------------------------------------------------------


def test_caller_memory_unknown_number(client):
    body = client.get("/caller-memory", params={"phone": OTHER_CALLER}).json()
    assert body == {"known": False, "upcoming_appointments": 0}


def test_caller_memory_recognises_a_returning_caller(client):
    _book(client)
    body = client.get("/caller-memory", params={"phone": CALLER}).json()
    assert body["known"] is True
    assert body["upcoming_appointments"] == 1


def test_caller_memory_never_returns_identity(client):
    """The pre-verification lookup must not leak a name — it runs before anyone proved who
    they are, and the phone may be in anyone's hand."""
    _book(client)
    body = client.get("/caller-memory", params={"phone": CALLER}).json()
    assert "name" not in body and "date_of_birth" not in body and "patient_id" not in body


# --- verification ------------------------------------------------------------------------


def test_verify_identity_succeeds_on_matching_dob(client):
    _book(client)
    resp = client.post("/verify-identity", json={"phone": CALLER, "date_of_birth": DOB})
    assert resp.status_code == 200
    assert resp.json()["name"] == "Dana Reyes"


def test_verify_accepts_punctuation_variants(client):
    """"3-15-1990" is the same birthday as "03/15/1990"; a caller should not fail over a dash."""
    _book(client)
    resp = client.post("/verify-identity", json={"phone": CALLER, "date_of_birth": "3-15-1990"})
    assert resp.status_code == 200


def test_verify_rejects_wrong_dob(client):
    _book(client)
    resp = client.post("/verify-identity", json={"phone": CALLER, "date_of_birth": WRONG_DOB})
    assert resp.status_code == 403


def test_unknown_number_and_wrong_dob_are_indistinguishable(client):
    """Same status AND same body, or the endpoint becomes a patient-enumeration oracle."""
    _book(client)
    wrong = client.post("/verify-identity", json={"phone": CALLER, "date_of_birth": WRONG_DOB})
    unknown = client.post("/verify-identity", json={"phone": OTHER_CALLER, "date_of_birth": DOB})
    assert wrong.status_code == unknown.status_code == 403
    assert wrong.json() == unknown.json()


# --- PHI reads fail closed ----------------------------------------------------------------


def test_list_appointments_for_verified_caller(client):
    booking = _book(client)
    resp = client.post("/appointments", json={"phone": CALLER, "date_of_birth": DOB})
    assert resp.status_code == 200
    ids = [a["confirmation_id"] for a in resp.json()["appointments"]]
    assert booking["confirmation_id"] in ids


def test_spoofed_ani_without_dob_gets_nothing(client):
    """The headline attack: the attacker has the number, not the birthday."""
    _book(client)
    for dob in (WRONG_DOB, "", "   ", "0000"):
        resp = client.post("/appointments", json={"phone": CALLER, "date_of_birth": dob})
        assert resp.status_code in (403, 422), f"DOB {dob!r} must not disclose appointments"
        assert "confirmation_id" not in resp.text


def test_cancel_rejects_wrong_dob(client):
    booking = _book(client)
    resp = client.post("/cancel", json={
        "phone": CALLER, "date_of_birth": WRONG_DOB,
        "confirmation_id": booking["confirmation_id"],
    })
    assert resp.status_code == 403
    still = client.post("/appointments", json={"phone": CALLER, "date_of_birth": DOB}).json()
    assert len(still["appointments"]) == 1, "the appointment must survive a failed cancel"


def test_cannot_touch_another_patients_booking(client):
    """Verified as yourself is not verified as everyone: a valid caller must not be able to
    cancel a confirmation id belonging to somebody else."""
    victim = _book(client, slot_index=0)
    _book(client, phone=OTHER_CALLER, dob="01/02/1980", name="Sam Okafor", slot_index=1)

    resp = client.post("/cancel", json={
        "phone": OTHER_CALLER, "date_of_birth": "01/02/1980",
        "confirmation_id": victim["confirmation_id"],
    })
    assert resp.status_code == 404
    survivors = client.post("/appointments", json={"phone": CALLER, "date_of_birth": DOB}).json()
    assert [a["confirmation_id"] for a in survivors["appointments"]] == [
        victim["confirmation_id"]
    ]


# --- reschedule / cancel ------------------------------------------------------------------


def test_reschedule_moves_the_appointment_and_frees_the_old_slot(client):
    booking = _book(client)
    open_slots = [
        s["slot_id"] for s in client.get("/availability").json()["slots"]
        if s["slot_id"] != booking["slot_id"]
    ]
    new_slot = open_slots[0]

    resp = client.post("/reschedule", json={
        "phone": CALLER, "date_of_birth": DOB,
        "confirmation_id": booking["confirmation_id"], "new_slot_id": new_slot,
    })
    assert resp.status_code == 200, resp.text
    assert resp.json()["slot_id"] == new_slot

    available = [s["slot_id"] for s in client.get("/availability").json()["slots"]]
    assert booking["slot_id"] in available, "the old slot must go back on the shelf"
    assert new_slot not in available, "the new slot must be taken"

    listed = client.post("/appointments", json={"phone": CALLER, "date_of_birth": DOB}).json()
    assert listed["appointments"][0]["slot_id"] == new_slot


def test_reschedule_is_idempotent(client):
    """A retried voice turn must not fail because the move already happened."""
    booking = _book(client)
    new_slot = [
        s["slot_id"] for s in client.get("/availability").json()["slots"]
        if s["slot_id"] != booking["slot_id"]
    ][0]
    body = {"phone": CALLER, "date_of_birth": DOB,
            "confirmation_id": booking["confirmation_id"], "new_slot_id": new_slot}
    first = client.post("/reschedule", json=body)
    second = client.post("/reschedule", json=body)
    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()


def test_reschedule_into_a_taken_slot_conflicts(client):
    mine = _book(client, slot_index=0)
    theirs = _book(client, phone=OTHER_CALLER, dob="01/02/1980", name="Sam Okafor", slot_index=1)
    resp = client.post("/reschedule", json={
        "phone": CALLER, "date_of_birth": DOB,
        "confirmation_id": mine["confirmation_id"], "new_slot_id": theirs["slot_id"],
    })
    assert resp.status_code == 409


def test_cancel_frees_the_slot_and_is_idempotent(client):
    booking = _book(client)
    body = {"phone": CALLER, "date_of_birth": DOB,
            "confirmation_id": booking["confirmation_id"], "reason": "feeling better"}
    first = client.post("/cancel", json=body)
    second = client.post("/cancel", json=body)
    assert first.status_code == second.status_code == 200
    assert first.json()["status"] == "cancelled"

    available = [s["slot_id"] for s in client.get("/availability").json()["slots"]]
    assert booking["slot_id"] in available
    listed = client.post("/appointments", json={"phone": CALLER, "date_of_birth": DOB}).json()
    assert listed["appointments"] == []


# --- staff tasks + clinic info -------------------------------------------------------------


def test_refill_creates_a_staff_task_and_never_approves(client):
    _book(client)
    resp = client.post("/staff-tasks", json={
        "phone": CALLER, "date_of_birth": DOB, "kind": "refill",
        "payload": {"medication": "the blue inhaler"},
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "open" and body["kind"] == "refill"
    assert "approved" not in resp.text.lower()


def test_refill_requires_verification(client):
    _book(client)
    resp = client.post("/staff-tasks", json={
        "phone": CALLER, "date_of_birth": WRONG_DOB, "kind": "refill", "payload": {},
    })
    assert resp.status_code == 403


def test_clinic_info_returns_a_curated_fact(client):
    body = client.get("/clinic-info", params={"topic": "hours"}).json()
    assert "Monday" in body["content"]
    assert "location" in body["topics"]


def test_unknown_topic_returns_the_topic_list_not_an_invention(client):
    body = client.get("/clinic-info", params={"topic": "wifi password"}).json()
    assert body["content"] is None
    assert body["topics"], "the model needs the real topics so it can pick one"


def test_booking_without_a_phone_creates_no_patient(client):
    """Existing callers (eval cases, the local path) send no ANI. They must still book, and
    must not create an unverifiable patient row."""
    slot_id = client.get("/availability").json()["slots"][0]["slot_id"]
    hold = client.post("/hold-slot", json={"slot_id": slot_id}).json()
    resp = client.post("/confirm-booking", json={
        "hold_id": hold["hold_id"], "patient_name": "Ghost", "reason": "checkup",
    })
    assert resp.status_code == 200
    assert client.get("/caller-memory", params={"phone": CALLER}).json()["known"] is False

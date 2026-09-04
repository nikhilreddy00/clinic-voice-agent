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


# --- name + DOB: the path that exists because ANI-only verification dead-ends ---------------
#
# Found on a live call. Three confirmed bookings existed for the caller, made before the API
# ever recorded a phone number, so no patient row existed for their ANI. The agent asked for a
# date of birth it could not possibly match, failed, and offered a staff member. The same hole
# swallows anything booked at the front desk or on the web, and any caller phoning from a
# different handset.


def _orphan_booking(client, *, name="Dana Reyes", dob=DOB, slot_index=0) -> dict:
    """A confirmed booking with NO phone — i.e. every booking made before Phase 13, and every
    booking a clinic makes through any other channel."""
    slots = client.get("/availability").json()["slots"]
    hold = client.post("/hold-slot", json={"slot_id": slots[slot_index]["slot_id"]}).json()
    resp = client.post("/confirm-booking", json={
        "hold_id": hold["hold_id"], "patient_name": name, "reason": "checkup",
        "date_of_birth": dob,
    })
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_name_and_dob_reach_a_booking_the_ani_cannot(client):
    _orphan_booking(client)
    # The number is unknown, so DOB alone must not be enough...
    assert client.post("/verify-identity", json={
        "phone": CALLER, "date_of_birth": DOB,
    }).status_code == 403
    # ...but the front-desk check gets in.
    resp = client.post("/verify-identity", json={
        "phone": CALLER, "date_of_birth": DOB, "name": "Dana Reyes",
    })
    assert resp.status_code == 200, resp.text
    appts = client.post("/appointments", json={"phone": CALLER, "date_of_birth": DOB})
    assert len(appts.json()["appointments"]) == 1


def test_verifying_by_name_enrolls_the_number_for_next_time(client):
    """The caller should not have to give their name on every future call."""
    _orphan_booking(client)
    assert client.get("/caller-memory", params={"phone": CALLER}).json()["known"] is False

    client.post("/verify-identity", json={
        "phone": CALLER, "date_of_birth": DOB, "name": "Dana Reyes",
    })

    memory = client.get("/caller-memory", params={"phone": CALLER}).json()
    assert memory["known"] is True and memory["upcoming_appointments"] == 1
    # And the fast path now works with no name at all.
    assert client.post("/verify-identity", json={
        "phone": CALLER, "date_of_birth": DOB,
    }).status_code == 200


def test_the_name_is_a_real_factor_not_a_formality(client):
    _orphan_booking(client)
    for bad in ("Sam Okafor", "", None):
        resp = client.post("/verify-identity", json={
            "phone": CALLER, "date_of_birth": DOB, "name": bad,
        })
        assert resp.status_code == 403, f"name {bad!r} must not verify"
    # ...and the right name with the wrong birthday is equally useless.
    assert client.post("/verify-identity", json={
        "phone": CALLER, "date_of_birth": WRONG_DOB, "name": "Dana Reyes",
    }).status_code == 403


def test_transcription_spacing_and_case_do_not_fail_a_caller(client):
    _orphan_booking(client)
    assert client.post("/verify-identity", json={
        "phone": CALLER, "date_of_birth": DOB,
        "name": "  dana   REYES ",
    }).status_code == 200


def test_an_enrolled_number_cannot_be_used_to_verify_as_someone_else(client):
    """The hijack this ordering prevents: whoever holds an enrolled patient's handset must not
    be able to name a different patient and have that person's bookings re-pointed at it."""
    _book(client, phone=CALLER, dob=DOB, name="Dana Reyes", slot_index=0)   # CALLER is enrolled
    victim = _orphan_booking(client, name="Sam Okafor", dob="01/02/1980", slot_index=1)

    resp = client.post("/verify-identity", json={
        "phone": CALLER, "date_of_birth": "01/02/1980", "name": "Sam Okafor",
    })
    assert resp.status_code == 403

    # Sam's booking must still be unattached and unreachable from that phone.
    listed = client.post("/appointments", json={"phone": CALLER, "date_of_birth": DOB}).json()
    assert victim["confirmation_id"] not in [a["confirmation_id"] for a in listed["appointments"]]


def test_a_caller_with_no_ani_can_still_verify(client):
    """A withheld caller ID is a normal thing, not an error."""
    _orphan_booking(client)
    resp = client.post("/verify-identity", json={
        "phone": "", "date_of_birth": DOB, "name": "Dana Reyes",
    })
    assert resp.status_code == 200
    assert client.post("/appointments", json={
        "phone": "", "date_of_birth": DOB, "name": "Dana Reyes",
    }).status_code == 200


# --- a phone number is a household, not a person ------------------------------------------
#
# The live defect these cover (2026-09-03): "Joe" (DOB 03/05/2001) booked from a number already
# enrolled to "Nick" (DOB 05/08/2003), was given confirmation E938C8F6, called back, and was
# refused twice on the exact date of birth he had just booked with. `patients` was
# UNIQUE (clinic_id, phone), so the first caller from a number owned it permanently.
#
# These are the edge cases that must hold WITHOUT placing a phone call.

SECOND_PERSON_DOB = "05/08/2003"


def test_a_second_person_on_a_shared_phone_can_verify_with_their_own_dob(client):
    """The exact live failure. Book as one person, book as another, both must verify."""
    _book(client, name="Nick", dob=DOB)
    _book(client, name="Joe", dob=SECOND_PERSON_DOB, slot_index=1)

    for name, dob in [("Nick", DOB), ("Joe", SECOND_PERSON_DOB)]:
        resp = client.post("/verify-identity",
                           json={"phone": CALLER, "date_of_birth": dob, "name": name})
        assert resp.status_code == 200, f"{name} could not verify: {resp.text}"
        assert resp.json()["name"] == name


def test_a_second_booking_does_not_overwrite_the_first_persons_name(client):
    """The row used to be a MERGE of two people — named Joe, carrying Nick's DOB.

    That is worse than the lockout it caused: verifying with Nick's date of birth would have
    returned Joe's name, and Nick's appointments with it.
    """
    _book(client, name="Nick", dob=DOB)
    _book(client, name="Joe", dob=SECOND_PERSON_DOB, slot_index=1)

    body = client.post("/verify-identity",
                       json={"phone": CALLER, "date_of_birth": DOB, "name": "Nick"}).json()
    assert body["name"] == "Nick"


def test_each_person_on_a_shared_phone_sees_only_their_own_appointments(client):
    """The disclosure the merged row would have caused, asserted directly."""
    nick = _book(client, name="Nick", dob=DOB)
    joe = _book(client, name="Joe", dob=SECOND_PERSON_DOB, slot_index=1)

    nicks = client.post("/appointments",
                        json={"phone": CALLER, "date_of_birth": DOB, "name": "Nick"}).json()
    ids = {a["confirmation_id"] for a in nicks["appointments"]}
    assert nick["confirmation_id"] in ids
    assert joe["confirmation_id"] not in ids, "Nick can see Joe's appointment"

    joes = client.post("/appointments",
                       json={"phone": CALLER, "date_of_birth": SECOND_PERSON_DOB,
                             "name": "Joe"}).json()
    ids = {a["confirmation_id"] for a in joes["appointments"]}
    assert joe["confirmation_id"] in ids
    assert nick["confirmation_id"] not in ids, "Joe can see Nick's appointment"


def test_a_third_dob_on_an_enrolled_phone_is_still_refused(client):
    """Widening the key must not widen who gets in. This is the anti-hijack rule."""
    _book(client, name="Nick", dob=DOB)
    _book(client, name="Joe", dob=SECOND_PERSON_DOB, slot_index=1)

    resp = client.post("/verify-identity",
                       json={"phone": CALLER, "date_of_birth": "01/01/1970", "name": "Nick"})
    assert resp.status_code == 403


def test_a_shared_phone_cannot_be_used_to_reach_a_stranger_by_name(client):
    """The reason _verify stops dead instead of falling through to the name+DOB search.

    Someone holding an enrolled handset must not be able to name a patient enrolled elsewhere
    and be handed their chart — even though that name and DOB would match on their own.
    """
    _book(client, name="Nick", dob=DOB)
    stranger = _book(client, phone=OTHER_CALLER, name="Priya Raman",
                     dob="11/22/1985", slot_index=1)

    resp = client.post("/verify-identity",
                       json={"phone": CALLER, "date_of_birth": "11/22/1985",
                             "name": "Priya Raman"})
    assert resp.status_code == 403, "an enrolled handset reached a stranger's record"

    appts = client.post("/appointments",
                        json={"phone": CALLER, "date_of_birth": "11/22/1985",
                              "name": "Priya Raman"})
    assert appts.status_code == 403
    assert stranger["confirmation_id"] not in appts.text


def test_the_same_birthday_spoken_differently_is_still_one_person(client):
    """The DOB is part of the key now, so a key made of raw speech would file one person twice.

    "3/5/2001" and "03/05/2001" are the same birthday; the agent's transcription of a spoken
    date is not stable enough to be a primary key without normalization.
    """
    _book(client, name="Joe", dob="03/05/2001")
    _book(client, name="Joe", dob="3/5/2001", slot_index=1)

    body = client.post("/appointments",
                       json={"phone": CALLER, "date_of_birth": "3-5-2001",
                             "name": "Joe"}).json()
    assert len(body["appointments"]) == 2, "one person was filed as two"


def test_a_second_person_can_reschedule_their_own_appointment(client):
    """End to end: the flow the live call was actually trying to complete."""
    _book(client, name="Nick", dob=DOB)
    joe = _book(client, name="Joe", dob=SECOND_PERSON_DOB, slot_index=1)

    slots = client.get("/availability").json()["slots"]
    target = next(s["slot_id"] for s in slots if s["slot_id"] != joe["slot_id"])

    resp = client.post("/reschedule", json={
        "confirmation_id": joe["confirmation_id"], "new_slot_id": target,
        "phone": CALLER, "date_of_birth": SECOND_PERSON_DOB, "name": "Joe",
    })
    assert resp.status_code == 200, resp.text
    assert resp.json()["slot_id"] == target


def test_a_second_person_cannot_cancel_the_first_persons_appointment(client):
    nick = _book(client, name="Nick", dob=DOB)
    _book(client, name="Joe", dob=SECOND_PERSON_DOB, slot_index=1)

    resp = client.post("/cancel", json={
        "confirmation_id": nick["confirmation_id"],
        "phone": CALLER, "date_of_birth": SECOND_PERSON_DOB, "name": "Joe",
    })
    assert resp.status_code == 404, "Joe cancelled Nick's appointment"


def test_caller_memory_counts_the_whole_household_not_one_member(client):
    """It used to GROUP BY patient id and take the first row — one arbitrary person's count.

    Measured: three confirmed appointments on a number, reported as two, and which two depended
    on row order. The number is the right unit here because this runs BEFORE verification —
    nobody has said who they are yet — and it still returns nothing identifying.
    """
    _book(client, name="Nick", dob=DOB, slot_index=0)
    _book(client, name="Nick", dob=DOB, slot_index=1)
    _book(client, name="Joe", dob=SECOND_PERSON_DOB, slot_index=2)

    for _ in range(5):  # the old bug was non-deterministic; one pass could get lucky
        body = client.get("/caller-memory", params={"phone": CALLER}).json()
        assert body == {"known": True, "upcoming_appointments": 3}


def test_caller_memory_still_discloses_nothing_identifying(client):
    """The count is a household fact; a name would be a disclosure to whoever holds the phone."""
    _book(client, name="Nick", dob=DOB)
    _book(client, name="Joe", dob=SECOND_PERSON_DOB, slot_index=1)

    body = client.get("/caller-memory", params={"phone": CALLER}).json()
    assert set(body) == {"known", "upcoming_appointments"}
    assert "Nick" not in str(body) and "Joe" not in str(body)


def test_two_people_sharing_a_phone_AND_a_birthday_collapse_into_one_record(client):
    """A KNOWN, DELIBERATE limitation — pinned here so it stays a decision, not an accident.

    `patients` is keyed (clinic, phone, date_of_birth). Two people on one handset who share a
    birthday — twins — therefore land on one row: the later booking renames it, and verifying
    as either returns that name and BOTH sets of appointments.

    Adding `name` to the key would fix twins and break something commoner: a caller who says
    "Nick" on one call and "Nicholas Kumar" on the next is one person, and a name-keyed record
    would file them as two, locking each out of the other's bookings. Spoken names are not
    stable enough to key on; birthdays are. The trade is deliberate.

    If twins ever need supporting, the fix is a real patient identifier the caller states — not
    a fuzzier name match, which weakens a credential to solve a data-modelling problem.
    """
    alex = _book(client, name="Alex", dob="07/07/2000", slot_index=0)
    sam = _book(client, name="Sam", dob="07/07/2000", slot_index=1)

    body = client.post("/verify-identity",
                       json={"phone": CALLER, "date_of_birth": "07/07/2000",
                             "name": "Alex"}).json()
    assert body["name"] == "Sam"  # the later booking's name won

    appts = client.post("/appointments",
                        json={"phone": CALLER, "date_of_birth": "07/07/2000",
                              "name": "Alex"}).json()
    ids = {a["confirmation_id"] for a in appts["appointments"]}
    assert ids == {alex["confirmation_id"], sam["confirmation_id"]}


def test_the_same_person_giving_a_fuller_name_stays_one_record(client):
    """The case the twins trade-off protects: "Nick" and "Nicholas Kumar" are one patient."""
    first = _book(client, name="Nick", dob=DOB, slot_index=0)
    second = _book(client, name="Nicholas Kumar", dob=DOB, slot_index=1)

    appts = client.post("/appointments",
                        json={"phone": CALLER, "date_of_birth": DOB}).json()
    ids = {a["confirmation_id"] for a in appts["appointments"]}
    assert ids == {first["confirmation_id"], second["confirmation_id"]}


def test_a_parent_can_book_for_a_child_from_the_same_phone(client):
    """The commonest real shared-handset case, and it must work without verification."""
    parent = _book(client, name="Joe", dob=SECOND_PERSON_DOB, slot_index=0)
    child = _book(client, name="Sarah", dob="01/01/2015", slot_index=1)

    for name, dob, mine, theirs in [
        ("Joe", SECOND_PERSON_DOB, parent, child),
        ("Sarah", "01/01/2015", child, parent),
    ]:
        body = client.post("/verify-identity",
                           json={"phone": CALLER, "date_of_birth": dob, "name": name}).json()
        assert body["name"] == name
        appts = client.post("/appointments",
                            json={"phone": CALLER, "date_of_birth": dob, "name": name}).json()
        ids = {a["confirmation_id"] for a in appts["appointments"]}
        assert mine["confirmation_id"] in ids
        assert theirs["confirmation_id"] not in ids

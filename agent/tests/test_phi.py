"""Phase 17 — the PHI boundary, enforced by running it rather than by reading it.

The defect this exists to prevent already happened twice: `confirm_booking` logged
`patient_name` verbatim on both its request and its result line, in a file where every other
sensitive field was carefully reduced to `set`/`unset`/`len=`. Nothing failed, because nothing
was checking.

So the check is a live one. Every tool is driven through the REAL `execute_tool` with sentinel
values no other part of the system could produce, and loguru's output is searched for them. A
source grep would pass on `f"...{name}..."`; this does not care how the string was built, only
whether the words came out.

`test_phi_sentinels_are_absent_from_every_tool_log` is the line `scripts/prove_ci_gate.sh`
plants a regression against.
"""

from __future__ import annotations

import pytest
from loguru import logger

from clinic_agent import phi
from clinic_agent.scheduling_tools import TOOL_ENDPOINTS, execute_tool

# Values that appear nowhere else in the project. If one of these reaches a log line, a real
# caller's equivalent would have too.
NAME = "Zebediah Quillfeather"
DOB = "01/02/1903"
PHONE = "+15558675309"
NOTES = "sharp pain in the left elbow since Tuesday, worse when lifting"
MEDICATION = "quillfeatherazepam"

SENTINELS = [NAME, DOB, NOTES, MEDICATION, "8675309"]

# One representative argument set per tool, PHI-maximal on purpose.
ARGUMENTS = {
    "check_availability": {"date": "2026-09-10", "reason_category": "checkup"},
    "hold_slot": {"slot_id": 7},
    "confirm_booking": {
        "hold_id": "h-1", "patient_name": NAME, "reason": "checkup",
        "date_of_birth": DOB, "new_patient": True, "symptom_notes": NOTES, "phone": PHONE,
    },
    "verify_identity": {"phone": PHONE, "date_of_birth": DOB, "name": NAME},
    "list_appointments": {"phone": PHONE, "date_of_birth": DOB, "name": NAME},
    "reschedule_appointment": {
        "confirmation_id": "ABC123", "new_slot_id": 9,
        "phone": PHONE, "date_of_birth": DOB, "name": NAME,
    },
    "cancel_appointment": {
        "confirmation_id": "ABC123", "phone": PHONE, "date_of_birth": DOB,
        "reason": "feeling better", "name": NAME,
    },
    "request_refill": {
        "phone": PHONE, "date_of_birth": DOB, "medication": MEDICATION,
        "notes": NOTES, "name": NAME,
    },
    "get_clinic_info": {"topic": "hours"},
}

# What each tool's client method returns — shaped like the real API responses, and echoing the
# PHI back the way the real ones do. The result lines are half of what is under test.
RESULTS = {
    "check_availability": {
        "ok": True, "count": 1,
        "slots": [{"slot_id": 7, "display_time": "Thursday at 9:00 AM",
                   "provider_name": "Dr. Rivera"}],
    },
    "hold_slot": {"ok": True, "slot_id": 7, "hold_id": "h-1", "expires_at": "2026-09-10T13:02Z"},
    "confirm_booking": {
        "ok": True, "confirmation_id": "ABC123", "display_time": "Thursday at 9:00 AM",
        "provider_name": "Dr. Rivera", "patient_name": NAME, "date_of_birth": DOB,
    },
    "verify_identity": {"ok": True, "verified": True, "name": NAME},
    "list_appointments": {
        "ok": True, "count": 1,
        "appointments": [{"confirmation_id": "ABC123", "patient_name": NAME}],
    },
    "reschedule_appointment": {
        "ok": True, "confirmation_id": "ABC123", "display_time": "Friday at 1:00 PM",
        "provider_name": "Dr. Rivera", "patient_name": NAME,
    },
    "cancel_appointment": {"ok": True, "confirmation_id": "ABC123", "patient_name": NAME},
    "request_refill": {"ok": True, "task_id": 4, "medication": MEDICATION},
    "get_clinic_info": {"ok": True, "topic": "hours", "content": "8 to 5, Monday to Friday"},
}

_METHOD = {
    "check_availability": "get_availability",
    "hold_slot": "hold_slot",
    "confirm_booking": "confirm_booking",
    "verify_identity": "verify_identity",
    "list_appointments": "list_appointments",
    "reschedule_appointment": "reschedule_appointment",
    "cancel_appointment": "cancel_appointment",
    "request_refill": "request_refill",
    "get_clinic_info": "clinic_info",
}


class ScriptedClient:
    """Every SchedulingClient method the tools use, returning the canned result. No HTTP."""

    def __init__(self, result: dict) -> None:
        self._result = result

    def __getattr__(self, _name):
        async def call(**_kwargs):
            return self._result
        return call


@pytest.fixture
def captured():
    lines: list[str] = []
    sink_id = logger.add(lines.append, level="DEBUG", format="{message}")
    try:
        yield lines
    finally:
        logger.remove(sink_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("tool", sorted(TOOL_ENDPOINTS))
async def test_phi_sentinels_are_absent_from_every_tool_log(tool, captured):
    """The gate. Every tool, request line and result line, no sentinel anywhere."""
    await execute_tool(ScriptedClient(RESULTS[tool]), tool, dict(ARGUMENTS[tool]))
    blob = "\n".join(captured)
    assert blob, f"{tool} logged nothing — the test would pass vacuously"
    for sentinel in SENTINELS:
        assert sentinel not in blob, f"{tool} leaked {sentinel!r}:\n{blob}"


@pytest.mark.asyncio
async def test_the_log_lines_still_say_something_useful(captured):
    """Redaction that erases the line's purpose gets removed by the next person to debug a call.

    So assert the useful half survives: the endpoint, the presence of a DOB, and a length for
    the clinical note.
    """
    await execute_tool(
        ScriptedClient(RESULTS["confirm_booking"]), "confirm_booking",
        dict(ARGUMENTS["confirm_booking"]),
    )
    blob = "\n".join(captured)
    assert "/confirm-booking" in blob
    assert "dob=set" in blob
    assert f"symptom_notes=len={len(NOTES)}" in blob
    assert "confirmation_id=ABC123" in blob


def test_every_phi_field_has_a_redaction_that_drops_the_value():
    for field in phi.PHI_FIELDS:
        rendered = phi.redact(field, "Quillfeather")
        assert "Quillfeather" not in rendered, field


def test_a_field_outside_the_set_passes_through():
    """Deny-by-default is over a NAMED set — a new PHI argument must be added to it. This test
    pins that fact so the behaviour is a decision rather than a surprise."""
    assert phi.redact("reason_category", "checkup") == "'checkup'"


def test_phone_keeps_four_digits_and_nothing_else():
    assert phi.redact("phone", PHONE) == "***5309"
    assert phi.redact("phone", None) == "unset"


def test_transcript_retention_switch(monkeypatch):
    monkeypatch.delenv("CLINIC_PHI_LOGS", raising=False)
    assert phi.retain_transcripts() is True
    assert phi.speech("hello there") == "'hello there'"

    monkeypatch.setenv("CLINIC_PHI_LOGS", "0")
    assert phi.retain_transcripts() is False
    assert "hello" not in phi.speech("hello there")
    assert phi.speech("hello there") == "«len=11»"


def test_scrubbing_a_trace_record_keeps_structure_and_drops_words(monkeypatch):
    record = {"type": "FinalTranscript", "seq": 4, "t": 1.5, "text": "my name is Zebediah",
              "confidence": 0.98}

    monkeypatch.setenv("CLINIC_PHI_LOGS", "1")
    assert phi.scrub_event(record) == record  # default path: traces are untouched

    monkeypatch.setenv("CLINIC_PHI_LOGS", "0")
    scrubbed = phi.scrub_event(record)
    assert "Zebediah" not in scrubbed["text"]
    assert scrubbed["seq"] == 4 and scrubbed["t"] == 1.5 and scrubbed["confidence"] == 0.98

    tool_use = {"type": "LLMToolUse", "name": "confirm_booking",
                "arguments": {"patient_name": NAME, "reason": "checkup"}}
    scrubbed = phi.scrub_event(tool_use)
    assert scrubbed["arguments"]["patient_name"] == "set"
    assert scrubbed["arguments"]["reason"] == "checkup"

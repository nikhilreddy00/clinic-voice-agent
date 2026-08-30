"""Pydantic request/response schemas for the scheduling API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SlotOut(BaseModel):
    slot_id: int
    provider_id: int
    provider_name: str
    specialty: str
    start_time: str  # ISO 8601 UTC
    reason_category: str


class AvailabilityResponse(BaseModel):
    slots: list[SlotOut]


class HoldRequest(BaseModel):
    slot_id: int = Field(..., description="ID of the slot to place a temporary hold on")


class HoldResponse(BaseModel):
    hold_id: str
    slot_id: int
    expires_at: str  # ISO 8601 UTC


class ConfirmRequest(BaseModel):
    hold_id: str
    patient_name: str = Field(..., min_length=1, description="Synthetic patient name")
    reason: str = Field(..., min_length=1, description="Coarse reason for the visit")
    # Richer intake fields (added with the extended booking flow). All OPTIONAL so older
    # callers/eval cases that only send name+reason still book; the agent's prompt collects
    # them on the normal path. Synthetic data only — no real PHI.
    date_of_birth: str | None = Field(
        None, description="Caller's date of birth, normalized to MM/DD/YYYY"
    )
    new_patient: bool | None = Field(
        None, description="True if a new patient, False if an existing/returning one"
    )
    symptom_notes: str | None = Field(
        None, description="One-sentence description of the caller's symptoms"
    )
    # Phase 13. Taken from the call's ANI by the agent, never asked of the caller. Booking with
    # a phone + DOB is what creates the patient record that makes the NEXT call a returning one.
    phone: str | None = Field(None, description="Caller's number in E.164, from the SIP ANI")


class ConfirmResponse(BaseModel):
    confirmation_id: str
    slot_id: int
    provider_name: str
    start_time: str
    patient_name: str
    date_of_birth: str | None = None
    new_patient: bool | None = None
    symptom_notes: str | None = None


# --- Phase 13: verified caller flows -----------------------------------------------------
#
# Every model below carries `phone` and `date_of_birth` together, on every request, and the
# database re-checks the pair each time (see db._verify). There is no session, no bearer of
# "already verified" state on the wire, and no verification token to steal or replay: the ANI
# alone is worthless because caller ID is spoofable, and the DOB alone is worthless because it
# is not tied to a number. That is the whole design.


class VerifiedRequest(BaseModel):
    """Base for anything that touches PHI."""

    phone: str = Field(..., min_length=1, description="Caller's number in E.164, from the ANI")
    date_of_birth: str = Field(..., min_length=1, description="Spoken DOB, normalized MM/DD/YYYY")


class VerifyResponse(BaseModel):
    patient_id: int
    name: str | None = None


class AppointmentOut(BaseModel):
    confirmation_id: str
    slot_id: int
    provider_name: str
    start_time: str
    reason: str


class AppointmentsResponse(BaseModel):
    appointments: list[AppointmentOut]


class RescheduleRequest(VerifiedRequest):
    confirmation_id: str
    new_slot_id: int


class RescheduleResponse(BaseModel):
    confirmation_id: str
    slot_id: int
    provider_name: str
    start_time: str


class CancelRequest(VerifiedRequest):
    confirmation_id: str
    reason: str | None = None


class CancelResponse(BaseModel):
    confirmation_id: str
    status: str


class StaffTaskRequest(VerifiedRequest):
    kind: str = Field(..., description="e.g. 'refill'")
    payload: dict = Field(default_factory=dict)


class StaffTaskResponse(BaseModel):
    task_id: int
    kind: str
    status: str


class CallerMemoryResponse(BaseModel):
    """Pre-verification. Carries no identity — see db.caller_memory for why."""

    known: bool
    upcoming_appointments: int


class ClinicInfoResponse(BaseModel):
    topic: str | None = None
    content: str | None = None
    topics: list[str]

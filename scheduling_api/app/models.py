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


class ConfirmResponse(BaseModel):
    confirmation_id: str
    slot_id: int
    provider_name: str
    start_time: str
    patient_name: str
    date_of_birth: str | None = None
    new_patient: bool | None = None
    symptom_notes: str | None = None

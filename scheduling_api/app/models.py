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


class ConfirmResponse(BaseModel):
    confirmation_id: str
    slot_id: int
    provider_name: str
    start_time: str
    patient_name: str

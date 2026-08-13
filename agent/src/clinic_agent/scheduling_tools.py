"""Phase 2 — scheduling-API tool calls for the LLM.

This wires the three mock scheduling-API endpoints (see `scheduling_api/`) into the Pipecat
LLM as callable functions. The LLM decides *when* to call them; this module is only the
HTTP plumbing + Pipecat function-handler glue + stage logging.

Endpoint ↔ function mapping (docs/build_spec.md "State ↔ scheduling API mapping"):

    check_availability  -> GET  /availability     (OFFER_SLOTS)
    hold_slot           -> POST /hold-slot         (CONFIRM_SLOT, on entry)
    confirm_booking     -> POST /confirm-booking   (BOOK)

There is deliberately **no release endpoint**: a rejected/abandoned hold simply expires on
its server-side TTL, which is exactly build_spec's "release = let the hold TTL expire". So a
caller changing their mind needs no extra call — the agent just re-checks availability.

Every call logs a `TOOL ▶` line (request + response summary) alongside the existing
`ASR ▶ / LLM ▶ / TTS ▶` stage logs, so the booking logic is visible in the terminal.

All results handed back to the LLM are plain dicts with an `ok` flag; on any HTTP/transport
error we return `{"ok": False, "error": ...}` rather than raising, so the model can apologize
and recover (re-offer / escalate) instead of the turn crashing.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime
from typing import TYPE_CHECKING

import httpx
from loguru import logger

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.services.llm_service import FunctionCallParams, LLMService

if TYPE_CHECKING:
    from .metrics import LatencyCollector

# Coarse, non-clinical reason categories a slot can be tagged with. Mirrors
# scheduling_api/app/seed_data.py REASON_CATEGORIES — the API filters on these exact values.
REASON_CATEGORIES = ["checkup", "follow-up", "sick-visit", "vaccination"]


def _format_slot_time(iso_start: str) -> str:
    """Turn an ISO-8601 UTC start time into speech-friendly text.

    e.g. "2026-07-06T09:00:00+00:00" -> "Monday, July 6 at 9:00 AM". The LLM speaks this
    string directly, so callers never hear a raw timestamp. (`%-d`/`%-I` strip the leading
    zero and are supported on macOS/Linux, the dev + CI targets.)
    """
    try:
        dt = datetime.fromisoformat(iso_start)
    except ValueError:
        return iso_start  # never seen in practice; degrade to the raw value rather than crash
    return dt.strftime("%A, %B %-d at %-I:%M %p")


def _shape_slot(slot: dict) -> dict:
    """Project an API slot down to what the LLM needs, adding a spoken time string."""
    return {
        "slot_id": slot["slot_id"],
        "provider_name": slot["provider_name"],
        "specialty": slot["specialty"],
        "reason_category": slot["reason_category"],
        "start_time": slot["start_time"],
        "display_time": _format_slot_time(slot["start_time"]),
    }


class SchedulingClient:
    """Thin async HTTP client over the mock scheduling API.

    Owns a single `httpx.AsyncClient` for the life of the call; create it once in the pipeline
    entrypoint and `await aclose()` in a finally. Each method returns an LLM-friendly dict and
    never raises for HTTP/transport failures — it maps them to `{"ok": False, "error": ...}`.
    """

    def __init__(self, base_url: str, *, timeout: float = 10.0, call_id: str | None = None,
                 api_token: str | None = None) -> None:
        headers = {}
        if api_token:
            headers["Authorization"] = f"Bearer {api_token}"
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"), timeout=timeout, headers=headers
        )
        # Scopes idempotency keys to this call, so two callers booking at the same moment never
        # collide on a key. Falls back to a random id when the pipeline doesn't supply one.
        self._call_id = call_id or uuid.uuid4().hex
        self._attempt = 0

    def _idempotency_key(self, operation: str, discriminator: str) -> str:
        """A key that is STABLE across retries of the same logical operation.

        This is the whole point: the agent's HTTP client sits inside a voice turn with a timeout,
        so a write can commit and still look like a failure. Retrying without a key books a
        second appointment. The key must therefore derive from *what is being done* (call +
        operation + target), never from a counter or a fresh uuid per attempt -- those change on
        the retry and defeat the mechanism entirely.
        """
        return f"{self._call_id}:{operation}:{discriminator}"

    async def aclose(self) -> None:
        await self._client.aclose()

    async def get_availability(
        self,
        *,
        date: str | None = None,
        reason: str | None = None,
        provider_id: int | None = None,
    ) -> dict:
        params: dict = {}
        if date:
            params["date"] = date
        if reason:
            params["reason"] = reason
        if provider_id is not None:
            params["provider_id"] = provider_id
        try:
            resp = await self._client.get("/availability", params=params)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            return {"ok": False, "error": f"could not reach the scheduling system ({exc})"}
        slots = [_shape_slot(s) for s in resp.json()["slots"]]
        return {"ok": True, "count": len(slots), "slots": slots}

    async def hold_slot(self, *, slot_id: int) -> dict:
        try:
            resp = await self._client.post(
                "/hold-slot",
                json={"slot_id": slot_id},
                headers={"Idempotency-Key": self._idempotency_key("hold", str(slot_id))},
            )
        except httpx.HTTPError as exc:
            return {"ok": False, "error": f"could not reach the scheduling system ({exc})"}
        if resp.status_code == 409:
            return {"ok": False, "status": 409, "error": "that slot was just taken — offer another"}
        if resp.status_code == 404:
            return {"ok": False, "status": 404, "error": f"slot {slot_id} does not exist"}
        try:
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            return {"ok": False, "error": f"the scheduling system returned an error ({exc})"}
        return {"ok": True, **resp.json()}

    async def confirm_booking(
        self,
        *,
        hold_id: str,
        patient_name: str,
        reason: str,
        date_of_birth: str | None = None,
        new_patient: bool | None = None,
        symptom_notes: str | None = None,
    ) -> dict:
        payload = {
            "hold_id": hold_id,
            "patient_name": patient_name,
            "reason": reason,
            "date_of_birth": date_of_birth,
            "new_patient": new_patient,
            "symptom_notes": symptom_notes,
        }
        try:
            resp = await self._client.post(
                "/confirm-booking",
                json=payload,
                # Keyed on the hold: confirming a given hold is the operation, and a retry of
                # that same confirmation must replay the original confirmation number.
                headers={"Idempotency-Key": self._idempotency_key("confirm", hold_id)},
            )
        except httpx.HTTPError as exc:
            return {"ok": False, "error": f"could not reach the scheduling system ({exc})"}
        if resp.status_code == 409:
            return {
                "ok": False,
                "status": 409,
                "error": "the hold expired or the slot was taken — re-offer available slots",
            }
        try:
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            return {"ok": False, "error": f"the scheduling system returned an error ({exc})"}
        data = resp.json()
        data["display_time"] = _format_slot_time(data["start_time"])
        return {"ok": True, **data}


# --- Pipecat function handlers ----------------------------------------------------------
# Each handler unpacks params.arguments, calls the client, logs a TOOL ▶ line, and hands the
# result back to the LLM via params.result_callback(...). Pipecat appends that result to the
# LLMContext as a tool message and re-runs the LLM automatically, so the model "sees" the real
# availability/hold/booking data and speaks from it — no manual context editing here.


def _tool_http_status(result: dict) -> int | None:
    """Best-effort HTTP status for a tool result (for the metrics sink).

    The client returns LLM-friendly dicts, not responses: success dicts carry no status (the call
    was 2xx), error dicts set `status` only for the 404/409 cases we special-case. So map ok→200,
    a set `status`→itself, and an unmapped failure (transport error)→None.
    """
    if result.get("ok"):
        return 200
    return result.get("status")


# Tool name -> the scheduling-API endpoint it maps to. Used for the metrics sink and the
# TOOL ▶ log lines, and shared by both execution paths (see execute_tool).
TOOL_ENDPOINTS = {
    "check_availability": "/availability",
    "hold_slot": "/hold-slot",
    "confirm_booking": "/confirm-booking",
}


async def execute_tool(
    client: SchedulingClient,
    name: str,
    arguments: dict,
    collector: "LatencyCollector | None" = None,
) -> tuple[dict, float, int | None]:
    """Run one scheduling tool call: HTTP + `TOOL ▶` logging + metrics.

    This is the single implementation shared by the Pipecat pipeline (via the function handlers
    registered below) and the Phase-10 in-house event loop (via
    ``core.adapters.tools.ToolExecutor``). Both paths must behave identically — same requests,
    same PHI-minimized log lines, same metrics — so there is exactly one copy of this logic
    rather than two that quietly diverge.

    Returns ``(result, latency_ms, http_status)``. Never raises for an API failure: the client
    maps those to ``{"ok": False, "error": ...}`` so the model can apologize and recover.
    """
    args = arguments or {}
    t0 = time.monotonic()

    if name == "check_availability":
        date = args.get("date")
        reason = args.get("reason_category")
        provider_id = args.get("provider_id")
        logger.info(
            f"TOOL ▶ GET /availability req={{date={date!r}, reason={reason!r}, "
            f"provider_id={provider_id!r}}}"
        )
        result = await client.get_availability(date=date, reason=reason, provider_id=provider_id)
    elif name == "hold_slot":
        slot_id = args.get("slot_id")
        logger.info(f"TOOL ▶ POST /hold-slot req={{slot_id={slot_id!r}}}")
        result = await client.hold_slot(slot_id=slot_id)
    elif name == "confirm_booking":
        # PHI minimization (governance: no patient data in logs). DOB and the free-text symptom
        # note are the most sensitive fields, so log only presence/length indicators, never the
        # values themselves. new_patient is a non-identifying boolean and is safe to log.
        symptom_notes = args.get("symptom_notes")
        logger.info(
            f"TOOL ▶ POST /confirm-booking req={{hold_id={args.get('hold_id')!r}, "
            f"patient_name={args.get('patient_name')!r}, reason={args.get('reason')!r}, "
            f"dob={'set' if args.get('date_of_birth') else 'unset'}, "
            f"new_patient={args.get('new_patient')!r}, "
            f"symptom_notes_len={len(symptom_notes) if symptom_notes else 0}}}"
        )
        result = await client.confirm_booking(
            hold_id=args.get("hold_id"),
            patient_name=args.get("patient_name"),
            reason=args.get("reason"),
            date_of_birth=args.get("date_of_birth"),
            new_patient=args.get("new_patient"),
            symptom_notes=symptom_notes,
        )
    else:
        logger.warning(f"TOOL ▶ unknown tool {name!r} requested by the model")
        return {"ok": False, "error": f"unknown tool {name}"}, 0.0, None

    latency_ms = (time.monotonic() - t0) * 1000
    http_status = _tool_http_status(result)
    endpoint = TOOL_ENDPOINTS[name]
    if collector:
        collector.record_tool(endpoint, http_status, latency_ms, bool(result.get("ok")))

    _log_tool_result(name, endpoint, result, collector)
    return result, latency_ms, http_status


def _log_tool_result(
    name: str, endpoint: str, result: dict, collector: "LatencyCollector | None"
) -> None:
    """Human-readable `TOOL ▶` result line, plus the empty-availability escalation mark."""
    if not result.get("ok"):
        logger.warning(f"TOOL ▶ {endpoint} FAILED → {result.get('error')!r}")
        return

    if name == "check_availability":
        if result["count"] == 0:
            # This is the escalation trigger. There is no live human-transfer path in this
            # build (warm transfer is Phase 15), so "escalate" means the agent speaks an
            # apology + hand-off line only. Logged explicitly so it's obvious under test.
            logger.warning(
                "TOOL ▶ GET /availability → 0 slots (no availability) — agent will ESCALATE: "
                "spoken hand-off message only, no human transfer in this build"
            )
            if collector:
                collector.mark_escalation()  # overridden by a later successful confirm_booking
            return
        preview = ", ".join(
            f"#{s['slot_id']} {s['display_time']} ({s['provider_name']})"
            for s in result["slots"][:4]
        )
        more = "" if result["count"] <= 4 else f" (+{result['count'] - 4} more)"
        logger.info(f"TOOL ▶ GET /availability → {result['count']} slots: {preview}{more}")
    elif name == "hold_slot":
        logger.info(
            f"TOOL ▶ POST /hold-slot → held slot {result['slot_id']} "
            f"hold_id={result['hold_id']} expires_at={result['expires_at']}"
        )
    elif name == "confirm_booking":
        logger.info(
            f"TOOL ▶ POST /confirm-booking → BOOKED confirmation_id={result['confirmation_id']} "
            f"{result['display_time']} with {result['provider_name']} for {result['patient_name']}"
        )


def register_scheduling_functions(
    llm: LLMService,
    client: SchedulingClient,
    collector: "LatencyCollector | None" = None,
) -> None:
    """Register the three scheduling functions on the LLM service (Pipecat path).

    `collector` (Phase 6, optional) receives a `tool` metrics event per call — endpoint, HTTP
    status, latency, success — and an escalation mark when availability comes back empty.
    """

    def _handler(name: str):
        async def handle(params: FunctionCallParams) -> None:
            result, _latency_ms, _status = await execute_tool(
                client, name, params.arguments or {}, collector
            )
            await params.result_callback(result)

        return handle

    for tool_name in TOOL_ENDPOINTS:
        llm.register_function(tool_name, _handler(tool_name))


def build_tools_schema() -> ToolsSchema:
    """The tool/function schema the LLM is allowed to call (Phase 2)."""
    reasons_list = "/".join(REASON_CATEGORIES)
    return ToolsSchema(
        standard_tools=[
            FunctionSchema(
                name="check_availability",
                description=(
                    "Look up open appointment slots. Call this once you know the caller's "
                    "preferred day (and, ideally, the reason for the visit). Returns a list of "
                    "slots, each with a slot_id and a spoken display_time. Offer the caller the "
                    "exact match if present, otherwise the 1-2 nearest alternatives from the "
                    "returned list. If the list is empty there is no availability — apologize "
                    "and offer to hand the caller to staff."
                ),
                properties={
                    "date": {
                        "type": "string",
                        "description": (
                            "Preferred day as YYYY-MM-DD (UTC). Resolve relative phrases like "
                            "'tomorrow' or 'next Tuesday' against today's date given in the "
                            "system prompt. Omit to see the soonest slots across all days."
                        ),
                    },
                    "reason_category": {
                        "type": "string",
                        "enum": REASON_CATEGORIES,
                        "description": (
                            f"Coarse visit category ({reasons_list}) mapped from the caller's "
                            "stated reason. Omit if it doesn't map cleanly."
                        ),
                    },
                    "provider_id": {
                        "type": "integer",
                        "description": "Filter to a specific provider id, only if the caller asks for one.",
                    },
                },
                required=[],
            ),
            FunctionSchema(
                name="hold_slot",
                description=(
                    "Place a short-lived hold on the slot the caller chose, BEFORE reading it "
                    "back for final confirmation, so it isn't lost while confirming. Use a "
                    "slot_id from a check_availability result. Returns a hold_id needed to book. "
                    "If it fails (slot just taken), apologize and offer another slot."
                ),
                properties={
                    "slot_id": {
                        "type": "integer",
                        "description": "The slot_id of the caller's chosen slot, from check_availability.",
                    }
                },
                required=["slot_id"],
            ),
            FunctionSchema(
                name="confirm_booking",
                description=(
                    "Commit the booking. Call ONLY after the caller has explicitly said yes to "
                    "the read-back of the held slot. Uses the hold_id from hold_slot. Include the "
                    "intake details (date_of_birth, new_patient, symptom_notes) collected earlier "
                    "in the call. Returns a confirmation_id to read back to the caller. If it "
                    "fails (hold expired), apologize and re-offer available slots."
                ),
                properties={
                    "hold_id": {
                        "type": "string",
                        "description": "The hold_id returned by hold_slot.",
                    },
                    "patient_name": {
                        "type": "string",
                        "description": "The caller's name, as collected earlier in the call.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "The coarse reason for the visit, as collected earlier.",
                    },
                    "date_of_birth": {
                        "type": "string",
                        "description": (
                            "The caller's date of birth, normalized to MM/DD/YYYY from whatever "
                            "spoken form they gave (e.g. 'March 15th 1990' -> '03/15/1990')."
                        ),
                    },
                    "new_patient": {
                        "type": "boolean",
                        "description": (
                            "True if the caller is a new patient, False if they've visited before."
                        ),
                    },
                    "symptom_notes": {
                        "type": "string",
                        "description": (
                            "One short sentence describing what's going on, in the caller's own "
                            "words. Not a medical history — one sentence only."
                        ),
                    },
                },
                required=["hold_id", "patient_name", "reason"],
            ),
        ]
    )

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

import os
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
    from .intents import Intent
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

    # Phase 15. Was 10 s, and that timeout sat INSIDE the voice turn: a slow backend meant ten
    # seconds of dead air, which a caller reads as a dropped call. 5 s is still far outside the
    # measured tool latency (tens of ms locally, low hundreds against Supabase) and the filler
    # line now covers the first 2.5 s of it. `CLINIC_TOOL_TIMEOUT_S` overrides.
    DEFAULT_TIMEOUT_S = float(os.getenv("CLINIC_TOOL_TIMEOUT_S", "5.0"))

    def __init__(self, base_url: str, *, timeout: float | None = None, call_id: str | None = None,
                 api_token: str | None = None) -> None:
        timeout = self.DEFAULT_TIMEOUT_S if timeout is None else timeout
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

    async def _send(self, method: str, url: str, *, safe: bool, **kwargs) -> httpx.Response:
        """One HTTP call, with one retry when a retry cannot do harm.

        `safe` is the whole design here, and it is deliberately narrow. A GET, or a POST
        carrying a stable Idempotency-Key, can be retried on anything transient. Everything
        else — /cancel, /reschedule, /staff-tasks — is retried ONLY when the request provably
        never reached the application: a connection failure, or a gateway status the proxy
        itself produced. Retrying a read timeout on /staff-tasks would file the refill twice,
        which is exactly the class of bug the idempotency keys exist to prevent elsewhere.
        """
        last_exc: httpx.HTTPError | None = None
        for attempt in (1, 2):
            try:
                resp = await self._client.request(method, url, **kwargs)
            except httpx.ConnectError as exc:
                last_exc = exc          # never reached the app; always safe to retry
            except httpx.HTTPError as exc:
                if not safe:
                    raise
                last_exc = exc
            else:
                retryable = resp.status_code in ({502, 503, 504} if not safe else
                                                 {429, 500, 502, 503, 504, 529})
                if not retryable or attempt == 2:
                    return resp
                logger.warning(f"TOOL ▶ {method} {url} -> {resp.status_code}; retrying once")
                continue
            if attempt == 2:
                raise last_exc
            logger.warning(f"TOOL ▶ {method} {url} failed ({last_exc}); retrying once")
        raise last_exc  # unreachable; keeps the type checker and the reader honest

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
            resp = await self._send("GET", "/availability", params=params, safe=True)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            return {"ok": False, "error": f"could not reach the scheduling system ({exc})"}
        slots = [_shape_slot(s) for s in resp.json()["slots"]]
        return {"ok": True, "count": len(slots), "slots": slots}

    async def hold_slot(self, *, slot_id: int) -> dict:
        try:
            resp = await self._send(
                "POST",
                "/hold-slot",
                json={"slot_id": slot_id},
                headers={"Idempotency-Key": self._idempotency_key("hold", str(slot_id))},
                safe=True,   # the key makes a replayed hold return the original hold
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
        phone: str | None = None,
    ) -> dict:
        payload = {
            "hold_id": hold_id,
            "patient_name": patient_name,
            "reason": reason,
            "date_of_birth": date_of_birth,
            "new_patient": new_patient,
            "symptom_notes": symptom_notes,
            # Phase 13: the ANI, injected by the reducer. It is what turns this booking into a
            # patient record, so the NEXT call from this number is a returning caller. Absent
            # on the local path and in the eval harness, where the booking still succeeds.
            "phone": phone,
        }
        try:
            resp = await self._send(
                "POST",
                "/confirm-booking",
                json=payload,
                safe=True,   # keyed on the hold: a replay returns the original confirmation
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

    # --- Phase 13: verified caller flows -------------------------------------------------
    #
    # `phone` and `date_of_birth` are NEVER model-supplied on these calls. The reducer injects
    # them from CallState (the ANI from SIP, the DOB the caller spoke and the API already
    # matched once), so a prompt injection cannot talk the agent into looking up somebody
    # else's chart by naming a different number. The API re-verifies the pair anyway.

    async def _verified_post(self, path: str, payload: dict, *, safe: bool = False) -> dict:
        """POST a verified-caller request and map the standard failures for the model.

        `safe=True` only for the read-shaped ones (verify, list). A cancel, a reschedule, or a
        staff task must not be replayed on a read timeout — see `_send`.
        """
        try:
            resp = await self._send("POST", path, json=payload, safe=safe)
        except httpx.HTTPError as exc:
            return {"ok": False, "error": f"could not reach the scheduling system ({exc})"}
        if resp.status_code == 403:
            return {
                "ok": False,
                "status": 403,
                # The wording matters: the model must not tell the caller their number is or
                # is not on file — that is the enumeration leak the API is careful to avoid.
                "error": "could not verify — ask for the date of birth again, or offer staff",
            }
        if resp.status_code == 404:
            return {"ok": False, "status": 404, "error": "no such appointment for this caller"}
        if resp.status_code == 409:
            return {"ok": False, "status": 409, "error": "that time was just taken — offer another"}
        try:
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            return {"ok": False, "error": f"the scheduling system returned an error ({exc})"}
        return {"ok": True, **resp.json()}

    async def caller_memory(self, *, phone: str) -> dict:
        """Pre-greeting lookup by ANI. Returns no identity — see the API's docstring."""
        try:
            resp = await self._send("GET", "/caller-memory", params={"phone": phone}, safe=True)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            # A memory miss must never block a call: the agent simply greets as it always did.
            return {"ok": False, "error": str(exc), "known": False, "upcoming_appointments": 0}
        return {"ok": True, **resp.json()}

    async def verify_identity(
        self, *, phone: str, date_of_birth: str, name: str | None = None
    ) -> dict:
        return await self._verified_post(
            "/verify-identity",
            {"phone": phone, "date_of_birth": date_of_birth, "name": name},
            safe=True,
        )

    async def list_appointments(
        self, *, phone: str, date_of_birth: str, name: str | None = None
    ) -> dict:
        result = await self._verified_post(
            "/appointments",
            {"phone": phone, "date_of_birth": date_of_birth, "name": name},
            safe=True,
        )
        if result.get("ok"):
            appts = [
                {**a, "display_time": _format_slot_time(a["start_time"])}
                for a in result.get("appointments", [])
            ]
            return {"ok": True, "count": len(appts), "appointments": appts}
        return result

    async def reschedule_appointment(
        self, *, confirmation_id: str, new_slot_id: int, phone: str, date_of_birth: str,
        name: str | None = None,
    ) -> dict:
        result = await self._verified_post("/reschedule", {
            "confirmation_id": confirmation_id, "new_slot_id": new_slot_id,
            "phone": phone, "date_of_birth": date_of_birth, "name": name,
        })
        if result.get("ok"):
            result["display_time"] = _format_slot_time(result["start_time"])
        return result

    async def cancel_appointment(
        self, *, confirmation_id: str, phone: str, date_of_birth: str,
        reason: str | None = None, name: str | None = None,
    ) -> dict:
        return await self._verified_post("/cancel", {
            "confirmation_id": confirmation_id, "phone": phone,
            "date_of_birth": date_of_birth, "reason": reason, "name": name,
        })

    async def request_refill(
        self, *, phone: str, date_of_birth: str, medication: str, notes: str | None = None,
        name: str | None = None,
    ) -> dict:
        return await self._verified_post("/staff-tasks", {
            "phone": phone, "date_of_birth": date_of_birth, "kind": "refill", "name": name,
            "payload": {"medication": medication, "notes": notes},
        })

    async def clinic_info(self, *, topic: str | None = None) -> dict:
        try:
            resp = await self._send(
                "GET", "/clinic-info", params={"topic": topic} if topic else {}, safe=True
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            return {"ok": False, "error": f"could not reach the scheduling system ({exc})"}
        return {"ok": True, **resp.json()}

    async def post_call_metrics(self, payload: dict) -> dict:
        """Ship one call's operational metrics at teardown (Phase 16).

        `safe=True` because the endpoint is idempotent on `call_id` — a replay replaces the
        call's rows rather than adding a second copy of the call, so a retry cannot double-count
        it in the dashboard's percentiles.

        Never raises. This runs during teardown of a call that has already ended: the caller has
        hung up, nothing is waiting on it, and a metrics sink that can take down a session is
        worse than no metrics sink. The local JSONL still holds the same records either way.
        """
        try:
            resp = await self._send("POST", "/call-metrics", json=payload, safe=True)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            return {"ok": False, "error": f"could not post call metrics ({exc})"}
        return {"ok": True, **resp.json()}


# --- Pipecat function handlers ----------------------------------------------------------
# Each handler unpacks params.arguments, calls the client, logs a TOOL ▶ line, and hands the
# result back to the LLM via params.result_callback(...). Pipecat appends that result to the
# LLMContext as a tool message and re-runs the LLM automatically, so the model "sees" the real
# availability/hold/booking data and speaks from it — no manual context editing here.


def _redact_phone(phone: str | None) -> str:
    """Last four digits only. The ANI is an identifier; logs are not a place to keep one."""
    digits = "".join(c for c in (phone or "") if c.isdigit())
    return f"***{digits[-4:]}" if digits else "unset"


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
    # Phase 13
    "verify_identity": "/verify-identity",
    "list_appointments": "/appointments",
    "reschedule_appointment": "/reschedule",
    "cancel_appointment": "/cancel",
    "request_refill": "/staff-tasks",
    "get_clinic_info": "/clinic-info",
}

# Tools whose arguments the REDUCER completes from CallState — the caller's number (from the
# SIP ANI) and the date of birth the API has already matched. The model never sees or supplies
# either, so no prompt injection can redirect a lookup at another patient's chart.
CALLER_SCOPED_TOOLS = frozenset({
    "verify_identity", "list_appointments", "reschedule_appointment",
    "cancel_appointment", "request_refill", "confirm_booking",
})

# Tools that must not execute until `CallState.identity_verified` is true. `verify_identity` is
# obviously not in the set — it is how the flag gets set. `confirm_booking` is not either: a
# NEW appointment discloses nothing, and requiring verification to book would lock out every
# first-time caller.
#
# This is the in-process half of a two-layer gate. The API re-verifies (phone, DOB) on every
# one of these calls regardless, because the flag lives in a process driven by a language
# model, and a language model is not a security boundary.
VERIFICATION_REQUIRED_TOOLS = frozenset({
    "list_appointments", "reschedule_appointment", "cancel_appointment", "request_refill",
})


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
            phone=args.get("phone"),
        )
    elif name == "verify_identity":
        # PHI minimization: the DOB is the verification secret. Log only that one was supplied.
        logger.info(
            f"TOOL ▶ POST /verify-identity req={{phone={_redact_phone(args.get('phone'))}, "
            f"dob={'set' if args.get('date_of_birth') else 'unset'}}}"
        )
        result = await client.verify_identity(
            phone=args.get("phone", ""), date_of_birth=args.get("date_of_birth", ""),
            name=args.get("name"),
        )
    elif name == "list_appointments":
        logger.info(f"TOOL ▶ POST /appointments req={{phone={_redact_phone(args.get('phone'))}}}")
        result = await client.list_appointments(
            phone=args.get("phone", ""), date_of_birth=args.get("date_of_birth", ""),
            name=args.get("name"),
        )
    elif name == "reschedule_appointment":
        logger.info(
            f"TOOL ▶ POST /reschedule req={{confirmation_id={args.get('confirmation_id')!r}, "
            f"new_slot_id={args.get('new_slot_id')!r}}}"
        )
        result = await client.reschedule_appointment(
            confirmation_id=args.get("confirmation_id", ""),
            new_slot_id=args.get("new_slot_id"),
            phone=args.get("phone", ""),
            date_of_birth=args.get("date_of_birth", ""),
            name=args.get("name"),
        )
    elif name == "cancel_appointment":
        logger.info(
            f"TOOL ▶ POST /cancel req={{confirmation_id={args.get('confirmation_id')!r}}}"
        )
        result = await client.cancel_appointment(
            confirmation_id=args.get("confirmation_id", ""),
            phone=args.get("phone", ""),
            date_of_birth=args.get("date_of_birth", ""),
            reason=args.get("reason"),
            name=args.get("name"),
        )
    elif name == "request_refill":
        # The medication name is clinical detail; log its presence, not the drug.
        logger.info(
            f"TOOL ▶ POST /staff-tasks req={{kind='refill', "
            f"medication={'set' if args.get('medication') else 'unset'}}}"
        )
        result = await client.request_refill(
            phone=args.get("phone", ""),
            date_of_birth=args.get("date_of_birth", ""),
            medication=args.get("medication", ""),
            notes=args.get("notes"),
            name=args.get("name"),
        )
    elif name == "get_clinic_info":
        logger.info(f"TOOL ▶ GET /clinic-info req={{topic={args.get('topic')!r}}}")
        result = await client.clinic_info(topic=args.get("topic"))
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
    elif name == "verify_identity":
        logger.info("TOOL ▶ POST /verify-identity → VERIFIED (caller matched a patient on file)")
    elif name == "list_appointments":
        logger.info(f"TOOL ▶ POST /appointments → {result['count']} upcoming")
    elif name == "reschedule_appointment":
        logger.info(
            f"TOOL ▶ POST /reschedule → moved {result['confirmation_id']} to "
            f"{result['display_time']} with {result['provider_name']}"
        )
    elif name == "cancel_appointment":
        logger.info(f"TOOL ▶ POST /cancel → cancelled {result['confirmation_id']}")
    elif name == "request_refill":
        logger.info(
            f"TOOL ▶ POST /staff-tasks → refill task {result['task_id']} OPEN for staff "
            "(the agent does not approve refills)"
        )
    elif name == "get_clinic_info":
        found = "hit" if result.get("content") else "miss"
        logger.info(f"TOOL ▶ GET /clinic-info → {found} topic={result.get('topic')!r}")


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


def _booking_tools() -> list[FunctionSchema]:
    """The three tools that book a NEW appointment. Unchanged since Phase 2."""
    reasons_list = "/".join(REASON_CATEGORIES)
    return [
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


def _caller_tools() -> list[FunctionSchema]:
    """Phase 13 — the tools that touch an EXISTING patient's appointments.

    Two things are deliberately absent from every schema below: the caller's phone number and
    their date of birth. Both are injected by the reducer from CallState, so the model cannot
    supply them, cannot be argued into supplying different ones, and never has to ask the
    caller for a number the agent is already connected to.
    """
    return [
        FunctionSchema(
            name="verify_identity",
            description=(
                "Check the caller against the clinic's records. Call this BEFORE any tool that "
                "reads or changes an existing appointment — those refuse until it succeeds. "
                "Ask for their FULL NAME and DATE OF BIRTH (one at a time, in natural "
                "sentences) and send both: the clinic may hold their appointment under a "
                "number this call is not coming from, and the name is what finds it. Do NOT "
                "call this until the caller has actually given you BOTH — never send a "
                "placeholder, an empty string, or a guess in either field. If it "
                "fails, ask once more in case you misheard the name or the date, then offer a "
                "staff member. NEVER say whether the number is on file, and never guess or "
                "read back a date of birth."
            ),
            properties={
                "date_of_birth": {
                    "type": "string",
                    "description": (
                        "The date of birth the caller just spoke, normalized to MM/DD/YYYY "
                        "(e.g. 'March 15th 1990' -> '03/15/1990')."
                    ),
                },
                "name": {
                    "type": "string",
                    "description": (
                        "The caller's full name as they gave it, exactly — do not correct the "
                        "spelling or expand a nickname."
                    ),
                },
            },
            required=["date_of_birth", "name"],
        ),
        FunctionSchema(
            name="list_appointments",
            description=(
                "List this caller's upcoming appointments. Requires verify_identity to have "
                "succeeded first. Each result has a confirmation_id (internal — never read it "
                "out unless the caller asks for their confirmation number) and a spoken "
                "display_time. If the list is empty, say you don't see anything upcoming and "
                "offer to book."
            ),
            properties={},
            required=[],
        ),
        FunctionSchema(
            name="reschedule_appointment",
            description=(
                "Move an existing appointment to a different open slot. Requires "
                "verify_identity first. Steps, in order: list_appointments to find the "
                "confirmation_id, check_availability for the caller's new preferred day, read "
                "the new time back and get an explicit yes, THEN call this. If it fails "
                "because the slot was taken, the original appointment is still intact — say "
                "so and offer another time."
            ),
            properties={
                "confirmation_id": {
                    "type": "string",
                    "description": "confirmation_id of the appointment to move, from list_appointments.",
                },
                "new_slot_id": {
                    "type": "integer",
                    "description": "slot_id of the new time, from check_availability.",
                },
            },
            required=["confirmation_id", "new_slot_id"],
        ),
        FunctionSchema(
            name="cancel_appointment",
            description=(
                "Cancel an existing appointment. Requires verify_identity first. Read the "
                "appointment back and get an explicit yes before calling — a cancellation the "
                "caller did not mean is not recoverable by them. Offer to rebook afterwards."
            ),
            properties={
                "confirmation_id": {
                    "type": "string",
                    "description": "confirmation_id of the appointment to cancel, from list_appointments.",
                },
                "reason": {
                    "type": "string",
                    "description": "Short reason if the caller volunteers one. Do not press for it.",
                },
            },
            required=["confirmation_id"],
        ),
        FunctionSchema(
            name="request_refill",
            description=(
                "Send a prescription refill request to clinic staff. Requires verify_identity "
                "first. This creates a task for a human — it does NOT approve anything. Tell "
                "the caller a staff member will review it and follow up; never say the refill "
                "is approved, is on its way, or is appropriate, and never discuss the "
                "medication itself."
            ),
            properties={
                "medication": {
                    "type": "string",
                    "description": "The medication as the caller named it. Do not correct or expand it.",
                },
                "notes": {
                    "type": "string",
                    "description": "One short sentence if the caller added something (pharmacy, urgency).",
                },
            },
            required=["medication"],
        ),
    ]


def _clinic_info_tool() -> FunctionSchema:
    return FunctionSchema(
        name="get_clinic_info",
        description=(
            "Look up a curated clinic fact — hours, location, parking, providers, "
            "appointment_prep, or insurance. This is the ONLY source you may speak these from: "
            "if it returns no content, say a staff member can confirm and offer to help with "
            "an appointment. Never invent an address, a phone number, or a provider."
        ),
        properties={
            "topic": {
                "type": "string",
                "enum": ["hours", "location", "parking", "providers", "appointment_prep",
                         "insurance"],
                "description": "Which fact the caller asked for.",
            }
        },
        required=["topic"],
    )


def build_tools_schema(intent: "Intent | None" = None) -> ToolsSchema:
    """The tool/function schema the LLM is allowed to call, scoped to the caller's intent.

    Phase 12 added the scoping. Through Phase 11 every turn carried all three schemas whatever
    the caller wanted, which costs tokens on every request and — the part that actually matters
    — costs accuracy: a model shown a booking tool while the caller is asking about a bill has
    an option it should not have.

    Phase 13 made the scoping load-bearing rather than merely efficient. There are nine tools
    now, five of which touch an existing patient's record, and the sets below are what keep
    `request_refill` invisible during a booking and `hold_slot` invisible during a
    cancellation. An intent that this build cannot complete still gets NO tools — a model with
    no tools cannot invent an outcome for a caller it cannot actually help.

    ``intent=None`` (turn one, before the classifier has answered) gets the booking set plus
    clinic info: this is a scheduling line, so being briefly over-equipped beats being
    under-equipped on the caller's opening sentence.
    """
    from .intents import Intent as _Intent

    booking = _booking_tools()
    caller = _caller_tools()
    by_name = {t.name: t for t in booking + caller}
    info = _clinic_info_tool()

    def pick(*names: str) -> list[FunctionSchema]:
        return [by_name[n] for n in names]

    if intent is None or intent is _Intent.SCHEDULE_APPOINTMENT:
        tools = booking + [info]
    elif intent is _Intent.RESCHEDULE_APPOINTMENT:
        # check_availability comes along: a reschedule needs a new time to move to.
        tools = pick("verify_identity", "list_appointments", "check_availability",
                     "reschedule_appointment") + [info]
    elif intent is _Intent.CANCEL_APPOINTMENT:
        tools = pick("verify_identity", "list_appointments", "cancel_appointment")
    elif intent is _Intent.MEDICATION_REFILL:
        tools = pick("verify_identity", "request_refill")
    elif intent in (_Intent.HOURS_LOCATION, _Intent.INSURANCE_VERIFICATION):
        # Insurance gets the fact lookup and nothing else. Its prompt tells the model to state
        # the clinic's general policy "from get_clinic_info" — and it used to be handed no
        # tools at all, so it was instructed to use something it did not have. A model in that
        # position either invents the policy or stalls; both are worse than the hand-off.
        tools = [info]
    else:
        tools = []

    return ToolsSchema(standard_tools=tools)

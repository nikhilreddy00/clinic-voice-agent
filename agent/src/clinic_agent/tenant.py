"""Phase 17 — which clinic is this process answering for?

The scheduling API became multi-tenant in this phase: `clinic_id` was on every row from Phase 9,
and what arrived now is the routing — an `X-Clinic-Slug` header, per-tenant facts, per-tenant
timezone, per-tenant escalation number. This module is the agent's half of that.

**Resolution happens once, at worker startup, from the DIALED number** (`LIVEKIT_PHONE_NUMBER`),
not per call. Two reasons, and the second is the honest one:

  * a lookup on the call's critical path would put an HTTP round trip in front of the greeting,
    and the greeting carries the AI disclosure — the Phase-13 rule that caller memory is loaded
    *after* the greeting and never awaited exists for exactly this;
  * with LiveKit Direct dispatch every caller lands in ONE shared room, so a process serves
    exactly one trunk and therefore exactly one clinic. Per-call resolution from
    `sip.trunkPhoneNumber` is a small change, and it belongs with room-per-call dispatch
    (`setup_livekit_sip.py --dispatch individual`), which is the same gate the session router
    sits behind. Building it now would be code that cannot run.

Every failure degrades to the default tenant rather than failing the call: an unreachable API,
an unconfigured number, a DID nobody has registered. A colder greeting is a bad call; a dropped
one is a worse call.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import httpx
from loguru import logger


@dataclass(frozen=True)
class Tenant:
    """A clinic's identity, as the agent needs it: something to say, a clock, and an exit."""

    slug: str
    name: str
    timezone: str
    transfer_number: str = ""


# The tenant this project has always had. Also the fallback, so every default path — local dev,
# the eval harness, the tests, a worker whose API is down — behaves exactly as it did before.
DEFAULT = Tenant(
    slug="grove-family",
    name="Grove Family Clinic",
    timezone="America/New_York",
)

_current: Tenant = DEFAULT


def current() -> Tenant:
    """The tenant this process is answering for. Never None."""
    return _current


def set_current(tenant: Tenant) -> None:
    """Set it directly — used by `load`, by the worker, and by tests."""
    global _current
    _current = tenant


async def load(base_url: str, did: str, *, timeout: float = 5.0) -> Tenant:
    """Resolve the dialed number to a tenant and make it current. Best-effort.

    `CLINIC_SLUG` overrides the lookup entirely, which is how a worker is pinned to a tenant
    without a phone number at all (local dev, a second worker for a clinic whose DID is not
    live yet).
    """
    override = os.getenv("CLINIC_SLUG", "").strip()
    if not did and not override:
        return current()

    try:
        async with httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=timeout) as client:
            resp = await client.get("/clinic", params={"did": did})
            resp.raise_for_status()
            body = resp.json()
    except Exception as exc:  # noqa: BLE001 — a tenant lookup must never fail a call
        logger.warning(
            f"[tenant] could not resolve {did or override!r} ({exc}); "
            f"serving as {DEFAULT.slug!r}"
        )
        return current()

    tenant = Tenant(
        slug=body["slug"],
        name=body["name"],
        timezone=body.get("timezone") or DEFAULT.timezone,
        transfer_number=(body.get("transfer_number") or "").strip(),
    )
    if override and tenant.slug != override:
        logger.warning(
            f"[tenant] CLINIC_SLUG={override!r} overrides the number's tenant {tenant.slug!r}"
        )
        tenant = Tenant(slug=override, name=tenant.name, timezone=tenant.timezone,
                        transfer_number=tenant.transfer_number)
    set_current(tenant)
    logger.info(f"[tenant] answering as {tenant.name!r} ({tenant.slug}, {tenant.timezone})")
    return tenant

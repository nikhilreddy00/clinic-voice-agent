"""Provision LiveKit SIP inbound trunk + dispatch rule for the clinic agent (Phase 5).

One-off, idempotent setup. Run it once (and again any time you want to confirm state):

    cd agent && uv run python scripts/setup_livekit_sip.py

It creates the two LiveKit resources an inbound call needs, if they don't already exist:

  1. An INBOUND SIP TRUNK bound to the clinic phone number (LIVEKIT_PHONE_NUMBER). This is
     what tells LiveKit "calls to this number belong to us."
  2. A DISPATCH RULE of type Direct that routes every inbound call on that trunk into ONE
     fixed room (config.TELEPHONY_ROOM_NAME = "clinic-inbound"), which is the room the agent
     process joins. Direct = single shared room, which is all a single demo call needs.

     Phase-6 concurrency note: to run multiple simultaneous calls you would instead use an
     INDIVIDUAL dispatch rule (api.SIPDispatchRuleIndividual(room_prefix=...)) so each call
     gets its own room, paired with LiveKit agent dispatch that starts one agent process per
     room. That is the "one container per session" model Phase 6 will Dockerize. We use Direct
     here deliberately: it needs no dispatch worker and is trivial to reason about for a demo.

The script is chatty on purpose: for each resource it prints whether it was FOUND (already
provisioned) or CREATED (provisioned now), so a re-run makes the current state obvious at a
glance. It never deletes or mutates an existing resource.
"""

from __future__ import annotations

import asyncio
import sys

from livekit import api

# Import the shared room-name constant + settings loader from the agent package so the room
# the dispatch rule targets is guaranteed to match the room the agent joins (single source).
from clinic_agent.config import TELEPHONY_ROOM_NAME, load_settings

_TRUNK_NAME = "Clinic inbound trunk"
_RULE_NAME = f"Clinic inbound → {TELEPHONY_ROOM_NAME} room"


def _fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(1)


async def _ensure_inbound_trunk(lk: api.LiveKitAPI, number: str) -> str:
    """Return the trunk id for `number`, creating an inbound trunk if none exists."""
    existing = await lk.sip.list_sip_inbound_trunk(api.ListSIPInboundTrunkRequest())
    for trunk in existing.items:
        if number in trunk.numbers:
            print(
                f"  [trunk]    FOUND    id={trunk.sip_trunk_id} name={trunk.name!r} "
                f"numbers={list(trunk.numbers)}"
            )
            return trunk.sip_trunk_id

    created = await lk.sip.create_sip_inbound_trunk(
        api.CreateSIPInboundTrunkRequest(
            trunk=api.SIPInboundTrunkInfo(name=_TRUNK_NAME, numbers=[number])
        )
    )
    print(
        f"  [trunk]    CREATED  id={created.sip_trunk_id} name={created.name!r} "
        f"numbers={list(created.numbers)}"
    )
    return created.sip_trunk_id


async def _ensure_dispatch_rule(lk: api.LiveKitAPI, trunk_id: str) -> str:
    """Return the id of a Direct dispatch rule → TELEPHONY_ROOM_NAME on `trunk_id`, creating one if absent."""
    existing = await lk.sip.list_sip_dispatch_rule(api.ListSIPDispatchRuleRequest())
    for rule in existing.items:
        direct = rule.rule.dispatch_rule_direct if rule.rule.WhichOneof("rule") == "dispatch_rule_direct" else None
        # trunk_ids == [] means "applies to all trunks", which also covers ours.
        applies_to_trunk = (not rule.trunk_ids) or (trunk_id in rule.trunk_ids)
        if direct and direct.room_name == TELEPHONY_ROOM_NAME and applies_to_trunk:
            print(
                f"  [dispatch] FOUND    id={rule.sip_dispatch_rule_id} room={TELEPHONY_ROOM_NAME!r} "
                f"trunk_ids={list(rule.trunk_ids) or ['<all>']}"
            )
            return rule.sip_dispatch_rule_id

    created = await lk.sip.create_sip_dispatch_rule(
        api.CreateSIPDispatchRuleRequest(
            name=_RULE_NAME,
            trunk_ids=[trunk_id],
            rule=api.SIPDispatchRule(
                dispatch_rule_direct=api.SIPDispatchRuleDirect(room_name=TELEPHONY_ROOM_NAME)
            ),
        )
    )
    print(
        f"  [dispatch] CREATED  id={created.sip_dispatch_rule_id} room={TELEPHONY_ROOM_NAME!r} "
        f"trunk_ids={[trunk_id]}"
    )
    return created.sip_dispatch_rule_id


async def main() -> None:
    settings = load_settings()
    if not (settings.livekit_url and settings.livekit_api_key and settings.livekit_api_secret):
        _fail("LIVEKIT_URL / LIVEKIT_API_KEY / LIVEKIT_API_SECRET must be set in agent/.env")
    if not settings.livekit_phone_number:
        _fail("LIVEKIT_PHONE_NUMBER must be set in agent/.env (the clinic number, e.g. +14842950169)")

    number = settings.livekit_phone_number
    print(f"LiveKit SIP setup → {settings.livekit_url}")
    print(f"  phone number: {number}")
    print(f"  target room:  {TELEPHONY_ROOM_NAME} (Direct dispatch)\n")

    lk = api.LiveKitAPI(
        url=settings.livekit_url,
        api_key=settings.livekit_api_key,
        api_secret=settings.livekit_api_secret,
    )
    try:
        trunk_id = await _ensure_inbound_trunk(lk, number)
        rule_id = await _ensure_dispatch_rule(lk, trunk_id)
    finally:
        await lk.aclose()

    print(
        f"\nDone. Inbound calls to {number} now route to room {TELEPHONY_ROOM_NAME!r}.\n"
        f"Start the agent with MODE=telephony and it will join that room and wait for the call.\n"
        f"  (trunk={trunk_id}, dispatch_rule={rule_id})"
    )


if __name__ == "__main__":
    asyncio.run(main())

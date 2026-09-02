"""Provision LiveKit SIP inbound trunk + dispatch rule for the clinic agent (Phase 5 / 11).

One-off, idempotent setup. Run it once (and again any time you want to confirm state):

    cd agent && uv run python scripts/setup_livekit_sip.py                  # Direct (default)
    cd agent && uv run python scripts/setup_livekit_sip.py --dispatch individual

It creates the two LiveKit resources an inbound call needs, if they don't already exist:

  1. An INBOUND SIP TRUNK bound to the clinic phone number (LIVEKIT_PHONE_NUMBER). This is
     what tells LiveKit "calls to this number belong to us."
  2. A DISPATCH RULE deciding which room each inbound call lands in.

DISPATCH MODES
--------------
**Direct** (default) routes every inbound call into ONE fixed room
(``config.TELEPHONY_ROOM_NAME``), which the long-lived agent process is already joined to. It
needs no control plane and is trivial to reason about — and it caps the system at exactly one
concurrent call, because a second caller would join the same room and the same conversation.

**Individual** (Phase 11) gives each call its own room ``<prefix>_<something>`` and fires a
``room_started`` webhook. The session router (``session_router/``) receives that webhook and
assigns the room to a worker, which starts one ``CallSession`` for it. This is what actually
supports concurrency.

Individual is **opt-in and not the default**, deliberately. Switching the rule changes how a
live phone number behaves for real callers: with Individual dispatch, an agent sitting in the
old shared room will never see another call, so flipping it without the router and workers
running takes the demo number off the air. Choosing that moment is an operator decision, not a
side effect of running a setup script.

The script is chatty on purpose: for each resource it prints whether it was FOUND (already
provisioned) or CREATED (provisioned now), so a re-run makes the current state obvious at a
glance. **It never deletes or mutates an existing resource** — including a dispatch rule of the
other kind. If you have a Direct rule and ask for Individual, it creates the Individual rule
and tells you the Direct one is still there and still winning; removing it is a deliberate,
manual step.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from livekit import api

# Import the shared room-name constant + settings loader from the agent package so the room
# the dispatch rule targets is guaranteed to match the room the agent joins (single source).
from clinic_agent.config import TELEPHONY_ROOM_NAME, load_settings

_TRUNK_NAME = "Clinic inbound trunk"
_RULE_NAME = f"Clinic inbound → {TELEPHONY_ROOM_NAME} room"

# Room-per-call prefix for Individual dispatch. LiveKit appends a unique suffix per call, and
# the session router keys assignments on the resulting room name.
ROOM_PREFIX = "clinic-call"
_INDIVIDUAL_RULE_NAME = f"Clinic inbound → {ROOM_PREFIX}_* (room per call)"


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


async def _ensure_individual_rule(lk: api.LiveKitAPI, trunk_id: str) -> str:
    """Return the id of an Individual (room-per-call) dispatch rule, creating one if absent."""
    existing = await lk.sip.list_sip_dispatch_rule(api.ListSIPDispatchRuleRequest())
    for rule in existing.items:
        kind = rule.rule.WhichOneof("rule")
        individual = rule.rule.dispatch_rule_individual if kind == "dispatch_rule_individual" else None
        applies_to_trunk = (not rule.trunk_ids) or (trunk_id in rule.trunk_ids)
        if individual and individual.room_prefix == ROOM_PREFIX and applies_to_trunk:
            print(
                f"  [dispatch] FOUND    id={rule.sip_dispatch_rule_id} "
                f"room_prefix={ROOM_PREFIX!r} trunk_ids={list(rule.trunk_ids) or ['<all>']}"
            )
            return rule.sip_dispatch_rule_id

    created = await lk.sip.create_sip_dispatch_rule(
        api.CreateSIPDispatchRuleRequest(
            name=_INDIVIDUAL_RULE_NAME,
            trunk_ids=[trunk_id],
            rule=api.SIPDispatchRule(
                dispatch_rule_individual=api.SIPDispatchRuleIndividual(room_prefix=ROOM_PREFIX)
            ),
        )
    )
    print(
        f"  [dispatch] CREATED  id={created.sip_dispatch_rule_id} "
        f"room_prefix={ROOM_PREFIX!r} trunk_ids={[trunk_id]}"
    )
    return created.sip_dispatch_rule_id


async def _warn_about_conflicting_rules(lk: api.LiveKitAPI, trunk_id: str, wanted: str) -> None:
    """Point out a leftover rule of the other kind rather than quietly deleting it.

    Two rules on one trunk is ambiguous, and which one wins is LiveKit's business, not ours.
    Deleting the other one automatically would mean a setup script silently changing how a live
    phone number behaves — so it is reported and left alone.
    """
    existing = await lk.sip.list_sip_dispatch_rule(api.ListSIPDispatchRuleRequest())
    other = "dispatch_rule_direct" if wanted == "individual" else "dispatch_rule_individual"
    stale = [
        rule
        for rule in existing.items
        if rule.rule.WhichOneof("rule") == other
        and ((not rule.trunk_ids) or trunk_id in rule.trunk_ids)
    ]
    if not stale:
        return
    print(
        f"\n  !! {len(stale)} existing {other.replace('dispatch_rule_', '').upper()} rule(s) "
        f"still apply to this trunk:"
    )
    for rule in stale:
        print(f"       {rule.sip_dispatch_rule_id}  {rule.name!r}")
    print(
        "     Two dispatch rules on one trunk is ambiguous. This script never deletes rules —\n"
        "     remove the one you don't want with:\n"
        "       lk sip dispatch delete <id>     (or the LiveKit dashboard)"
    )


async def main() -> None:
    parser = argparse.ArgumentParser(description="Provision LiveKit SIP for the clinic agent")
    parser.add_argument(
        "--dispatch",
        choices=("direct", "individual"),
        default="direct",
        help=(
            "direct: every call into one shared room (single concurrent call, the Phase-5 "
            "default). individual: a room per call for the Phase-11 router + workers."
        ),
    )
    args = parser.parse_args()

    settings = load_settings()
    if not (settings.livekit_url and settings.livekit_api_key and settings.livekit_api_secret):
        _fail("LIVEKIT_URL / LIVEKIT_API_KEY / LIVEKIT_API_SECRET must be set in agent/.env")
    if not settings.livekit_phone_number:
        _fail("LIVEKIT_PHONE_NUMBER must be set in agent/.env (the clinic number, e.g. +14842951203)")

    number = settings.livekit_phone_number
    individual = args.dispatch == "individual"
    print(f"LiveKit SIP setup → {settings.livekit_url}")
    print(f"  phone number: {number}")
    if individual:
        print(f"  target rooms: {ROOM_PREFIX}_* (Individual dispatch — room per call)\n")
    else:
        print(f"  target room:  {TELEPHONY_ROOM_NAME} (Direct dispatch)\n")

    lk = api.LiveKitAPI(
        url=settings.livekit_url,
        api_key=settings.livekit_api_key,
        api_secret=settings.livekit_api_secret,
    )
    try:
        trunk_id = await _ensure_inbound_trunk(lk, number)
        if individual:
            rule_id = await _ensure_individual_rule(lk, trunk_id)
        else:
            rule_id = await _ensure_dispatch_rule(lk, trunk_id)
        await _warn_about_conflicting_rules(lk, trunk_id, args.dispatch)
    finally:
        await lk.aclose()

    if individual:
        print(
            f"\nDone. Inbound calls to {number} now get their own room {ROOM_PREFIX}_*.\n"
            f"This needs the control plane running — a room with no worker assigned to it is a\n"
            f"caller listening to silence:\n"
            f"  1. session router:  uv run uvicorn app:app --port 8080   (in session_router/)\n"
            f"  2. point the LiveKit webhook at  <router-url>/livekit/webhook\n"
            f"  3. start one or more workers registered against that router\n"
            f"  (trunk={trunk_id}, dispatch_rule={rule_id})"
        )
    else:
        print(
            f"\nDone. Inbound calls to {number} now route to room {TELEPHONY_ROOM_NAME!r}.\n"
            f"Start the agent with MODE=telephony and it will join that room and wait for the call.\n"
            f"  (trunk={trunk_id}, dispatch_rule={rule_id})"
        )


if __name__ == "__main__":
    asyncio.run(main())

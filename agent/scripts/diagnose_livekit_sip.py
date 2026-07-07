"""Read-only diagnostic for the LiveKit SIP inbound routing chain (Phase 5 debugging).

Dumps the ACTUAL state of the two resources an inbound call depends on and checks whether the
routing chain for LIVEKIT_PHONE_NUMBER is complete:

    number → inbound trunk (numbers match?) → dispatch rule (Direct → clinic-inbound?) → room

It creates/deletes NOTHING. Run it when a call drops immediately, to see whether the drop is a
setup problem (no matching trunk/rule → LiveKit rejects the INVITE) or whether the chain is
intact and the problem is upstream (the PSTN number isn't actually routed to this LiveKit
project) or agent-side.

    cd agent && uv run python scripts/diagnose_livekit_sip.py
"""

from __future__ import annotations

import asyncio
import sys

from livekit import api

from clinic_agent.config import TELEPHONY_ROOM_NAME, load_settings


def _fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(1)


async def main() -> None:
    settings = load_settings()
    if not (settings.livekit_url and settings.livekit_api_key and settings.livekit_api_secret):
        _fail("LIVEKIT_URL / LIVEKIT_API_KEY / LIVEKIT_API_SECRET must be set in agent/.env")
    number = settings.livekit_phone_number
    print(f"LiveKit project: {settings.livekit_url}")
    print(f"Clinic number:   {number or '<UNSET>'}")
    print(f"Expected room:   {TELEPHONY_ROOM_NAME}\n")

    lk = api.LiveKitAPI(
        url=settings.livekit_url,
        api_key=settings.livekit_api_key,
        api_secret=settings.livekit_api_secret,
    )
    try:
        trunks = (await lk.sip.list_sip_inbound_trunk(api.ListSIPInboundTrunkRequest())).items
        rules = (await lk.sip.list_sip_dispatch_rule(api.ListSIPDispatchRuleRequest())).items
    finally:
        await lk.aclose()

    # --- Inbound trunks ---
    print(f"=== Inbound SIP trunks ({len(trunks)}) ===")
    if not trunks:
        print("  (none) — no inbound trunk exists, so LiveKit has no route for ANY inbound call.")
    matching_trunk_ids = []
    for t in trunks:
        nums = list(t.numbers)
        matches = number in nums if number else False
        # numbers == [] means "accept any called number".
        accepts_any = not nums
        if matches or accepts_any:
            matching_trunk_ids.append(t.sip_trunk_id)
        flag = "  ← matches our number" if matches else ("  ← accepts ANY number" if accepts_any else "")
        print(f"  id={t.sip_trunk_id} name={t.name!r} numbers={nums or ['<any>']}{flag}")

    # --- Dispatch rules ---
    print(f"\n=== SIP dispatch rules ({len(rules)}) ===")
    routes_to_room = False
    for r in rules:
        kind = r.rule.WhichOneof("rule")
        if kind == "dispatch_rule_direct":
            target = f"Direct → room {r.rule.dispatch_rule_direct.room_name!r}"
            room_ok = r.rule.dispatch_rule_direct.room_name == TELEPHONY_ROOM_NAME
        elif kind == "dispatch_rule_individual":
            target = f"Individual → room_prefix {r.rule.dispatch_rule_individual.room_prefix!r}"
            room_ok = False
        else:
            target = f"{kind}"
            room_ok = False
        trunk_ids = list(r.trunk_ids)
        trunk_ok = (not trunk_ids) or any(tid in matching_trunk_ids for tid in trunk_ids)
        if room_ok and trunk_ok:
            routes_to_room = True
        print(
            f"  id={r.sip_dispatch_rule_id} name={r.name!r} {target} "
            f"trunk_ids={trunk_ids or ['<all>']}"
        )

    # --- Verdict ---
    print("\n=== Verdict ===")
    if not number:
        print("  ✗ LIVEKIT_PHONE_NUMBER is unset — set it in agent/.env, then re-run setup.")
    elif not matching_trunk_ids:
        print(f"  ✗ No inbound trunk matches {number}. LiveKit will REJECT the call (immediate")
        print("    hang-up). Fix: run  uv run python scripts/setup_livekit_sip.py")
    elif not routes_to_room:
        print(f"  ✗ A trunk matches {number}, but no dispatch rule routes it to {TELEPHONY_ROOM_NAME!r}.")
        print("    LiveKit accepts the trunk but has nowhere to send the call → drop.")
        print("    Fix: run  uv run python scripts/setup_livekit_sip.py")
    else:
        print(f"  ✓ Routing chain is COMPLETE: {number} → trunk {matching_trunk_ids} →")
        print(f"    dispatch rule → room {TELEPHONY_ROOM_NAME!r}.")
        print("    If the call STILL drops immediately, the cause is NOT this setup. Check:")
        print("     1. Is the PSTN number actually routed to THIS LiveKit project? (dashboard →")
        print("        Telephony/SIP: does an inbound call even appear in the SIP log when you dial?)")
        print("     2. Is the agent running with MODE=telephony and showing 'Connected to")
        print(f"        {TELEPHONY_ROOM_NAME}'? (a missing agent usually still holds the call, but")
        print("        confirm it anyway.)")
        print("     3. Read the SIP call's status/disposition in the LiveKit dashboard — it states")
        print("        the exact reason the call ended.")


if __name__ == "__main__":
    asyncio.run(main())

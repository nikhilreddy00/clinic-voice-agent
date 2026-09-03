#!/usr/bin/env python3
"""Wipe every appointment and regenerate a fresh block of slots. Dry-run by default.

    uv run python scripts/reset_demo_data.py               # show what would go
    uv run python scripts/reset_demo_data.py --yes         # do it

Removes all bookings, patients, caller memory, staff tasks, and idempotency keys, then DELETES
and regenerates the slot table so availability starts from today again.

Slots are deleted rather than re-seeded over the top: `db._seed` inserts them with fixed ids and
`ON CONFLICT (id) DO NOTHING`, so re-running it against an existing table leaves every stale
start_time exactly where it was. That is correct for boot-time idempotency and useless for a
reset, which is the whole reason this script exists rather than "just restart the API".

Everything here is synthetic by policy (CLAUDE.md), so there is nothing to preserve — but this
still refuses to guess at a database, because "wipe all appointments" against the wrong one is
not recoverable.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import psycopg
from psycopg.rows import dict_row

from app import db

# What a reset clears. `clinics`, `providers`, and `clinic_facts` stay: they are the clinic
# itself, not call data, and dropping them would leave the agent with no one to book with.
WIPE = ("audit_log", "staff_tasks", "call_summaries", "caller_memory",
        "idempotency_keys", "bookings", "patients")


def _describe(url: str) -> str:
    from urllib.parse import urlparse

    u = urlparse(url)
    port = f":{u.port}" if u.port else ""
    return f"{u.username or '?'}@{u.hostname or '?'}{port}{u.path or ''}"


async def run(url: str, apply: bool) -> int:
    print(f"target: {_describe(url)}")
    print(f"mode  : {'APPLY — this deletes' if apply else 'dry run — read only'}\n")
    try:
        conn_ctx = await psycopg.AsyncConnection.connect(
            url, row_factory=dict_row, connect_timeout=15, autocommit=True
        )
    except psycopg.OperationalError as exc:
        print(f"ERROR: could not connect to {_describe(url)}\n\n  {exc}\n", file=sys.stderr)
        print("Set CLINIC_DATABASE_URL to the database the agent books against.\n"
              "For Supabase use the SESSION POOLER host — see CLAUDE.md.", file=sys.stderr)
        return 2

    async with conn_ctx as conn:
        present = {
            r["tablename"]
            for r in await (await conn.execute(
                "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
            )).fetchall()
        }
        print("current contents:")
        for table in ("bookings", "patients", "caller_memory", "staff_tasks", "slots"):
            if table not in present:
                continue
            n = (await (await conn.execute(f"SELECT count(*) AS n FROM {table}")).fetchone())["n"]
            print(f"  {table:16s} {n:6d}")

        if not apply:
            print("\ndry run — nothing deleted. Re-run with --yes to reset.")
            return 0

        targets = [t for t in WIPE if t in present]
        await conn.execute(f"TRUNCATE {', '.join(targets)} CASCADE")
        print(f"\ncleared: {', '.join(targets)}")

        # Slots go entirely, then come back dated from today. CASCADE above already removed the
        # bookings that referenced them.
        await conn.execute("TRUNCATE slots CASCADE")
        await db._seed(conn)

        rows = await (await conn.execute(
            """
            SELECT count(*) AS n, min(start_time) AS first, max(start_time) AS last
              FROM slots WHERE status = 'available'
            """
        )).fetchone()
        first = rows["first"].astimezone(db.CLINIC_TZ) if rows["first"] else None
        last = rows["last"].astimezone(db.CLINIC_TZ) if rows["last"] else None
        print(f"regenerated: {rows['n']} available slots")
        if first and last:
            print(f"  from {first:%A %d %B %H:%M} to {last:%A %d %B %H:%M} (clinic-local)")
        if not rows["n"]:
            print("  WARNING: no available slots — the agent cannot book", file=sys.stderr)
            return 1

        left = {
            t: (await (await conn.execute(f"SELECT count(*) AS n FROM {t}")).fetchone())["n"]
            for t in ("bookings", "patients") if t in present
        }
        print(f"  bookings={left.get('bookings', 0)} patients={left.get('patients', 0)}")
        print("\nreset complete — fresh slots, no appointments, no patients.")
        return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default=None, help="database to reset (default: $CLINIC_DATABASE_URL)")
    ap.add_argument("--yes", action="store_true", help="actually delete (default: dry run)")
    args = ap.parse_args()

    url = args.url or os.getenv("CLINIC_DATABASE_URL")
    if not url:
        print("ERROR: no database specified.\n\n"
              "  export CLINIC_DATABASE_URL='<the URL the agent books against>'\n"
              "  uv run python scripts/reset_demo_data.py --yes\n\n"
              "This deletes every appointment, so it will not fall back to a default.",
              file=sys.stderr)
        return 2
    return asyncio.run(run(url, args.yes))


if __name__ == "__main__":
    raise SystemExit(main())

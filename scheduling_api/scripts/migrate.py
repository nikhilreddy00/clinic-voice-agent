#!/usr/bin/env python
"""Apply the schema to a database and verify it — the Supabase bootstrap path.

    uv run python scripts/migrate.py                 # apply + verify
    uv run python scripts/migrate.py --check         # preflight only, no writes
    uv run python scripts/migrate.py --url postgresql://...

WHY THIS EXISTS RATHER THAN "just run the app once"
---------------------------------------------------
The app applies its own schema on boot (db.init_db is idempotent), so a migration script is not
strictly required. What IS required is a good error when the connection is wrong, because every
Supabase misconfiguration fails in a way that does not name its own cause:

  * Direct connection on an IPv4-only network -> the hostname resolves to an IPv6 address only,
    so the client hangs until timeout. Looks like "Supabase is down", is actually "use the
    session pooler".
  * Transaction pooler (port 6543) -> connects fine, works fine, then fails on the sixth
    execution of a hot query, because psycopg3 starts preparing statements after 5 and Supavisor
    transaction mode rejects them.
  * Paused free project -> connection refused. Free projects pause after ~7 days without
    database activity, and nothing about the error says so.

Each of those is detected and explained below instead of being left to a stack trace.
"""

from __future__ import annotations

import argparse
import asyncio
import ipaddress
import os
import socket
import sys
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import psycopg  # noqa: E402
from psycopg.rows import dict_row  # noqa: E402

from app import db  # noqa: E402

SESSION_POOLER_PORT = 5432
TRANSACTION_POOLER_PORT = 6543


def _describe_endpoint(url: str) -> list[str]:
    """Classify the connection target and warn about the traps, without connecting."""
    notes: list[str] = []
    parsed = urlparse(url)
    host = parsed.hostname or ""
    port = parsed.port or SESSION_POOLER_PORT

    is_supabase = "supabase.co" in host or "supabase.com" in host
    is_pooler = "pooler.supabase.com" in host

    if not is_supabase:
        notes.append(f"target: self-hosted/local Postgres at {host}:{port}")
        return notes

    if is_pooler and port == TRANSACTION_POOLER_PORT:
        notes.append(
            "target: Supabase TRANSACTION pooler (port 6543).\n"
            "    Prepared statements are unsupported here; db.py disables them automatically.\n"
            "    For a long-lived service like this one, the SESSION pooler (port 5432) is the\n"
            "    better endpoint — it keeps prepared statements and real session semantics."
        )
    elif is_pooler:
        notes.append("target: Supabase SESSION pooler (port 5432) — recommended for this service")
    else:
        notes.append(
            f"target: Supabase DIRECT connection ({host}:{port}).\n"
            "    This host is IPv6-only unless the project has the paid IPv4 add-on. If this\n"
            "    machine or your deploy target is IPv4-only, the connection will HANG rather\n"
            "    than fail cleanly. Prefer the session pooler:\n"
            "      postgresql://postgres.<ref>:<pw>@aws-<region>.pooler.supabase.com:5432/postgres"
        )

    # Resolve to confirm the IPv4/IPv6 story rather than only warning about it.
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
        families = {
            "IPv6" if isinstance(ipaddress.ip_address(i[4][0]), ipaddress.IPv6Address) else "IPv4"
            for i in infos
        }
        notes.append(f"    DNS resolves to: {', '.join(sorted(families))}")
        if families == {"IPv6"}:
            notes.append(
                "    WARNING: IPv6-only. This will hang on an IPv4-only network — use the pooler."
            )
    except socket.gaierror as exc:
        notes.append(f"    WARNING: hostname does not resolve ({exc})")

    return notes


async def _verify(url: str) -> int:
    """Connect and assert the invariants that matter, reporting each one."""
    print("\n[verify]")
    async with await psycopg.AsyncConnection.connect(
        url, row_factory=dict_row, connect_timeout=15
    ) as conn:
        version = (await (await conn.execute("SHOW server_version")).fetchone())["server_version"]
        major = int(version.split(".")[0])
        print(f"  postgres version         : {version}")
        if major < 13:
            print(f"  FAIL: this schema needs Postgres 13+, found {major}")
            return 1

        tables = {
            r["tablename"]
            for r in await (await conn.execute(
                "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
            )).fetchall()
        }
        missing = [t for t in db.TABLES_IN_DEPENDENCY_ORDER if t not in tables]
        print(f"  tables present           : {len(tables) - len(missing)}/"
              f"{len(db.TABLES_IN_DEPENDENCY_ORDER)}")
        if missing:
            print(f"  FAIL: missing tables: {missing}")
            return 1

        # RLS is the one that silently matters on Supabase: public is exposed via the Data API,
        # and bookings/patients hold PHI.
        unprotected = [
            r["tablename"]
            for r in await (await conn.execute(
                "SELECT tablename FROM pg_tables"
                " WHERE schemaname = 'public' AND rowsecurity = false"
            )).fetchall()
        ]
        if unprotected:
            print(f"  FAIL: RLS disabled on: {unprotected}")
            print("        On Supabase these are readable via the Data API with the")
            print("        browser-side publishable key. bookings holds PHI.")
            return 1
        print("  row level security       : enabled on all tables")

        slots = (await (await conn.execute(
            "SELECT count(*) AS n FROM slots WHERE status = 'available'"
        )).fetchone())["n"]
        print(f"  seeded available slots   : {slots}")
        if slots == 0:
            print("  FAIL: no available slots — seeding did not run")
            return 1

        sample = await (await conn.execute(
            "SELECT start_time FROM slots ORDER BY start_time LIMIT 1"
        )).fetchone()
        local = sample["start_time"].astimezone(db.CLINIC_TZ)
        print(f"  first slot (clinic-local): {local.isoformat()}")
        if not (7 <= local.hour <= 19):
            print(f"  FAIL: slot at {local.hour}:00 clinic-local is outside business hours —")
            print("        the timezone fix did not apply.")
            return 1

    print("\n  OK — schema applied, RLS on, seeded, times in business hours.")
    return 0


async def _apply(url: str) -> None:
    print("\n[apply] schema + seed (idempotent)")
    async with await psycopg.AsyncConnection.connect(
        url, row_factory=dict_row, connect_timeout=15, autocommit=True
    ) as conn:
        await conn.execute(db._SCHEMA_PATH.read_text())
        await db._seed(conn)
    print("  done")


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply and verify the clinic schema")
    parser.add_argument("--url", default=os.getenv("CLINIC_DATABASE_URL", db.DATABASE_URL))
    parser.add_argument("--check", action="store_true", help="Verify only; make no changes")
    args = parser.parse_args()

    if not args.url:
        print("ERROR: no database URL. Set CLINIC_DATABASE_URL or pass --url.", file=sys.stderr)
        return 2

    # Never print the password.
    parsed = urlparse(args.url)
    safe = f"{parsed.scheme}://{parsed.username or ''}@{parsed.hostname}:{parsed.port}{parsed.path}"
    print(f"[target] {safe}")
    for note in _describe_endpoint(args.url):
        print(f"  {note}")

    try:
        if not args.check:
            asyncio.run(_apply(args.url))
        return asyncio.run(_verify(args.url))
    except psycopg.OperationalError as exc:
        msg = str(exc)
        print(f"\nERROR: could not connect — {msg.strip()}", file=sys.stderr)
        if "timeout" in msg.lower() or "timed out" in msg.lower():
            print("\nA timeout on a Supabase host almost always means one of:", file=sys.stderr)
            print("  * the direct connection on an IPv4-only network (use the session pooler)",
                  file=sys.stderr)
            print("  * the project is PAUSED — free projects pause after ~7 days idle;",
                  file=sys.stderr)
            print("    resume it from the Supabase dashboard.", file=sys.stderr)
        elif "password" in msg.lower() or "authentication" in msg.lower():
            print("\nCheck the password, and note the pooler username is"
                  " `postgres.<project-ref>`, not `postgres`.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

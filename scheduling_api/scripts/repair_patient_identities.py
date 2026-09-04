#!/usr/bin/env python3
"""Split patient rows that merged two people on one phone number. Dry-run by default.

    uv run python scripts/repair_patient_identities.py              # report only
    uv run python scripts/repair_patient_identities.py --apply      # write

`patients` used to be UNIQUE (clinic_id, phone), so the first caller from a number owned it
permanently. A second household member booking from the same handset overwrote `name` on that
row but could not overwrite `date_of_birth`, which left ONE row describing TWO people — and
owning both their bookings.

`db.migrate_patient_identity` re-keys the table so this cannot happen again, but it cannot
un-merge rows that already exist: nothing in `patients` records who the second person was.
`bookings` does. Every booking carries its own `patient_name` and `date_of_birth` as the caller
gave them, so a booking whose details disagree with its patient row is a booking that belongs
to somebody else, and this repoints it at a row that actually describes them.

Read the dry run before applying. It touches PHI ownership, which is the thing most worth
getting wrong quietly.
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


def _describe(url: str) -> str:
    """host/database, with the password stripped — safe to print."""
    from urllib.parse import urlparse

    u = urlparse(url)
    host = u.hostname or "?"
    port = f":{u.port}" if u.port else ""
    return f"{u.username or '?'}@{host}{port}{u.path or ''}"


async def run(url: str, apply: bool) -> int:
    print(f"target: {_describe(url)}")
    print(f"mode  : {'APPLY — this writes' if apply else 'dry run — read only'}\n")
    try:
        conn_ctx = await psycopg.AsyncConnection.connect(
            url, row_factory=dict_row, connect_timeout=15, autocommit=not apply
        )
    except psycopg.OperationalError as exc:
        print(f"ERROR: could not connect to {_describe(url)}\n\n  {exc}", file=sys.stderr)
        print(
            "\nThis script repairs who owns which appointment, so it will not guess at a\n"
            "database. Point it at the one the agent actually books against:\n\n"
            "  export CLINIC_DATABASE_URL='postgresql://postgres.<ref>:<password>"
            "@aws-0-us-east-1.pooler.supabase.com:5432/postgres'\n"
            "  uv run python scripts/repair_patient_identities.py\n\n"
            "The password is in the Supabase dashboard under\n"
            "Project Settings -> Database (\"Reset database password\" if you do not have it).\n"
            "Use the SESSION POOLER host above — db.<ref>.supabase.co is IPv6-only and will\n"
            "hang on an IPv4 network. See CLAUDE.md -> Running the services.",
            file=sys.stderr,
        )
        return 2

    async with conn_ctx as conn:
        counts = await db.migrate_patient_identity(conn)
        if counts["normalized"] or counts["merged"]:
            print(f"[re-key] normalized {counts['normalized']} date(s), "
                  f"merged {counts['merged']} duplicate row(s)")

        rows = await (await conn.execute(
            """
            SELECT b.confirmation_id, b.patient_id, b.patient_name, b.date_of_birth,
                   b.clinic_id, p.name AS row_name, p.date_of_birth AS row_dob, p.phone
              FROM bookings b
              JOIN patients p ON p.id = b.patient_id
             WHERE b.date_of_birth IS NOT NULL
             ORDER BY b.confirmation_id
            """
        )).fetchall()

        mismatched = [
            r for r in rows
            if db.normalize_dob(r["date_of_birth"]) != db.normalize_dob(r["row_dob"])
        ]
        if not mismatched:
            print("no merged identities found — nothing to repair")
            return 0

        print(f"\n{len(mismatched)} booking(s) attached to a patient row describing someone else:\n")
        for r in mismatched:
            print(f"  {r['confirmation_id']}  booked by {r['patient_name']!r} "
                  f"(DOB {r['date_of_birth']})")
            print(f"      currently owned by row #{r['patient_id']} {r['row_name']!r} "
                  f"(DOB {r['row_dob']}) on {r['phone']}")

        if not apply:
            print("\ndry run — nothing written. Re-run with --apply to repair.")
            return 0

        for r in mismatched:
            target = await (await conn.execute(
                """
                INSERT INTO patients (clinic_id, phone, name, date_of_birth)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (clinic_id, phone, date_of_birth) DO UPDATE
                    SET name = EXCLUDED.name
                RETURNING id
                """,
                (r["clinic_id"], r["phone"], r["patient_name"],
                 db.normalize_dob(r["date_of_birth"])),
            )).fetchone()
            await conn.execute(
                "UPDATE bookings SET patient_id = %s WHERE confirmation_id = %s",
                (target["id"], r["confirmation_id"]),
            )
            print(f"  {r['confirmation_id']} -> patient #{target['id']} ({r['patient_name']})")

        # The row the bookings came from may still carry the wrong name: it was overwritten by
        # whoever booked last. Restore it from a booking that genuinely belongs to it.
        for r in mismatched:
            owner = await (await conn.execute(
                """
                SELECT b.patient_name FROM bookings b
                 WHERE b.patient_id = %s
                   AND b.date_of_birth IS NOT NULL
                 ORDER BY b.created_at LIMIT 1
                """,
                (r["patient_id"],),
            )).fetchone()
            if owner:
                await conn.execute(
                    "UPDATE patients SET name = %s WHERE id = %s",
                    (owner["patient_name"], r["patient_id"]),
                )
                print(f"  row #{r['patient_id']} name restored to {owner['patient_name']!r}")

        await conn.commit()
        print("\nrepaired.")
        return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default=None,
                    help="database to repair (default: $CLINIC_DATABASE_URL)")
    ap.add_argument("--apply", action="store_true", help="write changes (default: dry run)")
    args = ap.parse_args()

    # Deliberately NOT falling back to db.DATABASE_URL. That default is a local dev database,
    # and silently repairing PHI ownership in the wrong place is the one outcome worth an extra
    # setup step to prevent. An unset variable is a question, not a default.
    url = args.url or os.getenv("CLINIC_DATABASE_URL")
    if not url:
        print(
            "ERROR: no database specified.\n\n"
            "  export CLINIC_DATABASE_URL='<the URL the agent books against>'\n"
            "  uv run python scripts/repair_patient_identities.py\n\n"
            "or pass --url. This script changes who owns which appointment, so it will not\n"
            "fall back to a default. See CLAUDE.md -> Running the services.",
            file=sys.stderr,
        )
        return 2
    return asyncio.run(run(url, args.apply))


if __name__ == "__main__":
    raise SystemExit(main())

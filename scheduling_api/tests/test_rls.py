"""Row-level security actually denies an unprivileged reader.

Enabling RLS is easy to get wrong in a way that tests never notice: the service connects as an
owner/superuser role, which BYPASSES RLS, so the whole suite passes identically whether the
protection works or not. These tests create a role that mimics Supabase's `anon` — granted
SELECT, no bypassrls — and assert it sees nothing.

Why this matters here specifically: Supabase exposes the `public` schema through its Data API,
so tables in it can be reachable by `anon` with the project's publishable key, which is shipped
to browsers by design. `bookings` holds patient_name, date_of_birth, and symptom_notes.
Un-protected, that is a PHI disclosure from a browser console.
"""

from __future__ import annotations

import uuid

import psycopg
import pytest

from app import db

# Mimics Supabase's anon: a login role with table SELECT granted but no RLS bypass.
#
# The name carries a per-session random suffix. Against a connection pooler (Supabase's
# Supavisor), a role name that has been dropped and recreated leaves pooled backends holding the
# old role's OID, and every later connection under that name fails with
#
#     psycopg.errors.UndefinedObject: invalid role OID: 20169
#
# which looks like a permissions bug and is really pool staleness that outlives the test run. A
# fresh name has never been pooled, so it is always clean.
_ANON_ROLE = f"clinic_test_anon_{uuid.uuid4().hex[:8]}"
_ANON_PASSWORD = "anon_test_only"

_PHI_TABLES = ("bookings", "patients", "call_summaries", "caller_memory")


def _anon_connection_url() -> str:
    """Build a connection URL for the unprivileged role, valid on local Postgres AND Supabase.

    Supabase's pooler (Supavisor) is multi-tenant and routes by a suffix on the username:
    `<role>.<project-ref>`. Connecting as a bare role name is rejected before authentication
    even happens, with:

        FATAL: (ENOIDENTIFIER) no tenant identifier provided

    which reads like an auth failure but is really a routing failure. So the tenant suffix from
    the service's own username has to be carried over to the test role. On a local server the
    username has no suffix and this is a no-op.
    """
    from urllib.parse import quote, urlparse

    parsed = urlparse(db.DATABASE_URL)
    service_user = parsed.username or ""
    tenant_suffix = f".{service_user.split('.', 1)[1]}" if "." in service_user else ""

    host = parsed.hostname
    port = f":{parsed.port}" if parsed.port else ""
    return (
        f"postgresql://{_ANON_ROLE}{tenant_suffix}:{quote(_ANON_PASSWORD, safe='')}"
        f"@{host}{port}{parsed.path}"
    )


@pytest.fixture
def anon_url(client):
    """Ensure the unprivileged role exists and yield a URL that connects as it.

    Depends on `client` so the schema exists and the app has run at least once.

    Deliberately CREATE-IF-ABSENT with no teardown drop, rather than recreating the role per
    test. Against a connection pooler (Supabase's Supavisor), dropping and recreating a role
    between tests leaves pooled backends holding the previous role's OID, and the next
    connection fails with:

        psycopg.errors.UndefinedObject: invalid role OID: 20169

    which looks like a permissions bug and is really pool staleness. Creating the role once and
    leaving it in place avoids the churn entirely — and it is also several seconds faster over a
    network connection.
    """
    with psycopg.connect(db.DATABASE_URL, autocommit=True) as conn:
        exists = conn.execute(
            "SELECT 1 FROM pg_roles WHERE rolname = %s", (_ANON_ROLE,)
        ).fetchone()
        if not exists:
            conn.execute(f"CREATE ROLE {_ANON_ROLE} LOGIN PASSWORD '{_ANON_PASSWORD}'")
        # Re-granted every time: harmless, and it covers tables created since the last run.
        conn.execute(f"GRANT USAGE ON SCHEMA public TO {_ANON_ROLE}")
        conn.execute(f"GRANT SELECT ON ALL TABLES IN SCHEMA public TO {_ANON_ROLE}")
        # Sweep roles left by earlier sessions so they do not accumulate on a real project.
        stale = conn.execute(
            "SELECT rolname FROM pg_roles WHERE rolname LIKE 'clinic_test_anon%%' "
            "AND rolname <> %s",
            (_ANON_ROLE,),
        ).fetchall()
        for (name,) in stale:
            conn.execute(f"REVOKE ALL ON ALL TABLES IN SCHEMA public FROM {name}")
            conn.execute(f"REVOKE USAGE ON SCHEMA public FROM {name}")
            conn.execute(f"DROP ROLE IF EXISTS {name}")

    yield _anon_connection_url()


def _book_an_appointment(client) -> int:
    slots = client.get("/availability").json()["slots"]
    slot_id = slots[0]["slot_id"]
    hold_id = client.post("/hold-slot", json={"slot_id": slot_id}).json()["hold_id"]
    resp = client.post(
        "/confirm-booking",
        json={
            "hold_id": hold_id,
            "patient_name": "Jane Doe",
            "reason": "checkup",
            "date_of_birth": "01/02/1990",
            "symptom_notes": "sore throat for two days",
        },
    )
    assert resp.status_code == 200
    return slot_id


def test_rls_is_enabled_on_every_table(client):
    """A table added to schema.sql but missed in the RLS block would be silently exposed."""
    with psycopg.connect(db.DATABASE_URL) as conn:
        unprotected = [
            r[0]
            for r in conn.execute(
                """
                SELECT tablename FROM pg_tables
                 WHERE schemaname = 'public' AND rowsecurity = false
                """
            ).fetchall()
        ]
    assert unprotected == [], f"tables without RLS are reachable via a Data API: {unprotected}"


def test_an_anon_role_cannot_read_patient_data(client, anon_url):
    """The actual protection: PHI must be invisible to a role that doesn't bypass RLS."""
    _book_an_appointment(client)

    # Sanity check: the row genuinely exists when read as the service.
    with psycopg.connect(db.DATABASE_URL) as conn:
        assert conn.execute("SELECT count(*) FROM bookings").fetchone()[0] == 1

    with psycopg.connect(anon_url) as anon:
        for table in _PHI_TABLES:
            count = anon.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            assert count == 0, f"{table} leaked {count} row(s) to an unprivileged role"


def test_an_anon_role_cannot_read_appointment_state(client, anon_url):
    """Slots and providers are less sensitive but still shouldn't be world-readable."""
    _book_an_appointment(client)

    with psycopg.connect(anon_url) as anon:
        assert anon.execute("SELECT count(*) FROM slots").fetchone()[0] == 0
        assert anon.execute("SELECT count(*) FROM clinics").fetchone()[0] == 0


def test_an_anon_role_cannot_write(client, anon_url):
    """RLS blocks reads; the absence of an INSERT grant blocks writes."""
    with psycopg.connect(anon_url) as anon:
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            anon.execute(
                "INSERT INTO clinics (slug, name) VALUES ('evil', 'Evil Clinic')"
            )


def test_the_service_role_still_sees_everything(client):
    """Guards the other failure mode: RLS so strict the application itself is blinded."""
    slot_id = _book_an_appointment(client)

    with psycopg.connect(db.DATABASE_URL) as conn:
        row = conn.execute(
            "SELECT patient_name FROM bookings WHERE slot_id = %s", (slot_id,)
        ).fetchone()
    assert row is not None and row[0] == "Jane Doe"

    # And through the API, which is what actually matters.
    assert client.get("/availability").status_code == 200

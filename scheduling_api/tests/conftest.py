"""Test fixtures: point the API at a throwaway Postgres database and reset it per test.

Needs a running Postgres. Set CLINIC_TEST_DATABASE_URL, or start a disposable local cluster:

    initdb -D /tmp/pgclinic -U postgres --auth=trust
    pg_ctl -D /tmp/pgclinic -o "-p 55432 -c listen_addresses=127.0.0.1 \\
        -c unix_socket_directories=''" -l /tmp/pgclinic.log start
    createdb -h 127.0.0.1 -p 55432 -U postgres clinic_test

Tests are skipped (not failed) when no database is reachable, so the suite stays runnable on a
machine without Postgres — but note that skipping means the concurrency guarantees in
test_concurrency.py go unverified, which is the whole point of Phase 9. CI must provide one.
"""

import os

# Must be set BEFORE importing app.db, which reads the URL at import time.
os.environ.setdefault(
    "CLINIC_DATABASE_URL",
    os.getenv("CLINIC_TEST_DATABASE_URL", "postgresql://postgres@127.0.0.1:55432/clinic_test"),
)
# Sweep rarely in tests: the sweeper is housekeeping, and a tight loop just adds noise and
# contention to assertions about hold state.
os.environ.setdefault("CLINIC_SWEEP_INTERVAL_SECONDS", "3600")

import psycopg  # noqa: E402
import pytest  # noqa: E402
import pytest_asyncio  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app import db  # noqa: E402
from app.main import app  # noqa: E402

# Order matters: children before parents, so FK references are gone before the referenced rows.
_TABLES = (
    "audit_log",
    "staff_tasks",
    "clinic_facts",
    "call_summaries",
    "caller_memory",
    "idempotency_keys",
    "bookings",
    "slots",
    "patients",
    "providers",
    "clinics",
)


def _database_available() -> bool:
    try:
        with psycopg.connect(db.DATABASE_URL, connect_timeout=3) as conn:
            conn.execute("SELECT 1")
        return True
    except Exception:  # noqa: BLE001
        return False


_DB_UP = _database_available()
requires_db = pytest.mark.skipif(_DB_UP is False, reason=f"no Postgres at {db.DATABASE_URL}")


def _truncate_all() -> None:
    """Wipe every table between tests.

    TRUNCATE ... CASCADE rather than dropping the schema: it is far faster, and it keeps the
    identity sequences and constraints under test rather than recreating them each time.
    """
    with psycopg.connect(db.DATABASE_URL) as conn:
        existing = {
            r[0]
            for r in conn.execute(
                "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
            ).fetchall()
        }
        targets = [t for t in _TABLES if t in existing]
        if targets:
            conn.execute(f"TRUNCATE {', '.join(targets)} RESTART IDENTITY CASCADE")
        conn.commit()


@pytest.fixture
def client():
    """Fresh DB + TestClient per test. TestClient's context manager runs the lifespan."""
    if not _DB_UP:
        pytest.skip(f"no Postgres at {db.DATABASE_URL}")
    _truncate_all()
    with TestClient(app) as c:
        yield c


@pytest_asyncio.fixture
async def pool():
    """A live connection pool for tests that exercise db.py directly (concurrency tests).

    Bypasses the HTTP layer so a race can be driven with real concurrent coroutines rather than
    through TestClient, which serialises requests.
    """
    if not _DB_UP:
        pytest.skip(f"no Postgres at {db.DATABASE_URL}")
    _truncate_all()
    await db.open_pool()
    await db.init_db()
    try:
        yield db.pool()
    finally:
        await db.close_pool()

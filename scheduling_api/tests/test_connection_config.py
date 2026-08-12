"""Connection configuration — the Supabase/pooler traps.

These are offline (no database needed): they check how a connection string is INTERPRETED, not
what it connects to. That matters because the failure they guard against is delayed. psycopg3
starts preparing a statement only after `prepare_threshold` executions, and Supavisor's
transaction pooler rejects prepared statements — so a misconfigured deployment works perfectly
through the first few requests and then starts failing on a hot query. A test that only checked
"can we connect?" would pass and tell you nothing.
"""

from __future__ import annotations

import importlib

import pytest

from app import db


def _threshold_for(url: str, monkeypatch, override: str | None = None) -> int | None:
    """Reload app.db with a given URL — DATABASE_URL is read at import time."""
    monkeypatch.setenv("CLINIC_DATABASE_URL", url)
    if override is None:
        monkeypatch.delenv("CLINIC_DB_PREPARE_THRESHOLD", raising=False)
    else:
        monkeypatch.setenv("CLINIC_DB_PREPARE_THRESHOLD", override)
    reloaded = importlib.reload(db)
    try:
        return reloaded._prepare_threshold()
    finally:
        monkeypatch.undo()
        importlib.reload(db)


# =========================================================================================
# Transaction pooler: prepared statements must be OFF
# =========================================================================================


def test_transaction_pooler_disables_prepared_statements(monkeypatch):
    """Port 6543 is Supavisor transaction mode, which rejects prepared statements."""
    url = "postgresql://postgres.abc:pw@aws-0-us-east-1.pooler.supabase.com:6543/postgres"
    assert _threshold_for(url, monkeypatch) is None


def test_session_pooler_keeps_prepared_statements(monkeypatch):
    """Port 5432 on the pooler host is SESSION mode, which supports prepared statements.

    The distinction is only the port, which is exactly why this is easy to get wrong.
    """
    url = "postgresql://postgres.abc:pw@aws-0-us-east-1.pooler.supabase.com:5432/postgres"
    assert _threshold_for(url, monkeypatch) == 5


def test_direct_connection_keeps_prepared_statements(monkeypatch):
    url = "postgresql://postgres:pw@db.abcdefghijklmnop.supabase.co:5432/postgres"
    assert _threshold_for(url, monkeypatch) == 5


def test_local_postgres_keeps_prepared_statements(monkeypatch):
    assert _threshold_for("postgresql://postgres@127.0.0.1:55432/clinic_dev", monkeypatch) == 5


def test_port_6543_is_detected_without_a_trailing_database_name(monkeypatch):
    """Guards the string match: a URL can legitimately end at the port."""
    assert _threshold_for("postgresql://u:p@host.pooler.supabase.com:6543", monkeypatch) is None


def test_a_database_named_like_the_port_does_not_false_positive(monkeypatch):
    """`:6543/` must match a port, not a database that happens to contain those digits."""
    url = "postgresql://postgres@127.0.0.1:5432/db6543"
    assert _threshold_for(url, monkeypatch) == 5


# =========================================================================================
# Override
# =========================================================================================


def test_explicit_override_disables_preparation(monkeypatch):
    url = "postgresql://postgres@127.0.0.1:5432/clinic_dev"
    for value in ("none", "off", "NONE"):
        assert _threshold_for(url, monkeypatch, override=value) is None


def test_explicit_override_sets_a_numeric_threshold(monkeypatch):
    url = "postgresql://postgres@127.0.0.1:5432/clinic_dev"
    assert _threshold_for(url, monkeypatch, override="1") == 1


def test_override_wins_over_pooler_autodetection(monkeypatch):
    """An escape hatch is only useful if it actually overrides the automatic behaviour."""
    url = "postgresql://postgres.abc:pw@aws-0-us-east-1.pooler.supabase.com:6543/postgres"
    assert _threshold_for(url, monkeypatch, override="10") == 10


# =========================================================================================
# Portability: nothing here should require a bleeding-edge Postgres
# =========================================================================================


def test_schema_uses_no_postgres_18_only_features():
    """Supabase runs Postgres 15-17; local dev here happens to be 18.

    A schema that only works on 18 would apply cleanly in development and fail on deploy, so
    this pins the features actually used to things available in PG 13+.
    """
    schema = (db._SCHEMA_PATH).read_text().lower()

    # Features the schema relies on, all long-established.
    assert "generated always as identity" in schema  # PG 10+
    assert "timestamptz" in schema
    assert "jsonb" in schema  # PG 9.4+

    # Things that would raise the required version.
    for pg18ism in ("uuidv7(", "virtual generated", "temporal", "without overlaps"):
        assert pg18ism not in schema, f"schema uses a Postgres-18-only feature: {pg18ism}"

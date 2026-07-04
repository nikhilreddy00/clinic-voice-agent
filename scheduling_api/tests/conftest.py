"""Test fixtures. Point the API at a throwaway SQLite DB and reset it per test."""

import os
import tempfile

# Must be set BEFORE importing app.db (which reads the path at import time).
os.environ["CLINIC_DB_PATH"] = os.path.join(tempfile.mkdtemp(), "test_clinic.db")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app import db  # noqa: E402
from app.main import app  # noqa: E402


@pytest.fixture
def client():
    """Fresh DB + TestClient for each test."""
    if os.path.exists(db.DB_PATH):
        os.remove(db.DB_PATH)
    db.init_db()
    with TestClient(app) as c:
        yield c

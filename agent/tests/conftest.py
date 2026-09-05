"""Test-session guards.

`clinic_agent.config` calls `load_dotenv()` at import, which is right for the agent and wrong
for a test run: the moment a real `CLINIC_OTEL_ENDPOINT` landed in `agent/.env`, every
`CallSession` in the suite began shipping spans to Grafana at teardown. The suite went from 5.0s
to 17.2s and started depending on a network and on one developer's credentials.

Nothing here should reach a third party. Unsetting the endpoint is enough — `otel_enabled()`
is the single switch, and with it off the SDK is never even imported.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True, scope="session")
def _no_telemetry_from_tests():
    import os

    saved = {k: os.environ.pop(k, None) for k in ("CLINIC_OTEL_ENDPOINT", "CLINIC_OTEL_HEADERS")}
    yield
    for key, value in saved.items():
        if value is not None:
            os.environ[key] = value

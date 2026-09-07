"""Phase 17 — the agent's half of multi-tenancy.

The API tests prove a tenant cannot see another tenant's data. These prove the agent actually
NAMES a tenant: the right clinic in the greeting and the prompt, the right zone to reason about
"tomorrow" in, the right number to escalate to, and the right header on every request.

The thing most worth pinning is the failure mode. A tenant lookup happens at worker startup and
must degrade — an unreachable API, an unknown DID, no number configured at all — to the default
clinic with a warning. A colder greeting is a bad call; a call that never connects is worse.
"""

from __future__ import annotations

import httpx
import pytest

from clinic_agent import prompts, tenant
from clinic_agent.core import events as ev
from clinic_agent.core.actions import Speak
from clinic_agent.core.reducer import reduce
from clinic_agent.core.state import CallState
from clinic_agent.scheduling_tools import SchedulingClient

BAYSIDE = tenant.Tenant(
    slug="bayside-health",
    name="Bayside Health Partners",
    timezone="America/Chicago",
    transfer_number="+18885550100",
)


@pytest.fixture(autouse=True)
def _restore_default():
    yield
    tenant.set_current(tenant.DEFAULT)


# --- what the caller hears ------------------------------------------------------------------


def test_the_greeting_names_the_tenant_but_never_varies_the_disclosure():
    tenant.set_current(BAYSIDE)
    greeting = prompts.greeting_for("telephony")
    assert "Bayside Health Partners" in greeting
    assert "Grove" not in greeting
    # A clinic does not get to configure whether its callers are told they are talking to an AI.
    assert prompts.AI_DISCLOSURE in greeting
    assert prompts.RECORDING_CONSENT in greeting


def test_the_greeting_comes_from_the_TRACE_not_from_process_state():
    """`reduce` is pure: a trace recorded at one clinic must replay to that clinic's greeting on
    a machine configured for another. The name travels on CallStarted for exactly this."""
    state = CallState()
    state, _ = reduce(state, ev.CallStarted(call_id="c1", mode="telephony",
                                            clinic_name="Bayside Health Partners"))
    tenant.set_current(tenant.DEFAULT)  # the replaying machine is a Grove worker
    _, actions = reduce(state, ev.CallerPresent(participant_id="sip_1", phone="+15551234321"))

    (speak,) = [a for a in actions if isinstance(a, Speak)]
    assert "Bayside Health Partners" in speak.text


def test_a_pre_tenancy_trace_still_replays_to_the_original_greeting():
    """Old traces carry no clinic_name. They must not start saying something new."""
    state, _ = reduce(CallState(), ev.CallStarted(call_id="c1", mode="local"))
    _, actions = reduce(state, ev.CallerPresent(participant_id="local-mic"))
    (speak,) = [a for a in actions if isinstance(a, Speak)]
    assert speak.text == prompts.GREETING


# --- what the model is told ------------------------------------------------------------------


def test_the_prompt_carries_the_tenants_name_and_clock():
    tenant.set_current(BAYSIDE)
    prompt = prompts.build_system_prompt()
    assert "Bayside Health Partners" in prompt
    assert "America/Chicago" in prompt
    assert "Grove" not in prompt


def test_the_date_table_is_built_in_the_tenants_zone():
    """A clinic in Central time reasoning off Eastern dates is the Phase-4 date-grounding
    defect with a tenant column added. Near midnight the two zones disagree on today."""
    from datetime import datetime, timezone

    # 04:30 UTC — 11:30 PM in Chicago, 12:30 AM in New York. Different calendar days.
    now = datetime(2026, 9, 9, 4, 30, tzinfo=timezone.utc)

    tenant.set_current(BAYSIDE)
    central = prompts.build_system_prompt(now=now)
    tenant.set_current(tenant.DEFAULT)
    eastern = prompts.build_system_prompt(now=now)

    assert "Today's date is 2026-09-08" in central
    assert "Today's date is 2026-09-09" in eastern


# --- what goes on the wire -------------------------------------------------------------------


def test_every_request_carries_the_tenant_and_the_call_id():
    tenant.set_current(BAYSIDE)
    client = SchedulingClient("http://example.invalid", call_id="call-7")
    headers = client._client.headers
    assert headers["X-Clinic-Slug"] == "bayside-health"
    assert headers["X-Call-Id"] == "call-7"


def test_an_explicit_slug_beats_the_process_tenant():
    client = SchedulingClient("http://example.invalid", clinic_slug="grove-family")
    assert client._client.headers["X-Clinic-Slug"] == "grove-family"


# --- resolution, and every way it can fail ----------------------------------------------------


@pytest.mark.asyncio
async def test_a_dialed_number_resolves_to_its_tenant(monkeypatch):
    body = {"slug": "bayside-health", "name": "Bayside Health Partners",
            "timezone": "America/Chicago", "did": "+18885550142",
            "transfer_number": "+18885550100"}
    monkeypatch.setattr(httpx.AsyncClient, "get",
                        lambda self, url, **kw: _response(200, body))

    resolved = await tenant.load("http://api.invalid", "+18885550142")
    assert resolved == BAYSIDE
    assert tenant.current() == BAYSIDE


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [
    httpx.ConnectError("refused"),
    httpx.ReadTimeout("slow"),
])
async def test_an_unreachable_api_serves_the_default_clinic(monkeypatch, failure):
    def boom(self, url, **kw):
        raise failure
    monkeypatch.setattr(httpx.AsyncClient, "get", boom)

    assert await tenant.load("http://api.invalid", "+18885550142") == tenant.DEFAULT


@pytest.mark.asyncio
async def test_an_unknown_number_serves_the_default_clinic(monkeypatch):
    monkeypatch.setattr(httpx.AsyncClient, "get", lambda self, url, **kw: _response(404, {}))
    assert await tenant.load("http://api.invalid", "+19998887777") == tenant.DEFAULT


@pytest.mark.asyncio
async def test_no_number_configured_does_not_even_call_the_api(monkeypatch):
    def boom(self, url, **kw):  # pragma: no cover - must not run
        raise AssertionError("looked up a tenant with no DID and no CLINIC_SLUG")
    monkeypatch.setattr(httpx.AsyncClient, "get", boom)
    monkeypatch.delenv("CLINIC_SLUG", raising=False)
    assert await tenant.load("http://api.invalid", "") == tenant.DEFAULT


def _response(status: int, body: dict):
    async def _await_me():
        return httpx.Response(status, json=body, request=httpx.Request("GET", "http://x"))
    return _await_me()

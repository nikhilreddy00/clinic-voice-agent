"""Phase 15 — the warm transfer, up to (but not including) the PSTN leg.

What is testable offline is everything that has ever gone wrong here: whether a transfer is
attempted at all, what request it builds, what it does when there is nowhere to send the
caller, and — the one that matters most — that the caller is told before their line moves.

The actual PSTN leg needs a phone and a destination number and is the single part of this
phase that a live call has to prove.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from clinic_agent.config import Settings
from clinic_agent.core import events as ev
from clinic_agent.core.transfer import SIPTransfer, _masked, _tel_uri


def _settings(**overrides) -> Settings:
    base = Settings(
        mode="telephony",
        deepgram_api_key="d",
        anthropic_api_key="a",
        anthropic_model="claude-haiku-4-5",
        groq_api_key="",
        groq_model="",
        cartesia_api_key="c",
        cartesia_voice_id="v",
        livekit_url="wss://lk.test",
        livekit_api_key="key",
        livekit_api_secret="secret",
        livekit_phone_number="+14842951203",
        scheduling_api_base_url="http://127.0.0.1:8000",
        transfer_number="+15551234567",
    )
    return replace(base, **overrides)


class _FakeAPI:
    """Stands in for livekit_api.LiveKitAPI as an async context manager."""

    last_request = None
    raised = None

    def __init__(self, *args, **kwargs):
        self.sip = self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def transfer_sip_participant(self, request):
        if _FakeAPI.raised is not None:
            raise _FakeAPI.raised
        _FakeAPI.last_request = request
        return object()


@pytest.fixture(autouse=True)
def _fake_livekit(monkeypatch):
    from clinic_agent.core import transfer as transfer_mod

    _FakeAPI.last_request = None
    _FakeAPI.raised = None
    monkeypatch.setattr(transfer_mod.livekit_api, "LiveKitAPI", _FakeAPI)
    yield


def _emitted() -> list:
    return []


@pytest.mark.asyncio
async def test_a_configured_transfer_builds_the_request_livekit_expects():
    emitted = _emitted()
    t = SIPTransfer(_settings(), "clinic-inbound", emitted.append)

    await t.transfer("sip_15551110000", reason="llm_unavailable", summary="reason=x", urgent=False)

    request = _FakeAPI.last_request
    assert request.participant_identity == "sip_15551110000"
    assert request.room_name == "clinic-inbound"
    assert request.transfer_to == "tel:+15551234567"
    assert request.play_dialtone is True, "silence while the leg moves reads as a dropped call"
    assert emitted == [], "a successful transfer produces no fallback"


@pytest.mark.asyncio
async def test_no_destination_configured_falls_back_instead_of_failing_silently():
    """The default state of this repo. The caller must still hear the callback promise."""
    emitted = _emitted()
    t = SIPTransfer(_settings(transfer_number=""), "clinic-inbound", emitted.append)

    await t.transfer("sip_1", reason="no_match", summary="", urgent=False)

    assert _FakeAPI.last_request is None
    assert isinstance(emitted[0], ev.TransferFailed)
    assert "CLINIC_TRANSFER_NUMBER" in emitted[0].reason


@pytest.mark.asyncio
async def test_the_local_path_has_no_sip_leg_to_transfer():
    emitted = _emitted()
    t = SIPTransfer(_settings(mode="local"), "clinic-inbound", emitted.append)

    await t.transfer("local-mic", reason="caller_requested_human", summary="", urgent=False)

    assert _FakeAPI.last_request is None
    assert isinstance(emitted[0], ev.TransferFailed)


@pytest.mark.asyncio
async def test_a_refused_transfer_is_reported_not_raised():
    """A provider error here must not take the call down — the caller is still on the line."""
    emitted = _emitted()
    _FakeAPI.raised = RuntimeError("twirp error not_found: participant is gone")
    t = SIPTransfer(_settings(), "clinic-inbound", emitted.append)

    await t.transfer("sip_1", reason="stt_unavailable", summary="", urgent=True)

    assert isinstance(emitted[0], ev.TransferFailed)
    assert "not_found" in emitted[0].reason


@pytest.mark.asyncio
async def test_a_missing_participant_identity_falls_back():
    emitted = _emitted()
    t = SIPTransfer(_settings(), "clinic-inbound", emitted.append)

    await t.transfer("", reason="no_match", summary="", urgent=False)

    assert _FakeAPI.last_request is None and isinstance(emitted[0], ev.TransferFailed)


@pytest.mark.parametrize(
    "configured, expected",
    [("+15551234567", "tel:+15551234567"), ("sip:desk@clinic.example", "sip:desk@clinic.example")],
)
def test_both_a_number_and_a_sip_uri_are_accepted(configured, expected):
    assert _tel_uri(configured) == expected


def test_the_destination_is_masked_in_logs():
    assert _masked("+15551234567") == "***4567"


# --- the ordering property, end to end through the session --------------------------------


@pytest.mark.asyncio
async def test_the_session_transfers_the_caller_livekit_identified(tmp_path, monkeypatch):
    """`caller_identity` comes off CallerPresent and is what moves the right leg.

    Getting this wrong on a worker hosting N calls would transfer the wrong caller, which is
    why the identity travels on the event rather than being looked up.
    """
    monkeypatch.setenv("CLINIC_LOG_DIR", str(tmp_path))
    from clinic_agent.core.actions import TransferToHuman
    from test_session import ScriptedSession

    session = ScriptedSession(script=[], tool_results={}, record=False)
    session.state = replace(session.state, caller_identity="sip_15559998888")
    captured: dict = {}

    async def _fake_transfer(identity, *, reason, summary, urgent):
        captured.update(identity=identity, reason=reason, urgent=urgent)

    session._transfer.transfer = _fake_transfer
    await session._execute(
        TransferToHuman(reason="no_match", summary="reason=no_match", urgent=False)
    )

    assert captured["identity"] == "sip_15559998888"
    assert captured["reason"] == "no_match"

"""Phase 15 — the warm transfer, for real.

Through Phase 14 :class:`~clinic_agent.core.actions.TransferToHuman` was a log line reading
``NOT IMPLEMENTED``. Three separate paths reached it — the emergency script, a caller asking
for a person, and an empty availability window — and all three ended with someone being told
that help was coming and then reaching nobody. That is the worst failure in the system: it is
the one the caller believes.

What this module does is small on purpose: LiveKit already owns the SIP leg, so transferring is
one REST call against the room the caller is in. What it does *not* do is decide anything — the
reducer decides, having already spoken the hand-off line, and this either performs the transfer
or reports :class:`~clinic_agent.core.events.TransferFailed` so the reducer can promise the
caller a callback instead of dropping them.

**Nothing here speaks.** The line the caller hears is a deterministic ``Speak`` action from the
reducer, played before this runs, for the same reason the greeting is: the wording of a hand-off
is governance, and the model is often the thing that just failed.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from loguru import logger

from livekit import api as livekit_api

from ..config import Settings
from . import events as ev

EmitFn = Callable[[ev.Event], None]


class SIPTransfer:
    """Moves the caller's SIP leg to a human, or says plainly that it could not.

    Constructed per call but holds nothing per call — the identity comes in on the transfer,
    so a worker hosting N sessions could share one of these. Kept per-session for now because
    the LiveKit client it builds is short-lived and the transfer happens at most once a call.
    """

    def __init__(self, settings: Settings, room_name: str, emit: EmitFn) -> None:
        self._settings = settings
        self._room_name = room_name
        self._emit = emit

    @property
    def destination(self) -> str:
        return (self._settings.transfer_number or "").strip()

    def _unavailable(self, reason: str) -> None:
        """Report, at WARNING, and hand the reducer the fallback.

        Loud on purpose: a transfer that silently does nothing looks in the logs exactly like
        one that worked, which is how the `NOT IMPLEMENTED` branch survived four phases.
        """
        logger.warning(f"[transfer] not performed — {reason}")
        self._emit(ev.TransferFailed(t=time.monotonic(), reason=reason))

    async def transfer(self, identity: str, *, reason: str, summary: str, urgent: bool) -> None:
        """Attempt the hand-off. Never raises: a failed transfer is a dialogue event."""
        # The summary is the routing note a person picks the call up with. It carries no name,
        # no date of birth and no clinical text — see reducer._transfer_summary — precisely
        # because it lands in logs and traces.
        logger.info(
            f"[transfer] handing off (urgent={urgent}) reason={reason!r} — {summary}"
        )

        if not self.destination:
            return self._unavailable("CLINIC_TRANSFER_NUMBER is not set")
        if self._settings.mode != "telephony" or not identity:
            # There is no SIP leg on the local path. Saying so beats pretending.
            return self._unavailable(f"no SIP participant to transfer (mode={self._settings.mode})")

        request = livekit_api.TransferSIPParticipantRequest(
            participant_identity=identity,
            room_name=self._room_name,
            transfer_to=_tel_uri(self.destination),
            # The caller hears ringing rather than silence while the leg is set up. Their last
            # impression of this system should not be a pause they read as a dropped call.
            play_dialtone=True,
        )
        try:
            async with livekit_api.LiveKitAPI(
                self._settings.livekit_url,
                self._settings.livekit_api_key,
                self._settings.livekit_api_secret,
            ) as lk:
                await lk.sip.transfer_sip_participant(request)
        except Exception as exc:  # noqa: BLE001 - the caller must still be told something
            return self._unavailable(f"livekit refused the transfer ({exc})")

        logger.info(f"[transfer] caller {identity} transferred to {_masked(self.destination)}")


def _tel_uri(number: str) -> str:
    """LiveKit wants a URI. A bare number is the common way to configure this, so accept both.

    >>> _tel_uri("+15551234567")
    'tel:+15551234567'
    >>> _tel_uri("sip:desk@clinic.example")
    'sip:desk@clinic.example'
    """
    return number if ":" in number else f"tel:{number}"


def _masked(number: str) -> str:
    """Log destinations the way the rest of this codebase logs numbers."""
    return f"***{number[-4:]}" if len(number) > 4 else "***"

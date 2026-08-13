"""Entrypoint for the Phase-10 in-house event loop: ``python -m clinic_agent.core``.

``MODE`` is the only switch, exactly as before: ``local`` runs the laptop mic/speaker,
``telephony`` joins the LiveKit SIP room and waits for an inbound call. The dialogue, tools,
prompts, mic gate, and barge-in thresholds are identical to the Pipecat path — which is the
point, since "live call behavior matches, with equal-or-better latency" is how this phase is
judged.

``clinic_agent.pipeline`` (Pipecat) is unchanged and still runnable. Keeping both means a
problem in the new engine costs a one-word command change, not a live phone number.
"""

from __future__ import annotations

import asyncio
import os
import sys

from loguru import logger

from ..config import load_settings, require_phase1_keys, require_telephony_keys
from .session import CallSession


def _configure_logging() -> None:
    """Route logging off the audio-critical event loop.

    Synchronous stderr writes on the loop that also has to deliver 20 ms of audio on time
    produce output underruns — audible cutouts. ``enqueue=True`` hands log records to a
    background thread instead.
    """
    level = os.getenv("CLINIC_LOG_LEVEL", "INFO").upper()
    logger.remove()
    logger.add(sys.stderr, level=level, enqueue=True)


async def run() -> None:
    _configure_logging()

    settings = load_settings()
    require_phase1_keys(settings)  # fail fast on missing keys before opening the mic
    if settings.mode == "telephony":
        require_telephony_keys(settings)

    session = CallSession(settings)
    logger.info(
        "Starting clinic voice agent (Phase 10: in-house event loop). "
        + (
            "Waiting for an inbound call; Ctrl-C to stop."
            if settings.mode == "telephony"
            else "Speak into your mic; Ctrl-C to stop."
        )
    )
    try:
        await session.run()
    except asyncio.CancelledError:
        session.hangup("cancelled")
        raise


def main() -> None:
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        logger.info("[session] interrupted — shutting down")


if __name__ == "__main__":
    main()

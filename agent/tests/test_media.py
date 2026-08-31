"""Playback completion — the event that reopens the caller's microphone.

`BotStoppedSpeaking` is not bookkeeping. The mic gate is closed while the bot speaks, so an
utterance that never finishes means the agent goes permanently deaf: the caller talks, nothing
reaches STT, and the call is over without a single error in any log. That happened live.
"""

from __future__ import annotations

import asyncio

import pytest

from clinic_agent.core import events as ev
from clinic_agent.core.adapters.media import MediaAdapter


class FakeMedia(MediaAdapter):
    """A media adapter with an instant device, so tests exercise the loop, not the hardware."""

    STALL_TIMEOUT_S = 0.2

    def __init__(self, events: list) -> None:
        super().__init__(emit=events.append, on_audio=None)
        self.written = bytearray()

    async def _write(self, pcm: bytes) -> None:
        self.written.extend(pcm)

    async def _settle(self) -> None:
        """Let the playback loop run to quiescence."""
        for _ in range(20):
            await asyncio.sleep(0)


def _stops(events, utterance_id: str) -> list:
    return [e for e in events
            if isinstance(e, ev.BotStoppedSpeaking) and e.utterance_id == utterance_id]


@pytest.mark.asyncio
async def test_a_short_utterance_still_ends():
    """The live deadlock, reproduced. One chunk, written before Cartesia says `done`.

    The loop had already gone back to blocking on an empty queue, and end_utterance's old
    `if self._speaking is None` test then did nothing at all. The mic never reopened.
    """
    events: list = []
    media = FakeMedia(events)
    await media.start()
    try:
        await media.play("utt-3", b"\x00" * 640)
        await media._settle()                      # the chunk is written; loop is now idle
        await media.end_utterance("utt-3")         # ...and only THEN does synthesis finish
        await media._settle()

        assert _stops(events, "utt-3"), "the bot never stopped speaking — the call goes deaf"
        assert _stops(events, "utt-3")[0].completed is True
    finally:
        await media.aclose()


@pytest.mark.asyncio
async def test_an_utterance_that_never_gets_audio_still_ends():
    """Synthesis that produces nothing at all must not strand the call in SPEAKING."""
    events: list = []
    media = FakeMedia(events)
    await media.start()
    try:
        await media.end_utterance("utt-9")
        await media._settle()
        assert _stops(events, "utt-9")
    finally:
        await media.aclose()


@pytest.mark.asyncio
async def test_a_provider_that_never_finishes_cannot_hold_the_mic_shut():
    """The backstop for the other half: audio arrives, `done` never does."""
    events: list = []
    media = FakeMedia(events)
    await media.start()
    try:
        await media.play("utt-4", b"\x00" * 640)
        await media._settle()
        assert not _stops(events, "utt-4"), "ended before the stall timeout"

        await asyncio.sleep(media.STALL_TIMEOUT_S * 1.5)
        stops = _stops(events, "utt-4")
        assert stops, "a stalled provider left the microphone closed forever"
        assert stops[0].completed is False, "a stall is not a completed utterance"
    finally:
        await media.aclose()


@pytest.mark.asyncio
async def test_a_multi_sentence_utterance_ends_once_after_the_last_chunk():
    events: list = []
    media = FakeMedia(events)
    await media.start()
    try:
        await media.play("utt-5", b"\x01" * 640)
        await media.play("utt-5", b"\x02" * 640)
        await media.end_utterance("utt-5")
        await media._settle()

        starts = [e for e in events if isinstance(e, ev.BotStartedSpeaking)]
        assert len(starts) == 1 and len(_stops(events, "utt-5")) == 1
        assert len(media.written) == 1280, "audio was dropped"
    finally:
        await media.aclose()


@pytest.mark.asyncio
async def test_barge_in_still_ends_the_utterance_as_interrupted():
    events: list = []
    media = FakeMedia(events)
    await media.start()
    try:
        await media.play("utt-6", b"\x00" * 640)
        await media._settle()
        await media.clear("utt-6")
        await media._settle()

        stops = _stops(events, "utt-6")
        assert stops and stops[0].completed is False
    finally:
        await media.aclose()

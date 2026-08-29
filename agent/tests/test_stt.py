"""Phase 10 — DeepgramSTT socket lifecycle.

``test_session`` runs the whole loop with a *fake* STT, so nothing there exercises the real
socket. This covers the part that only the real adapter has: staying connected across the
silence windows a phone call is full of.

The bug this file exists for: Deepgram closes a stream that has received no audio for 10 s
(``net0001``). Two ordinary windows exceed that — waiting for an inbound SIP call, and any bot
utterance longer than 10 s, since the mic gate drops all input while the bot speaks. On the
first live call through this engine the socket died 12 s after start-up, 32 s before the caller
arrived, and the agent ran the whole call deaf.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from clinic_agent.core import events as ev
from clinic_agent.core.adapters import stt as stt_mod


class FakeWS:
    """Just enough websocket: records sends, never yields an inbound message."""

    def __init__(self) -> None:
        self.sent: list = []
        self.closed = False
        self._blocked = asyncio.Event()

    async def send(self, data) -> None:
        self.sent.append(data)

    def __aiter__(self):
        return self

    async def __anext__(self):
        await self._blocked.wait()  # no inbound traffic; the receive task is cancelled at close
        raise StopAsyncIteration

    async def close(self) -> None:
        self.closed = True

    # --- assertions ---------------------------------------------------------------------

    def text_messages(self) -> list[dict]:
        return [json.loads(m) for m in self.sent if isinstance(m, str)]

    def keepalives(self) -> int:
        return sum(1 for m in self.text_messages() if m.get("type") == "KeepAlive")


@pytest.fixture()
def fake_ws(monkeypatch) -> FakeWS:
    ws = FakeWS()

    async def _connect(*args, **kwargs):
        return ws

    monkeypatch.setattr(stt_mod.websockets, "connect", _connect)
    return ws


def build(events: list, interval: float = 0.02) -> stt_mod.DeepgramSTT:
    adapter = stt_mod.DeepgramSTT(api_key="test", emit=events.append)
    adapter.KEEPALIVE_INTERVAL_S = interval  # instance attr shadows the class default
    return adapter


@pytest.mark.asyncio
async def test_keepalive_holds_the_socket_open_through_silence(fake_ws):
    """The regression: an idle stream must be kept alive, not left to time out."""
    events: list[ev.Event] = []
    adapter = build(events)
    await adapter.start()
    try:
        await asyncio.sleep(0.11)  # ~5 intervals of pure silence
    finally:
        await adapter.aclose()

    assert fake_ws.keepalives() >= 3, (
        f"expected repeated KeepAlive frames during silence, got {fake_ws.text_messages()}"
    )
    assert not [e for e in events if isinstance(e, ev.ProviderDegraded)]


@pytest.mark.asyncio
async def test_audio_defers_the_keepalive(fake_ws):
    """Audio already resets Deepgram's timer — a keepalive on top is pure noise."""
    events: list[ev.Event] = []
    adapter = build(events, interval=0.05)
    await adapter.start()
    try:
        for _ in range(6):
            await adapter.send_audio(b"\x00\x00" * 160)
            await asyncio.sleep(0.01)
    finally:
        await adapter.aclose()

    assert fake_ws.keepalives() == 0, "keepalive fired while audio was flowing"
    assert sum(1 for m in fake_ws.sent if isinstance(m, bytes)) == 6


@pytest.mark.asyncio
async def test_keepalive_stops_when_the_socket_closes(fake_ws):
    """A dead socket must not leave a task spinning for the rest of the process's life."""
    events: list[ev.Event] = []
    adapter = build(events)
    await adapter.start()
    await adapter.aclose()
    await asyncio.sleep(0.05)

    task = adapter._keepalive_task
    assert task is None or task.done()


# --- endpointing: the caller pausing to think is not the end of their turn -------------------

import pytest as _pytest  # noqa: E402  (grouped with the endpointing block it serves)

from clinic_agent.core.endpointing import looks_unfinished  # noqa: E402


# Every string below is a REAL transcript from the first booked call through this engine
# (logs/traces/20260829T182056478576Z.jsonl). The left column is what the caller actually said;
# the right is whether the agent should have waited. It answered all of them immediately.
@_pytest.mark.parametrize(
    "text,unfinished",
    [
        # --- genuinely mid-sentence: the agent talked over a caller who was still going -----
        ("Well, there is a severe pain in the", True),
        ("And as well as", True),
        ("Well, I had a surgery when I was", True),
        ("in", True),
        ("How about", True),
        ("Well,", True),
        ("yeah. I'm fine with", True),
        # --- complete answers: these must NOT be delayed, or every turn gets slower ---------
        ("It's been two weeks.", False),
        ("Six?", False),
        ("Seven.", False),
        ("one PM.", False),
        ("Sunday or Monday?", False),
        ("Nothing. Everything looks good.", False),
        ("Yeah. That's correct.", False),
        ("No. I'm good.", False),
        ("Thank you.", False),
        ("This is the first time visit.", False),
        ("I had a fracture for ankle.", False),
        # --- unpunctuated but complete: NOT the strong signal, so no long window. They get
        # --- the short one via grace_seconds() below, which is cheap insurance either way.
        ("December eight two thousand", False),
        ("Nikki Kumarati", False),
        # --- unpunctuated AND mid-thought, but the last word is a fine sentence ending.
        # --- The word list cannot see these; only the missing full stop can.
        ("Well, I don't know the exact reason, but I ate", False),
        ("what's", False),
        # --- degenerate input ----------------------------------------------------------------
        ("", False),
        ("   ", False),
        ("...", False),
    ],
)
def test_looks_unfinished_against_real_transcripts(text, unfinished):
    assert looks_unfinished(text) is unfinished


# --- the grace window, end to end through the adapter ---------------------------------------


def _fast_grace(monkeypatch) -> None:
    """Run the real graded logic on a test-sized clock.

    The durations themselves are asserted by the grace_seconds table above; these tests are
    about the adapter's behaviour around the window, so they shrink it rather than restating it.
    """
    from clinic_agent.core import endpointing

    monkeypatch.setattr(endpointing, "STRONG_GRACE_S", 0.05)
    monkeypatch.setattr(endpointing, "WEAK_GRACE_S", 0.05)


def _results(text: str, *, is_final: bool, speech_final: bool = False) -> dict:
    return {
        "type": "Results",
        "is_final": is_final,
        "speech_final": speech_final,
        "channel": {"alternatives": [{"transcript": text, "confidence": 0.99}]},
    }


@pytest.mark.asyncio
async def test_a_finished_sentence_is_not_delayed(fake_ws):
    """The fast path must stay fast — most turns are already correct."""
    events: list[ev.Event] = []
    adapter = build(events)
    await adapter.start()
    try:
        adapter._handle(_results("It's been two weeks.", is_final=True, speech_final=True))
        finals = [e for e in events if isinstance(e, ev.FinalTranscript)]
        assert len(finals) == 1, "a complete sentence waited when it should not have"
        assert finals[0].text == "It's been two weeks."
    finally:
        await adapter.aclose()


@pytest.mark.asyncio
async def test_a_mid_sentence_pause_waits_then_resumes(fake_ws, monkeypatch):
    """The exact live failure: 'Well, I had a surgery when I was' → agent cut in."""
    events: list[ev.Event] = []
    adapter = build(events)
    _fast_grace(monkeypatch)
    await adapter.start()
    try:
        adapter._handle(
            _results("Well, I had a surgery when I was", is_final=True, speech_final=True)
        )
        assert not [e for e in events if isinstance(e, ev.FinalTranscript)], (
            "flushed a mid-sentence utterance instead of waiting"
        )

        # The caller carries on — the pause was a breath.
        adapter._handle(_results("in college", is_final=False))
        adapter._handle(_results("in college.", is_final=True, speech_final=True))
        await asyncio.sleep(0.12)

        finals = [e for e in events if isinstance(e, ev.FinalTranscript)]
        assert len(finals) == 1, "the resumed speech produced more than one turn"
        assert finals[0].text == "Well, I had a surgery when I was in college."
    finally:
        await adapter.aclose()


@pytest.mark.asyncio
async def test_a_caller_who_truly_trails_off_still_gets_an_answer(fake_ws, monkeypatch):
    """One grace window per utterance. Silence must not become a stall."""
    events: list[ev.Event] = []
    adapter = build(events)
    _fast_grace(monkeypatch)
    await adapter.start()
    try:
        adapter._handle(_results("How about", is_final=True, speech_final=True))
        assert not [e for e in events if isinstance(e, ev.FinalTranscript)]
        await asyncio.sleep(0.12)  # nothing more arrives

        finals = [e for e in events if isinstance(e, ev.FinalTranscript)]
        assert len(finals) == 1, "the turn never completed — the caller would hear silence"
        assert finals[0].text == "How about"
    finally:
        await adapter.aclose()


@pytest.mark.asyncio
async def test_utterance_end_does_not_preempt_an_open_grace_window(fake_ws, monkeypatch):
    """Deepgram's backstop fires sooner than the grace window and must not cut it short."""
    events: list[ev.Event] = []
    adapter = build(events)
    _fast_grace(monkeypatch)
    await adapter.start()
    try:
        adapter._handle(_results("in", is_final=True, speech_final=True))
        adapter._handle({"type": "UtteranceEnd"})
        assert not [e for e in events if isinstance(e, ev.FinalTranscript)], (
            "UtteranceEnd flushed while the grace window was still open"
        )
        await asyncio.sleep(0.12)
        assert len([e for e in events if isinstance(e, ev.FinalTranscript)]) == 1
    finally:
        await adapter.aclose()


# --- TTS: an interruption is not a provider failure ------------------------------------------


@pytest.mark.asyncio
async def test_cartesia_error_for_a_cancelled_context_is_not_a_degradation(monkeypatch):
    """Three of these fired on the first booked call, purely from the caller barging in.

    `degraded` exists to drive failover. Marking the provider unhealthy every time someone
    interrupts would make the Phase-15 circuit breaker trip on a perfectly healthy call.
    """
    from clinic_agent.core.adapters import tts as tts_mod

    events: list[ev.Event] = []
    played: list = []
    adapter = tts_mod.CartesiaTTS(
        api_key="test", voice_id="v", emit=events.append,
        play=lambda *a: asyncio.sleep(0),
        end_utterance=lambda *a: asyncio.sleep(0),
        clear_playback=lambda *a: asyncio.sleep(0),
    )

    # utt-gone was cancelled, so the adapter is no longer tracking it.
    await adapter._handle({"type": "error", "context_id": "utt-gone", "error": None})
    assert not [e for e in events if isinstance(e, ev.ProviderDegraded)]

    # An error on a context we ARE still synthesising is a real fault and must surface.
    adapter._active.add("utt-live")
    await adapter._handle({"type": "error", "context_id": "utt-live", "error": "voice not found"})
    degraded = [e for e in events if isinstance(e, ev.ProviderDegraded)]
    assert len(degraded) == 1
    assert "voice not found" in degraded[0].reason


@_pytest.mark.parametrize(
    "text,seconds",
    [
        # Complete — must not be delayed at all.
        ("It's been two weeks.", 0.0),
        ("Seven.", 0.0),
        ("Sunday or Monday?", 0.0),
        # Unambiguously mid-clause — buy the caller real time.
        ("Yes. I would like to schedule an appointment with the", 1.6),
        ("I got some of", 1.6),
        ("Well,", 1.6),
        ("in", 1.6),
        # Merely unpunctuated — short insurance, so names and dates stay fast.
        ("December eight two thousand", 0.7),
        ("Well, I don't know the exact reason, but I ate", 0.7),
        ("what's", 0.7),
        ("", 0.0),
    ],
)
def test_grace_is_graded_by_how_certain_the_signal_is(text, seconds):
    from clinic_agent.core.endpointing import grace_seconds

    assert grace_seconds(text) == seconds

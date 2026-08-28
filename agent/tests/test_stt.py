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

"""Phase 15 — a provider socket dies mid-call and the call keeps working.

Before this, both adapters said so in their own comments: STT's "nothing reconnects, so a dead
socket means a call that hears nothing and looks fine", and TTS sending a reply into a closed
connection and returning. Either one produced a call that sounded like the caller had been
hung up on while every log line stayed green.

No network. Both adapters take an injectable ``connect`` factory (the same seam
``CallSession._build_*`` uses), so a socket can be killed on cue.
"""

from __future__ import annotations

import asyncio
import json

import pytest
import websockets

from clinic_agent.core import events as ev
from clinic_agent.core.adapters import stt as stt_mod
from clinic_agent.core.adapters import tts as tts_mod


def _no_backoff(monkeypatch, module):
    """Keep the shape of the backoff, drop the wall-clock cost."""
    monkeypatch.setattr(module, "RECONNECT_BACKOFF_S", (0.0, 0.0, 0.0))


class _Socket:
    """A websocket that can be killed. Inbound messages are scripted; sends are recorded."""

    def __init__(self, inbound=()):
        self.sent: list = []
        self.closed = False
        self._inbound = list(inbound)
        self._gate = asyncio.Event()

    async def send(self, data):
        if self.closed:
            raise websockets.ConnectionClosed(None, None)
        self.sent.append(data)

    def __aiter__(self):
        return self

    async def __anext__(self):
        while self._inbound:
            return self._inbound.pop(0)
        if self.closed:
            raise websockets.ConnectionClosed(None, None)
        await self._gate.wait()
        raise StopAsyncIteration

    def kill(self):
        """Drop the connection the way a provider does: sends raise, the reader unblocks."""
        self.closed = True
        self._gate.set()

    async def close(self):
        self.closed = True
        self._gate.set()

    def messages(self):
        return [json.loads(m) for m in self.sent if isinstance(m, str)]


def _factory(*sockets):
    """Hand out sockets in order; raise once the list runs out (a provider that stays down)."""
    queue = list(sockets)

    async def connect():
        if not queue:
            raise ConnectionError("provider unreachable")
        return queue.pop(0)

    return connect


def _kinds(emitted):
    return [type(e).__name__ for e in emitted]


# --- STT ------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stt_reconnects_after_the_socket_dies_and_keeps_hearing(monkeypatch):
    _no_backoff(monkeypatch, stt_mod)
    first, second = _Socket(), _Socket()
    emitted: list = []
    stt = stt_mod.DeepgramSTT("k", emitted.append, connect=_factory(first, second))
    await stt.start()

    first.kill()
    await asyncio.sleep(0.05)

    assert stt._ws is second
    assert "ProviderRecovered" in _kinds(emitted)
    assert not any(isinstance(e, ev.ProviderDegraded) and e.fatal for e in emitted)

    # The whole point: audio sent after the drop reaches the NEW socket.
    await stt.send_audio(b"\x00\x01" * 160)
    assert second.sent, "post-reconnect audio must reach the new socket"
    await stt.aclose()


@pytest.mark.asyncio
async def test_stt_gives_up_after_the_bounded_attempts_and_says_so_fatally(monkeypatch):
    """A caller talking to a deaf agent must reach the ladder, not a healthy-looking log."""
    _no_backoff(monkeypatch, stt_mod)
    first = _Socket()
    emitted: list = []
    stt = stt_mod.DeepgramSTT("k", emitted.append, connect=_factory(first))
    await stt.start()

    first.kill()
    await asyncio.sleep(0.05)

    fatal = [e for e in emitted if isinstance(e, ev.ProviderDegraded) and e.fatal]
    assert fatal and fatal[0].provider == "stt"
    assert "ProviderRecovered" not in _kinds(emitted)
    await stt.aclose()


@pytest.mark.asyncio
async def test_stt_teardown_does_not_start_a_reconnect(monkeypatch):
    """Closing the call closes the socket; racing a reconnect against it wastes the attempt."""
    _no_backoff(monkeypatch, stt_mod)
    first, spare = _Socket(), _Socket()
    emitted: list = []
    stt = stt_mod.DeepgramSTT("k", emitted.append, connect=_factory(first, spare))
    await stt.start()

    await stt.aclose()
    await asyncio.sleep(0.05)

    assert stt._ws is not spare
    assert "ProviderRecovered" not in _kinds(emitted)


# --- TTS ------------------------------------------------------------------------------------


def _tts(emitted, connect):
    spoken: list = []
    return (
        tts_mod.CartesiaTTS(
            "k",
            "voice",
            emitted.append,
            play=lambda ctx, pcm: asyncio.sleep(0),
            end_utterance=lambda ctx: asyncio.sleep(0),
            clear_playback=lambda ctx: asyncio.sleep(0),
            connect=connect,
        ),
        spoken,
    )


@pytest.mark.asyncio
async def test_tts_reconnects_and_re_sends_the_utterance_in_flight(monkeypatch):
    _no_backoff(monkeypatch, tts_mod)
    first, second = _Socket(), _Socket()
    emitted: list = []
    tts, _ = _tts(emitted, _factory(first, second))
    await tts.start()

    await tts.speak("u1", "Your appointment is confirmed.", final=False)
    first.kill()
    await tts.speak("u1", " See you Tuesday.", final=True)
    await asyncio.sleep(0.05)

    transcripts = [m.get("transcript") for m in second.messages()]
    assert "Your appointment is confirmed." in transcripts, (
        "the utterance in flight must be replayed — otherwise the caller hears half a sentence "
        "and then nothing"
    )
    assert " See you Tuesday." in transcripts
    assert "ProviderRecovered" in _kinds(emitted)
    await tts.aclose()


@pytest.mark.asyncio
async def test_tts_gives_up_fatally_when_the_provider_stays_down(monkeypatch):
    _no_backoff(monkeypatch, tts_mod)
    first = _Socket()
    emitted: list = []
    tts, _ = _tts(emitted, _factory(first))
    await tts.start()

    await tts.speak("u1", "Hello.", final=False)
    first.kill()
    await tts.speak("u1", " More.", final=True)
    await asyncio.sleep(0.05)

    assert [e for e in emitted if isinstance(e, ev.ProviderDegraded) and e.fatal]
    await tts.aclose()


@pytest.mark.asyncio
async def test_a_finished_utterance_is_not_replayed(monkeypatch):
    """Replay is for the sentence that was cut off, never for one the caller already heard."""
    _no_backoff(monkeypatch, tts_mod)
    first, second = _Socket(), _Socket()
    emitted: list = []
    tts, _ = _tts(emitted, _factory(first, second))
    await tts.start()

    await tts.speak("u1", "All done.", final=True)
    await tts._handle({"type": "done", "context_id": "u1"})   # playback finished
    first.kill()
    await tts.speak("u2", "Next question?", final=True)
    await asyncio.sleep(0.05)

    transcripts = [m.get("transcript") for m in second.messages()]
    assert "All done." not in transcripts
    assert "Next question?" in transcripts
    await tts.aclose()


@pytest.mark.asyncio
async def test_tts_reconnects_when_the_RECEIVE_loop_is_the_one_that_notices(monkeypatch):
    """The reconnect runs on the receive task itself, so it must not cancel that task.

    It used to: `self._task.cancel()` inside `_reconnect` killed the coroutine that was running
    it, at the next await — which is the re-send. The utterance was silently dropped and the
    socket looked healthy.
    """
    _no_backoff(monkeypatch, tts_mod)
    first, second = _Socket(), _Socket()
    emitted: list = []
    tts, _ = _tts(emitted, _factory(first, second))
    await tts.start()

    await tts.speak("u1", "Half a sentence", final=False)
    first.kill()                      # the receive loop notices before anything else does
    await asyncio.sleep(0.05)

    assert "ProviderRecovered" in _kinds(emitted)
    transcripts = [m.get("transcript") for m in second.messages()]
    assert "Half a sentence" in transcripts
    await tts.aclose()

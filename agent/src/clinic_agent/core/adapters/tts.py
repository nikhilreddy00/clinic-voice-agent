"""Phase 10 — Cartesia streaming TTS adapter.

Speaks over Cartesia's websocket API directly. The mechanism that matters is the **context**:
one ``context_id`` per bot utterance, with each sentence sent as ``continue: true``. The
reducer emits a :class:`~clinic_agent.core.actions.Speak` per sentence as the model produces
it, and because they all land on the same context the caller hears one continuous reply with
correct prosody across sentence boundaries — rather than separately-synthesized fragments
butted together. The last ``Speak`` for an utterance carries ``final=True``, which sends the
empty ``continue: false`` message that closes and flushes the context.

Cancellation is the barge-in path, and cancelling the Cartesia context only stops *synthesis*
— audio already queued for playback keeps going. So a cancel here also tells the media adapter
to drop its buffered output. Doing only the first half is the classic bug where the bot keeps
talking for a second after being interrupted.

This adapter deliberately does **not** emit ``BotStartedSpeaking`` / ``BotStoppedSpeaking``.
Cartesia's ``done`` means synthesis finished, which on a fast connection is well before the
caller has heard the audio; treating it as the end of the bot's turn would reopen the mic
mid-sentence and feed the bot's own tail back into STT. Those events belong to the media
adapter, which is the only component that knows when playback actually drained.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from collections.abc import Awaitable, Callable

import websockets
from loguru import logger

from .. import events as ev
from ..audio import OUTPUT_SAMPLE_RATE

CARTESIA_URL = "wss://api.cartesia.ai/tts/websocket"
CARTESIA_VERSION = "2026-03-01"

EmitFn = Callable[[ev.Event], None]
PlayFn = Callable[[str, bytes], Awaitable[None]]
EndFn = Callable[[str], Awaitable[None]]
ClearFn = Callable[[str], Awaitable[None]]


class CartesiaTTS:
    """One TTS websocket for one call."""

    def __init__(
        self,
        api_key: str,
        voice_id: str,
        emit: EmitFn,
        play: PlayFn,
        end_utterance: EndFn,
        clear_playback: ClearFn,
        *,
        model: str = "sonic-3.5",
        sample_rate: int = OUTPUT_SAMPLE_RATE,
    ) -> None:
        self._api_key = api_key
        self._voice_id = voice_id
        self._model = model
        self._sample_rate = sample_rate
        self._emit = emit
        self._play = play
        self._end_utterance = end_utterance
        self._clear_playback = clear_playback

        self._ws: websockets.ClientConnection | None = None
        self._task: asyncio.Task | None = None
        # Utterance ids the caller should still hear. A chunk whose context is not in here is
        # audio for a cancelled utterance that was already in flight — drop it.
        self._active: set[str] = set()

    async def start(self) -> None:
        self._ws = await websockets.connect(
            f"{CARTESIA_URL}?api_key={self._api_key}&cartesia_version={CARTESIA_VERSION}"
        )
        self._task = asyncio.create_task(self._receive_loop(), name="tts-receive")
        logger.info(f"[tts] cartesia connected (model={self._model}, {self._sample_rate} Hz)")

    async def speak(self, utterance_id: str, text: str, *, final: bool) -> None:
        """Append ``text`` to ``utterance_id``'s context; ``final`` closes and flushes it."""
        if self._ws is None:
            return
        if not text and not final:
            return

        self._active.add(utterance_id)
        message = {
            "transcript": text,
            "continue": not final,
            "context_id": utterance_id,
            "model_id": self._model,
            "voice": {"mode": "id", "id": self._voice_id},
            "output_format": {
                "container": "raw",
                "encoding": "pcm_s16le",
                "sample_rate": self._sample_rate,
            },
            "add_timestamps": False,
        }
        try:
            await self._ws.send(json.dumps(message))
        except websockets.ConnectionClosed as exc:
            self._ws = None
            self._emit(ev.ProviderDegraded(t=time.monotonic(), provider="tts", reason=str(exc)))
            return
        if final:
            logger.info("TTS  ▶ synthesis flushed for utterance")

    async def cancel(self, utterance_id: str) -> None:
        """Stop synthesis AND drop buffered playback, so the bot goes quiet immediately."""
        self._active.discard(utterance_id)
        if self._ws is not None:
            try:
                await self._ws.send(json.dumps({"context_id": utterance_id, "cancel": True}))
            except websockets.ConnectionClosed:
                self._ws = None
        await self._clear_playback(utterance_id)

    async def _receive_loop(self) -> None:
        assert self._ws is not None
        try:
            async for raw in self._ws:
                if isinstance(raw, bytes):
                    continue
                await self._handle(json.loads(raw))
        except websockets.ConnectionClosed as exc:
            logger.warning(f"[tts] cartesia connection closed: {exc}")
            self._emit(ev.ProviderDegraded(t=time.monotonic(), provider="tts", reason=str(exc)))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - TTS must never take the call down
            logger.error(f"[tts] receive loop error: {exc}")
            self._emit(ev.ProviderDegraded(t=time.monotonic(), provider="tts", reason=str(exc)))

    async def _handle(self, message: dict) -> None:
        kind = message.get("type")
        context_id = message.get("context_id") or ""

        if kind == "chunk":
            if context_id not in self._active:
                return  # audio for an utterance the caller already interrupted
            await self._play(context_id, base64.b64decode(message["data"]))

        elif kind == "done":
            # Synthesis complete. Playback is not — the media adapter emits the end-of-turn
            # event once its buffer actually drains.
            self._active.discard(context_id)
            await self._end_utterance(context_id)

        elif kind == "error":
            # Cancelling a context makes Cartesia answer with an error frame, so every barge-in
            # produced one of these. It was logged at ERROR and marked the whole provider
            # degraded — on the first booked call, three times, purely from the caller
            # interrupting. `degraded` is meant to drive failover, so poisoning it with normal
            # turn-taking would make the Phase-15 breaker fire on healthy calls.
            #
            # A context we are no longer tracking is one we cancelled ourselves. That is
            # expected, not a fault.
            detail = message.get("error") or message.get("message") or message
            if context_id not in self._active:
                logger.debug(f"[tts] cartesia closed a cancelled context: {detail}")
                await self._clear_playback(context_id)
                return
            logger.error(f"[tts] cartesia error: {detail}")
            self._active.discard(context_id)
            self._emit(ev.ProviderDegraded(t=time.monotonic(), provider="tts", reason=str(detail)))
            await self._clear_playback(context_id)

    async def aclose(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        if self._ws is not None:
            await self._ws.close()
            self._ws = None

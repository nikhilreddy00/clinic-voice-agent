"""Phase 10 — Deepgram streaming STT adapter.

Talks to Deepgram's realtime endpoint over a raw websocket rather than through Pipecat's
service wrapper. That is the point of the phase: the orchestration is ours, and the vendor
boundary is one small, readable adapter that produces events.

Two Deepgram signals matter and they are not the same thing:

* ``is_final`` — this *segment* of transcript will not change. Deepgram emits several of these
  inside one spoken sentence.
* ``speech_final`` — Deepgram's endpointer believes the **utterance** is over.

Emitting a :class:`FinalTranscript` per ``is_final`` would hand the reducer three user messages
for one sentence, and each would cancel and restart the LLM — three times the cost and worse
latency than doing nothing. So segments are accumulated and released on ``speech_final``, with
Deepgram's ``UtteranceEnd`` message as the backstop for the case where the endpointer never
fires (it can be swallowed when the caller trails off into background noise).
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from urllib.parse import urlencode

import websockets
from loguru import logger

from .. import events as ev
from ..audio import INPUT_SAMPLE_RATE, NUM_CHANNELS

DEEPGRAM_URL = "wss://api.deepgram.com/v1/listen"

EmitFn = Callable[[ev.Event], None]
PartialFn = Callable[[str, float], None]


class DeepgramSTT:
    """One streaming STT connection for one call."""

    def __init__(
        self,
        api_key: str,
        emit: EmitFn,
        *,
        on_partial: PartialFn | None = None,
        model: str = "nova-3",
        language: str = "en-US",
        sample_rate: int = INPUT_SAMPLE_RATE,
    ) -> None:
        self._api_key = api_key
        self._emit = emit
        self._on_partial = on_partial
        self._sample_rate = sample_rate
        self._params = {
            "model": model,
            "language": language,
            "encoding": "linear16",
            "sample_rate": str(sample_rate),
            "channels": str(NUM_CHANNELS),
            "interim_results": "true",   # required for the turn-level barge-in word count
            "punctuate": "true",         # sentence boundaries drive the reducer's TTS chunking
            "smart_format": "false",
            "endpointing": "300",        # ms of silence Deepgram treats as end of utterance
            "utterance_end_ms": "1000",  # backstop when the endpointer never fires
        }

        self._ws: websockets.ClientConnection | None = None
        self._task: asyncio.Task | None = None
        self._segments: list[str] = []
        self._confidences: list[float] = []

    async def start(self) -> None:
        url = f"{DEEPGRAM_URL}?{urlencode(self._params)}"
        self._ws = await websockets.connect(
            url, additional_headers={"Authorization": f"Token {self._api_key}"}
        )
        self._task = asyncio.create_task(self._receive_loop(), name="stt-receive")
        logger.info(f"[stt] deepgram connected (model={self._params['model']}, {self._sample_rate} Hz)")

    async def send_audio(self, pcm: bytes) -> None:
        """Push one PCM frame. Silently drops when disconnected — the call outlives the socket."""
        if self._ws is None:
            return
        try:
            await self._ws.send(pcm)
        except websockets.ConnectionClosed:
            self._ws = None
            self._emit(ev.ProviderDegraded(t=time.monotonic(), provider="stt", reason="closed"))

    async def _receive_loop(self) -> None:
        assert self._ws is not None
        try:
            async for raw in self._ws:
                if isinstance(raw, bytes):
                    continue
                self._handle(json.loads(raw))
        except websockets.ConnectionClosed as exc:
            logger.warning(f"[stt] deepgram connection closed: {exc}")
            self._emit(ev.ProviderDegraded(t=time.monotonic(), provider="stt", reason=str(exc)))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - STT must never take the call down
            logger.error(f"[stt] receive loop error: {exc}")
            self._emit(ev.ProviderDegraded(t=time.monotonic(), provider="stt", reason=str(exc)))

    def _handle(self, message: dict) -> None:
        kind = message.get("type")
        if kind == "UtteranceEnd":
            self._flush(time.monotonic())
            return
        if kind != "Results":
            return

        alternatives = message.get("channel", {}).get("alternatives") or [{}]
        text = (alternatives[0].get("transcript") or "").strip()
        confidence = alternatives[0].get("confidence")
        now = time.monotonic()

        if not message.get("is_final"):
            if text:
                logger.debug(f"ASR  ▷ interim: {text!r}")
                self._emit(ev.PartialTranscript(t=now, text=text))
                if self._on_partial:
                    self._on_partial(text, now)
            return

        if text:
            self._segments.append(text)
            if isinstance(confidence, (int, float)):
                self._confidences.append(float(confidence))
        if message.get("speech_final"):
            self._flush(now)

    def _flush(self, now: float) -> None:
        if not self._segments:
            return
        text = " ".join(self._segments)
        confidence = (
            sum(self._confidences) / len(self._confidences) if self._confidences else None
        )
        self._segments.clear()
        self._confidences.clear()
        logger.info(f"ASR  ▶ transcript received: {text!r}")
        self._emit(ev.FinalTranscript(t=now, text=text, confidence=confidence))

    async def aclose(self) -> None:
        if self._ws is not None:
            try:
                await self._ws.send(json.dumps({"type": "CloseStream"}))
            except (websockets.ConnectionClosed, RuntimeError):
                pass
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        if self._ws is not None:
            await self._ws.close()
            self._ws = None

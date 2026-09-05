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
from ..endpointing import grace_seconds, looks_unfinished
from ..reliability import CircuitBreaker

DEEPGRAM_URL = "wss://api.deepgram.com/v1/listen"

EmitFn = Callable[[ev.Event], None]
PartialFn = Callable[[str, float], None]
ConnectFn = Callable[[], "websockets.ClientConnection"]

# Phase 15. Reconnect delays, in seconds. Short and few on purpose: past about two seconds the
# caller has already repeated themselves, and the ladder's next rung (a human) is the better
# answer than a fourth attempt.
RECONNECT_BACKOFF_S = (0.25, 0.5, 1.0)


class DeepgramSTT:
    """One streaming STT connection for one call."""

    # Deepgram closes a stream that has received no audio for 10 s (error net0001). A call has
    # two ordinary silence windows longer than that: the wait for an inbound SIP call, which is
    # unbounded, and any bot utterance over ~10 s, because the mic gate drops every input frame
    # while the bot speaks. Both used to kill the socket, and since nothing reconnects, the
    # agent then ran the rest of the call deaf. A KeepAlive text frame resets the timer without
    # being billed as audio; Deepgram asks for one every 3-5 s.
    KEEPALIVE_INTERVAL_S = 5.0

    # A caller who pauses, resumes, and pauses again is still composing one sentence, so the
    # window renews rather than firing once. Capped because the point is patience, not a stall:
    # past this many holds the agent answers with what it has, which is what the warm
    # "take your time" reply is for.
    MAX_GRACE_WINDOWS = 3

    def __init__(
        self,
        api_key: str,
        emit: EmitFn,
        *,
        on_partial: PartialFn | None = None,
        model: str = "nova-3",
        language: str = "en-US",
        sample_rate: int = INPUT_SAMPLE_RATE,
        connect: ConnectFn | None = None,
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
        self._keepalive_task: asyncio.Task | None = None
        self._last_send = 0.0
        self._grace_task: asyncio.Task | None = None
        self._grace_windows = 0  # holds spent on the current utterance
        self._segments: list[str] = []
        self._confidences: list[float] = []

        # Phase 15. Until now nothing reconnected: a socket that closed mid-call left the agent
        # deaf for the rest of it, with a healthy-looking log. Injectable so the chaos tests can
        # kill the connection without a network.
        self._connect = connect or self._default_connect
        self._breaker = CircuitBreaker("stt", threshold=len(RECONNECT_BACKOFF_S), cooldown_s=5.0)
        self._reconnecting = False
        self._closing = False

    async def _default_connect(self):
        url = f"{DEEPGRAM_URL}?{urlencode(self._params)}"
        return await websockets.connect(
            url, additional_headers={"Authorization": f"Token {self._api_key}"}
        )

    async def start(self) -> None:
        self._ws = await self._connect()
        self._start_loops()
        logger.info(f"[stt] deepgram connected (model={self._params['model']}, {self._sample_rate} Hz)")

    def _start_loops(self) -> None:
        self._last_send = time.monotonic()
        self._task = asyncio.create_task(self._receive_loop(), name="stt-receive")
        self._keepalive_task = asyncio.create_task(self._keepalive_loop(), name="stt-keepalive")

    def _schedule_reconnect(self, reason: str) -> None:
        """Reconnect off the caller's path. Safe to call from anywhere, including the hot
        audio path — the backoff must never block a 20 ms frame."""
        if self._reconnecting or self._closing:
            return
        self._reconnecting = True
        asyncio.create_task(self._reconnect(reason), name="stt-reconnect")

    async def _reconnect(self, reason: str) -> None:
        self._emit(ev.ProviderDegraded(t=time.monotonic(), provider="stt", reason=reason))
        try:
            for delay in RECONNECT_BACKOFF_S:
                if self._closing or not self._breaker.allow(time.monotonic()):
                    break
                await asyncio.sleep(delay)
                if self._closing:
                    return
                try:
                    ws = await self._connect()
                except Exception as exc:  # noqa: BLE001 - a failed attempt is just an attempt
                    self._breaker.record_failure(time.monotonic())
                    logger.warning(f"[stt] reconnect failed: {exc}")
                    continue
                self._breaker.record_success()
                self._ws = ws
                if self._keepalive_task is not None:
                    self._keepalive_task.cancel()
                self._start_loops()
                logger.info("[stt] deepgram reconnected")
                self._emit(ev.ProviderRecovered(t=time.monotonic(), provider="stt"))
                return
            # Out of attempts. `fatal` is what tells the reducer this call can no longer hear —
            # the difference between one blip and a caller talking to a deaf agent.
            logger.error("[stt] deepgram is gone; the call can no longer hear")
            self._emit(
                ev.ProviderDegraded(
                    t=time.monotonic(), provider="stt", reason=reason, fatal=True
                )
            )
        finally:
            self._reconnecting = False

    async def send_audio(self, pcm: bytes) -> None:
        """Push one PCM frame. Silently drops when disconnected — the call outlives the socket."""
        if self._ws is None:
            return
        try:
            await self._ws.send(pcm)
            self._last_send = time.monotonic()
        except websockets.ConnectionClosed:
            self._ws = None
            self._schedule_reconnect("closed")

    async def _keepalive_loop(self) -> None:
        """Send a KeepAlive whenever the stream has been silent for a full interval.

        Sleeping only the *remaining* time rather than a fixed tick keeps the real gap at or
        under the interval. A fixed tick would allow a gap of nearly twice it, which is how a
        5 s keepalive quietly becomes a 10 s one and hits the very timeout it exists to avoid.
        """
        try:
            while True:
                ws = self._ws
                if ws is None:
                    return
                idle_for = time.monotonic() - self._last_send
                if idle_for < self.KEEPALIVE_INTERVAL_S:
                    await asyncio.sleep(self.KEEPALIVE_INTERVAL_S - idle_for)
                    continue
                try:
                    await ws.send(json.dumps({"type": "KeepAlive"}))
                    self._last_send = time.monotonic()
                except websockets.ConnectionClosed:
                    self._ws = None
                    self._schedule_reconnect("closed")
                    return
        except asyncio.CancelledError:
            raise

    async def _receive_loop(self) -> None:
        assert self._ws is not None
        try:
            async for raw in self._ws:
                if isinstance(raw, bytes):
                    continue
                self._handle(json.loads(raw))
        except websockets.ConnectionClosed as exc:
            logger.warning(f"[stt] deepgram connection closed: {exc}")
            self._ws = None
            self._schedule_reconnect(str(exc))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - STT must never take the call down
            logger.error(f"[stt] receive loop error: {exc}")
            self._ws = None
            self._schedule_reconnect(str(exc))

    def _handle(self, message: dict) -> None:
        kind = message.get("type")
        if kind == "UtteranceEnd":
            # While a grace window is open it owns the flush. Deepgram's backstop fires at
            # utterance_end_ms, which is sooner, and letting it through here would cut the
            # pause short and undo the whole mechanism.
            if self._grace_pending():
                return
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
                self._cancel_grace()  # the caller resumed — they were only thinking
                logger.debug(f"ASR  ▷ interim: {text!r}")
                self._emit(ev.PartialTranscript(t=now, text=text))
                if self._on_partial:
                    self._on_partial(text, now)
            return

        if text:
            self._cancel_grace()
            self._segments.append(text)
            if isinstance(confidence, (int, float)):
                self._confidences.append(float(confidence))
        if message.get("speech_final"):
            self._end_of_turn(now)

    def _end_of_turn(self, now: float) -> None:
        """Deepgram says the turn is over. Decide whether the caller agrees."""
        text = " ".join(self._segments).strip()
        wait = grace_seconds(text) if text else 0.0
        if wait and self._grace_windows < self.MAX_GRACE_WINDOWS:
            self._grace_windows += 1
            logger.info(
                f"[stt] holding the turn {wait:.1f}s "
                f"({self._grace_windows}/{self.MAX_GRACE_WINDOWS}) — {text[-48:]!r}"
            )
            self._emit(ev.TurnHeld(
                t=now, tail=text[-48:], seconds=wait,
                window=self._grace_windows, windows_max=self.MAX_GRACE_WINDOWS,
            ))
            self._grace_task = asyncio.create_task(
                self._grace_then_flush(wait), name="stt-grace"
            )
            return
        if wait:
            # Patience exhausted on a sentence that is STILL visibly mid-clause. Sending it is
            # the least-bad option left — the alternative is dead air — but it is a failure,
            # and the trace should say so rather than look like a normal turn.
            logger.info(f"[stt] releasing an unfinished turn after {self.MAX_GRACE_WINDOWS} "
                        f"holds — {text[-48:]!r}")
            self._emit(ev.TurnHeld(
                t=now, tail=text[-48:], seconds=0.0,
                window=self._grace_windows, windows_max=self.MAX_GRACE_WINDOWS, released=True,
            ))
        self._flush(now)

    async def _grace_then_flush(self, wait: float) -> None:
        """Wait out one hold, then RE-DECIDE rather than flushing unconditionally.

        This is the fix for a live call where the caller was cut off four times in twenty
        seconds. Each hold used to expire straight into a flush, so a visibly-unfinished
        sentence bought exactly one 1.6 s window — and a caller who pauses longer than that to
        gather their thoughts ("I would like to reschedule my …") got answered mid-sentence.
        The reply then talked over their next attempt, which the mic gate clipped, which
        produced another fragment: the loop the caller experienced as being interrupted
        constantly.

        Re-entering _end_of_turn re-reads the (unchanged) segments and opens another window
        while the text still looks unfinished, up to MAX_GRACE_WINDOWS. Patience for a
        mid-clause fragment is therefore ~4.8 s rather than 1.6 s, while a finished sentence
        still flushes immediately — grace_seconds returns 0.0 for those, so they never reach
        this path at all.
        """
        try:
            await asyncio.sleep(wait)
        except asyncio.CancelledError:
            return
        self._grace_task = None
        now = time.monotonic()
        # Renew ONLY on the strong signal. A merely-unpunctuated turn ("December eight two
        # thousand", "Nikhil Kumar") is usually a complete answer, and re-deciding on those
        # would silently turn one 0.7 s window into three — taxing every name and date in the
        # call, which is the exact trade the two-tier design exists to avoid. Measured on a
        # live call: "Hi. I would like to book an appointment" was held three times.
        if looks_unfinished(" ".join(self._segments).strip()):
            self._end_of_turn(now)
            return
        self._flush(now)

    def _grace_pending(self) -> bool:
        return self._grace_task is not None and not self._grace_task.done()

    def _cancel_grace(self) -> None:
        """More speech arrived — the pause was a breath, not an ending."""
        if self._grace_pending():
            self._grace_task.cancel()
        self._grace_task = None

    def _flush(self, now: float) -> None:
        self._grace_task = None
        self._grace_windows = 0
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
        # Set first: a socket closing during teardown must not start a reconnect race.
        self._closing = True
        self._cancel_grace()
        if self._keepalive_task is not None:
            self._keepalive_task.cancel()
            try:
                await self._keepalive_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._keepalive_task = None
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

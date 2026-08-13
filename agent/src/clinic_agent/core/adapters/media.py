"""Phase 10 — media adapters: audio in and out.

Two implementations behind one interface, exactly as before: the laptop mic/speaker
(``MODE=local``) and a LiveKit room fed by an inbound SIP call (``MODE=telephony``). Media
transport is the one layer this project deliberately keeps buying — writing a SIP/WebRTC stack
is where the rewrite would die, and it is not what the loop rebuild is about.

The shared base owns **playback**, which is more than a queue:

* It emits :class:`BotStartedSpeaking` when the first audio of an utterance is actually written
  to the device, and :class:`BotStoppedSpeaking` only once the buffer has drained *and*
  synthesis has finished. The reducer reopens the caller's turn on that second event, so
  sourcing it from anywhere earlier (Cartesia's ``done``, say) would unmute the mic while the
  bot is still audible and feed its own voice back into STT.
* ``clear()`` drops queued audio immediately. This is the half of barge-in that cancelling the
  TTS context does not do — without it the caller interrupts and still hears another second of
  bot.

Input is pushed straight to the :class:`~clinic_agent.core.adapters.turn.TurnEngine`; raw audio
never becomes an event and never reaches the reducer.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable

from loguru import logger

from .. import events as ev
from ..audio import (
    INPUT_SAMPLE_RATE,
    NUM_CHANNELS,
    OUTPUT_FRAME_BYTES,
    OUTPUT_SAMPLE_RATE,
)

EmitFn = Callable[[ev.Event], None]
AudioInFn = Callable[[bytes, float], Awaitable[None]]


class MediaAdapter:
    """Base: playback queue, speaking-state events, and the audio-in hook."""

    def __init__(self, emit: EmitFn, on_audio: AudioInFn) -> None:
        self._emit = emit
        self._on_audio = on_audio
        self._queue: asyncio.Queue[tuple[str, bytes]] = asyncio.Queue()
        self._synthesis_done: set[str] = set()
        self._speaking: str | None = None
        self._playback_task: asyncio.Task | None = None
        self._closing = False

    # --- playback (called by the TTS adapter) -------------------------------------------

    async def play(self, utterance_id: str, pcm: bytes) -> None:
        await self._queue.put((utterance_id, pcm))

    async def end_utterance(self, utterance_id: str) -> None:
        """Synthesis for this utterance is complete; no more chunks are coming."""
        self._synthesis_done.add(utterance_id)
        if self._speaking is None and self._queue.empty():
            # Everything already drained before the done message arrived.
            self._finish(utterance_id, completed=True)

    async def clear(self, utterance_id: str) -> None:
        """Barge-in: throw away everything queued and stop the device mid-buffer."""
        dropped = 0
        remaining: list[tuple[str, bytes]] = []
        while not self._queue.empty():
            item = self._queue.get_nowait()
            if item[0] == utterance_id:
                dropped += 1
            else:
                remaining.append(item)
        for item in remaining:
            self._queue.put_nowait(item)

        self._synthesis_done.discard(utterance_id)
        await self._flush_device()
        if dropped:
            logger.info(f"[media] dropped {dropped} queued audio chunks on interruption")
        if self._speaking == utterance_id:
            self._finish(utterance_id, completed=False)

    def _finish(self, utterance_id: str, *, completed: bool) -> None:
        self._speaking = None
        self._synthesis_done.discard(utterance_id)
        self._emit(
            ev.BotStoppedSpeaking(
                t=time.monotonic(), utterance_id=utterance_id, completed=completed
            )
        )

    async def _playback_loop(self) -> None:
        while not self._closing:
            utterance_id, pcm = await self._queue.get()
            if self._speaking != utterance_id:
                self._speaking = utterance_id
                self._emit(ev.BotStartedSpeaking(t=time.monotonic(), utterance_id=utterance_id))

            for offset in range(0, len(pcm), OUTPUT_FRAME_BYTES):
                if self._speaking != utterance_id:
                    break  # cancelled mid-chunk
                await self._write(pcm[offset : offset + OUTPUT_FRAME_BYTES])

            if (
                self._speaking == utterance_id
                and self._queue.empty()
                and utterance_id in self._synthesis_done
            ):
                await self._drain_device()
                self._finish(utterance_id, completed=True)

    # --- subclass hooks -----------------------------------------------------------------

    async def _write(self, pcm: bytes) -> None:
        """Write one output frame. MUST pace — return no sooner than the audio is consumed."""
        raise NotImplementedError

    async def _drain_device(self) -> None:
        """Block until already-written audio has finished playing."""

    async def _flush_device(self) -> None:
        """Discard audio already handed to the device."""

    async def start(self) -> None:
        self._playback_task = asyncio.create_task(self._playback_loop(), name="playback")

    async def aclose(self) -> None:
        self._closing = True
        if self._playback_task is not None:
            self._playback_task.cancel()
            try:
                await self._playback_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass


class LocalMedia(MediaAdapter):
    """Laptop mic + speaker via PyAudio (``MODE=local``).

    Both PyAudio calls are blocking, so they run on a dedicated thread each. Doing the output
    write on the event loop was the Phase-1 choppy-playback bug: synchronous device I/O on the
    loop that also has to deliver the next 20 ms of audio produces output underruns.
    """

    def __init__(self, emit: EmitFn, on_audio: AudioInFn) -> None:
        super().__init__(emit, on_audio)
        import pyaudio  # imported lazily so the telephony path needs no audio device

        self._pyaudio = pyaudio.PyAudio()
        self._fmt = pyaudio.paInt16
        self._in_stream = None
        self._out_stream = None
        self._capture_task: asyncio.Task | None = None

    async def start(self) -> None:
        self._in_stream = self._pyaudio.open(
            format=self._fmt,
            channels=NUM_CHANNELS,
            rate=INPUT_SAMPLE_RATE,
            input=True,
            frames_per_buffer=INPUT_SAMPLE_RATE // 50,  # 20 ms
        )
        self._out_stream = self._pyaudio.open(
            format=self._fmt,
            channels=NUM_CHANNELS,
            rate=OUTPUT_SAMPLE_RATE,
            output=True,
        )
        await super().start()
        self._capture_task = asyncio.create_task(self._capture_loop(), name="mic-capture")
        logger.info(
            f"[media] local mic {INPUT_SAMPLE_RATE} Hz → speaker {OUTPUT_SAMPLE_RATE} Hz"
        )
        # The mic is live the instant the stream opens, so the caller is already "present".
        self._emit(ev.CallerPresent(t=time.monotonic(), participant_id="local-mic"))

    async def _capture_loop(self) -> None:
        loop = asyncio.get_running_loop()
        chunk = INPUT_SAMPLE_RATE // 50
        while not self._closing:
            pcm = await loop.run_in_executor(
                None, lambda: self._in_stream.read(chunk, exception_on_overflow=False)
            )
            await self._on_audio(pcm, time.monotonic())

    async def _write(self, pcm: bytes) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._out_stream.write, pcm)

    async def _flush_device(self) -> None:
        # PyAudio has no "drop buffered output" call; stopping and restarting the stream is
        # the supported way to discard it, and it is what makes barge-in sound instant.
        if self._out_stream is not None:
            self._out_stream.stop_stream()
            self._out_stream.start_stream()

    async def aclose(self) -> None:
        await super().aclose()
        if self._capture_task is not None:
            self._capture_task.cancel()
            try:
                await self._capture_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        for stream in (self._in_stream, self._out_stream):
            if stream is not None:
                stream.stop_stream()
                stream.close()
        self._pyaudio.terminate()


class LiveKitMedia(MediaAdapter):
    """LiveKit room audio (``MODE=telephony``), fed by the inbound SIP participant.

    ``AudioSource`` paces playback and exposes ``clear_queue()`` and ``wait_for_playout()``,
    which is precisely the drain/flush contract the base class needs — so barge-in and
    end-of-turn detection are exact here rather than approximated.
    """

    def __init__(self, emit: EmitFn, on_audio: AudioInFn, url: str, token: str, room_name: str) -> None:
        super().__init__(emit, on_audio)
        from livekit import rtc

        self._rtc = rtc
        self._url = url
        self._token = token
        self._room_name = room_name
        self._room = rtc.Room()
        self._source = rtc.AudioSource(OUTPUT_SAMPLE_RATE, NUM_CHANNELS)
        self._stream_tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        rtc = self._rtc

        @self._room.on("track_subscribed")
        def _on_track(track, publication, participant) -> None:  # noqa: ANN001
            if track.kind == rtc.TrackKind.KIND_AUDIO:
                self._stream_tasks.append(
                    asyncio.create_task(self._consume(track), name="livekit-audio-in")
                )

        @self._room.on("participant_connected")
        def _on_join(participant) -> None:  # noqa: ANN001
            logger.info(f"[telephony] caller joined room (participant={participant.identity})")
            self._emit(
                ev.CallerPresent(t=time.monotonic(), participant_id=participant.identity)
            )

        @self._room.on("participant_disconnected")
        def _on_leave(participant) -> None:  # noqa: ANN001
            logger.info(f"[telephony] caller left room (participant={participant.identity})")
            self._emit(ev.Hangup(t=time.monotonic(), reason="caller_left"))

        await self._room.connect(self._url, self._token)
        track = rtc.LocalAudioTrack.create_audio_track("agent-voice", self._source)
        await self._room.local_participant.publish_track(
            track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
        )
        await super().start()
        logger.info(
            f"[telephony] joined room {self._room_name!r}; waiting for the inbound SIP caller"
        )

        # A SIP call can be bridged in before the agent finishes connecting, in which case the
        # participant_connected event already fired and would never fire again.
        for participant in self._room.remote_participants.values():
            self._emit(ev.CallerPresent(t=time.monotonic(), participant_id=participant.identity))
            break

    async def _consume(self, track) -> None:  # noqa: ANN001
        stream = self._rtc.AudioStream(
            track, sample_rate=INPUT_SAMPLE_RATE, num_channels=NUM_CHANNELS
        )
        async for event in stream:
            await self._on_audio(bytes(event.frame.data), time.monotonic())

    async def _write(self, pcm: bytes) -> None:
        samples = len(pcm) // (2 * NUM_CHANNELS)
        if samples == 0:
            return
        frame = self._rtc.AudioFrame(pcm, OUTPUT_SAMPLE_RATE, NUM_CHANNELS, samples)
        await self._source.capture_frame(frame)

    async def _drain_device(self) -> None:
        await self._source.wait_for_playout()

    async def _flush_device(self) -> None:
        self._source.clear_queue()

    async def aclose(self) -> None:
        await super().aclose()
        for task in self._stream_tasks:
            task.cancel()
        await self._room.disconnect()

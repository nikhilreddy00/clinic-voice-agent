"""Pipecat pipeline for the clinic voice agent — Phase 1 (local ASR->LLM->TTS loop).

Phase 1 scope: one working local voice loop over the laptop mic/speaker.

    mic -> Deepgram (ASR) -> Groq/Llama (LLM) -> Cartesia (TTS) -> speaker

On start the agent speaks a fixed greeting that includes the mandatory AI disclosure
(spoken deterministically, NOT LLM-generated, so the disclosure wording is exact every run),
then handles the caller's follow-up turn with a minimal, tightly scoped system prompt.

Deliberately OUT of scope here (later phases):
    - slot-filling / booking / scheduling-API tool calls  (Phase 2)
    - the full dialogue state machine + validation/fallbacks (Phase 3, docs/build_spec.md)
    - telephony / SIP                                        (Phase 7)

Pipeline order (Pipecat 1.5.x):
    transport.input() -> vad -> stt -> [log ASR] -> user_agg
        -> llm -> [log LLM/TTS] -> tts -> transport.output() -> assistant_agg
"""

from __future__ import annotations

import asyncio
import os
import sys
import time

from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InputAudioRawFrame,
    LLMFullResponseEndFrame,
    OutputAudioRawFrame,
    StartFrame,
    TextFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.groq.llm import GroqLLMService
from pipecat.transports.local.audio import (
    LocalAudioOutputTransport,
    LocalAudioTransport,
    LocalAudioTransportParams,
)
from pipecat.workers.runner import WorkerRunner

from .config import load_settings, require_phase1_keys
from .prompts import GREETING, PHASE1_SYSTEM_PROMPT

# Single source of truth for the output sample rate. Cartesia's TTS output, the pipeline's
# audio_out_sample_rate, and the local speaker stream are all pinned to this so there is no
# hidden resample on the playback path (see the choppy-playback diagnosis).
OUTPUT_SAMPLE_RATE = 24000


def _configure_logging() -> None:
    """Route logging OFF the audio-critical event loop.

    Pipecat installs a synchronous stderr loguru sink on import. Synchronous terminal I/O on
    the event loop that also dispatches 40 ms audio writes injects latency mid-playback ->
    output underruns -> choppy speech. We replace it with a single `enqueue=True` sink (logs
    are handed to a background thread) at INFO, which also drops pipecat's per-utterance DEBUG
    chatter (`Bot started/stopped speaking`, `BotSpeakingFrame`) off the hot path entirely.
    """
    level = os.getenv("CLINIC_LOG_LEVEL", "INFO").upper()
    logger.remove()
    logger.add(sys.stderr, level=level, enqueue=True)


class InstrumentedAudioOutputTransport(LocalAudioOutputTransport):
    """Local speaker output with playback diagnostics (no behavior change).

    Adds the observability the blocking PyAudio API otherwise hides:
      - logs the resolved output sample rate, 40 ms chunk size, and the PyAudio stream's
        output latency (i.e. how small the default buffer really is) on startup;
      - warns on a "feed gap": when the interval between successive chunk writes materially
        exceeds one chunk's duration, that is an audible cutout / output underrun.

    It deliberately does NOT change buffering or pacing — that would be Fix B (an explicit
    jitter buffer), only reached for if these logs still show gaps after Fix A.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._expected_chunk_secs = 0.0
        self._last_write_t: float | None = None

    async def start(self, frame: StartFrame):
        await super().start(frame)  # resolves _sample_rate, opens the stream, sets chunk size
        chunk_bytes = self.audio_chunk_size
        self._expected_chunk_secs = chunk_bytes / (
            self._sample_rate * self._params.audio_out_channels * 2
        )
        try:
            stream_latency_ms = self._out_stream.get_output_latency() * 1000
        except Exception:
            stream_latency_ms = float("nan")
        logger.info(
            f"[audio-out] speaker stream: sample_rate={self._sample_rate} Hz, "
            f"chunk={chunk_bytes} bytes (~{self._expected_chunk_secs * 1000:.0f} ms), "
            f"pyaudio_frames_per_buffer=default/unspecified, "
            f"stream_output_latency={stream_latency_ms:.1f} ms"
        )

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        now = time.monotonic()
        if self._last_write_t is not None:
            gap_ms = (now - self._last_write_t) * 1000
            chunk_ms = self._expected_chunk_secs * 1000
            # Steady playback writes one chunk every ~chunk_ms (blocking write throttles).
            # A gap between ~2x chunk and ~1s is a within-utterance stall (audible cutout);
            # multi-second gaps are just the silence between bot turns, so ignore those.
            if 2 * chunk_ms < gap_ms < 1000:
                logger.warning(
                    f"[audio-out] feed gap {gap_ms:.0f} ms (chunk ~{chunk_ms:.0f} ms) "
                    f"— audible cutout / likely output underrun"
                )
        self._last_write_t = now
        return await super().write_audio_frame(frame)


class InstrumentedLocalAudioTransport(LocalAudioTransport):
    """LocalAudioTransport that returns the instrumented speaker output."""

    def output(self) -> FrameProcessor:
        if not self._output:
            self._output = InstrumentedAudioOutputTransport(self._pyaudio, self._params)
        return self._output


class BotSpeechInputGate(FrameProcessor):
    """Half-duplex mic gate: drop mic audio while the bot is speaking (+ hangover).

    The laptop has no acoustic echo cancellation, so the mic captures the bot's own TTS from
    the speaker. That echo makes the VAD fire a false "user started speaking", which both
    (a) interrupts and cuts the bot off mid-sentence and (b) gets transcribed and answered —
    a feedback loop. Placed BEFORE the VAD, this gate drops InputAudioRawFrames while the bot
    speaks (plus a short hangover for the echo tail), so the echo never reaches VAD or STT.

    This makes the agent half-duplex (no barge-in during bot speech). Real barge-in / VAD
    tuning is Phase 4; this is the minimal Phase-1 echo fix only.
    """

    HANGOVER_SECS = 0.4  # stay muted briefly after the bot stops, to swallow the echo tail

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._bot_speaking = False
        self._unmute_at = 0.0
        self._suppressed = 0

    def _muted(self) -> bool:
        return self._bot_speaking or time.monotonic() < self._unmute_at

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, BotStartedSpeakingFrame):
            if not self._bot_speaking:
                logger.info("[mic-gate] bot started speaking → muting mic")
            self._bot_speaking = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False
            self._unmute_at = time.monotonic() + self.HANGOVER_SECS

        if isinstance(frame, InputAudioRawFrame):
            if self._muted():
                self._suppressed += 1
                # Heartbeat every ~1s (50 x 20ms frames) so suppression is visible in the
                # console without spamming a line per 20ms audio frame.
                if self._suppressed % 50 == 1:
                    logger.info(
                        f"[mic-gate] muting mic while bot speaks (dropped {self._suppressed} frames)"
                    )
                return  # drop: do not forward the bot's echo to VAD/STT
            if self._suppressed:
                logger.info(
                    f"[mic-gate] mic re-opened (dropped {self._suppressed} echo frames total)"
                )
                self._suppressed = 0

        await self.push_frame(frame, direction)


class StageLogger(FrameProcessor):
    """Pass-through processor that logs each pipeline stage to the console.

    So the loop is observable end to end (not just audible): it prints the ASR transcript,
    the assembled LLM response, and when TTS synthesis is triggered. It never mutates or
    drops frames — every frame is forwarded unchanged.
    """

    def __init__(self, *, log_asr: bool = False, log_llm: bool = False, **kwargs):
        super().__init__(**kwargs)
        self._log_asr = log_asr
        self._log_llm = log_llm
        self._llm_response_parts: list[str] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if self._log_asr and isinstance(frame, TranscriptionFrame) and frame.text.strip():
            logger.info(f"ASR  ▶ transcript received: {frame.text!r}")

        if self._log_llm:
            # The LLM streams its answer as TextFrames, terminated by a full-response-end
            # frame. Accumulate the chunks and log the complete reply once, then note that
            # the same text is what flows downstream into TTS.
            if isinstance(frame, TextFrame):
                self._llm_response_parts.append(frame.text)
            elif isinstance(frame, LLMFullResponseEndFrame):
                response = "".join(self._llm_response_parts).strip()
                self._llm_response_parts.clear()
                if response:
                    logger.info(f"LLM  ▶ response generated: {response!r}")
                    logger.info("TTS  ▶ synthesis triggered for LLM response")

        await self.push_frame(frame, direction)


async def run_agent() -> None:
    """Build and run the Phase-1 local voice-agent pipeline."""
    _configure_logging()  # non-blocking sink before anything hits the audio loop

    settings = load_settings()
    require_phase1_keys(settings)  # fail fast on missing keys before opening the mic

    # --- Local audio transport (laptop mic + speaker) ------------------------------------
    # Output sample rate is pinned end to end (transport + pipeline + Cartesia) to avoid any
    # hidden resample on playback. The instrumented transport adds playback diagnostics only.
    transport = InstrumentedLocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_out_sample_rate=OUTPUT_SAMPLE_RATE,
        )
    )

    # --- Services ------------------------------------------------------------------------
    vad = VADProcessor(vad_analyzer=SileroVADAnalyzer())
    stt = DeepgramSTTService(api_key=settings.deepgram_api_key)
    llm = GroqLLMService(
        api_key=settings.groq_api_key,
        settings=GroqLLMService.Settings(model=settings.groq_model),
    )
    tts = CartesiaTTSService(
        api_key=settings.cartesia_api_key,
        sample_rate=OUTPUT_SAMPLE_RATE,  # produce audio at the exact speaker rate
        settings=CartesiaTTSService.Settings(voice=settings.cartesia_voice_id),
    )
    logger.info(
        f"[audio-out] configured rates: pipeline_out={OUTPUT_SAMPLE_RATE} Hz, "
        f"cartesia_tts={OUTPUT_SAMPLE_RATE} Hz (resample on playback path should be a no-op)"
    )

    # --- Conversation context (seeded with the minimal Phase-1 system prompt) ------------
    context = LLMContext(messages=[{"role": "system", "content": PHASE1_SYSTEM_PROMPT}])
    aggregators = LLMContextAggregatorPair(context)

    # --- Pipeline ------------------------------------------------------------------------
    # The mic gate sits BEFORE the VAD so the bot's own echo never reaches VAD/STT.
    pipeline = Pipeline(
        [
            transport.input(),
            BotSpeechInputGate(name="mic-gate"),
            vad,
            stt,
            StageLogger(log_asr=True, name="asr-logger"),
            aggregators.user(),
            llm,
            StageLogger(log_llm=True, name="llm-logger"),
            tts,
            transport.output(),
            aggregators.assistant(),
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            audio_out_sample_rate=OUTPUT_SAMPLE_RATE,
        ),
    )

    @worker.event_handler("on_pipeline_started")
    async def _greet(worker: PipelineWorker, _frame: StartFrame) -> None:
        # pipecat calls this handler as (worker, frame) — both args are required.
        # Deliver the greeting + AI disclosure deterministically (exact wording, every run).
        logger.info("TTS  ▶ synthesis triggered for greeting/disclosure")
        await worker.queue_frames([TTSSpeakFrame(GREETING)])

    logger.info("Starting clinic voice agent (Phase 1 local loop). Speak into your mic; Ctrl-C to stop.")
    await WorkerRunner().run(worker)


def main() -> None:
    asyncio.run(run_agent())


if __name__ == "__main__":
    main()

"""Pipecat pipeline for the clinic voice agent — Phase 5 (telephony mode switch).

ASR->LLM->TTS booking loop that runs over EITHER the laptop mic/speaker (MODE=local, the
Phase-1 path) OR a LiveKit SIP room fed by an inbound phone call (MODE=telephony, Phase 5):

    <transport in> -> Deepgram (ASR) -> Anthropic/Claude (LLM) <-> scheduling API tools
        -> Cartesia (TTS) -> <transport out>

The mode switch (see run_agent) changes ONLY the transport and which event fires the greeting;
the ASR->LLM->TTS chain, scheduling tool calls, echo-safe mic gate, and barge-in are identical
in both modes. On the first caller/mic contact the agent speaks a fixed greeting that includes
the mandatory AI disclosure (spoken deterministically, NOT LLM-generated, so the wording is
exact every run); on telephony that greeting also carries the call-recording consent line
(prompts.greeting_for). It then runs the full slot-fill -> offer -> hold -> confirm -> book ->
close flow, with the LLM deciding when to call check_availability / hold_slot / confirm_booking
(see scheduling_tools.py and the PHASE2 system prompt in prompts.py).

Deliberately OUT of scope here (later phases):
    - observability (latency percentiles, structured logs) + Dockerize/deploy (Phase 6)
    - warm human transfer on escalation                                       (Phase 7)

Pipeline order (Pipecat 1.5.x):
    transport.input() -> mic-gate -> vad -> stt -> [log ASR] -> user_agg
        -> llm (<-> scheduling tools) -> [log LLM/TTS] -> tts -> transport.output() -> assistant_agg
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
    InterruptionFrame,
    LLMFullResponseEndFrame,
    OutputAudioRawFrame,
    StartFrame,
    TextFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
    UserStartedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.transports.base_transport import BaseTransport
from pipecat.transports.livekit.transport import LiveKitParams, LiveKitTransport
from pipecat.transports.local.audio import (
    LocalAudioOutputTransport,
    LocalAudioTransport,
    LocalAudioTransportParams,
)
from pipecat.turns.user_start.min_words_user_turn_start_strategy import (
    MinWordsUserTurnStartStrategy,
)
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.workers.runner import WorkerRunner

from livekit import api as livekit_api

from .barge_in import BargeInConfig, MicGateLogic, barge_in_min_words, frame_rms
from .config import (
    TELEPHONY_ROOM_NAME,
    Settings,
    load_settings,
    require_phase1_keys,
    require_telephony_keys,
)
from .metrics import LatencyCollector, MetricsTap
from .prompts import build_phase2_system_prompt, greeting_for
from .scheduling_tools import (
    SchedulingClient,
    build_tools_schema,
    register_scheduling_functions,
)

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
    """Echo-suppressing mic gate that still allows barge-in (Phase 4).

    The laptop has no acoustic echo cancellation, so the mic captures the bot's own TTS. In
    Phase 1 this gate was fully half-duplex — it dropped ALL mic audio while the bot spoke, so
    the echo never reached VAD/STT, at the cost of no barge-in.

    Phase 4 keeps the echo protection but lets genuine interruptions through: while the bot
    speaks, input is suppressed by default, but SUSTAINED voiced audio (see
    :class:`MicGateLogic`) opens the gate so a real utterance reaches STT and Pipecat's
    ``MinWordsUserTurnStartStrategy`` — which broadcasts the interruption that cancels the bot's
    turn. Short blips / the ~400 ms echo tail stay suppressed. The gate also counts
    false-positive interruptions (echo opened the gate but no real turn followed) as a trust
    metric. All decision logic lives in the dependency-free :class:`MicGateLogic`; this class is
    just the Pipecat frame plumbing + logging.
    """

    def __init__(self, config: BargeInConfig | None = None, **kwargs):
        super().__init__(**kwargs)
        self._logic = MicGateLogic(config or BargeInConfig.from_env())

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        now = time.monotonic()

        if isinstance(frame, BotStartedSpeakingFrame):
            resolved = self._logic.on_bot_started(now)
            logger.info(
                "[mic-gate] bot speaking → guarding mic (sustained speech can still barge in)"
            )
            if resolved == "false_positive":
                logger.info(
                    f"[barge-in] false-positive interruption — echo opened the gate but no real "
                    f"turn followed (total false={self._logic.bargein_false})"
                )
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._logic.on_bot_stopped(now)
        elif isinstance(frame, (InterruptionFrame, UserStartedSpeakingFrame)):
            if self._logic.on_interruption(now) == "true_positive":
                logger.info(
                    f"[barge-in] real interruption — caller barged in over the bot "
                    f"(total real={self._logic.bargein_true})"
                )

        if isinstance(frame, InputAudioRawFrame):
            rms = frame_rms(frame.audio)
            channels = getattr(frame, "num_channels", 1) or 1
            n_samples = (len(frame.audio) // 2) // channels
            frame_ms = (n_samples / frame.sample_rate * 1000.0) if frame.sample_rate else 20.0
            decision = self._logic.on_audio(now, rms, frame_ms)
            if decision == "OPEN":
                logger.info(
                    f"[barge-in] sustained input during bot speech → opening mic "
                    f"(candidate #{self._logic.bargein_candidates})"
                )
            elif decision == "SUPPRESS":
                # Heartbeat every ~1s (50 x 20ms frames) so suppression is visible without
                # spamming a line per 20ms audio frame.
                if self._logic.suppressed_frames % 50 == 1:
                    logger.info(
                        f"[mic-gate] suppressing echo while bot speaks "
                        f"(dropped {self._logic.suppressed_frames} frames)"
                    )
                return  # drop: do not forward the bot's echo to VAD/STT

        await self.push_frame(frame, direction)

    def barge_in_stats(self) -> dict:
        """Resolve any dangling candidate and return the barge-in counters (for a teardown log)."""
        self._logic.resolve_pending_on_shutdown()
        return self._logic.stats()


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


def _build_local_transport() -> InstrumentedLocalAudioTransport:
    """Laptop mic + speaker transport (MODE=local, the Phase-1 path).

    Output sample rate is pinned end to end (transport + pipeline + Cartesia) to avoid any
    hidden resample on playback. The instrumented transport adds playback diagnostics only.
    """
    return InstrumentedLocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_out_sample_rate=OUTPUT_SAMPLE_RATE,
        )
    )


def _livekit_join_token(settings: Settings, room_name: str, identity: str) -> str:
    """Mint a LiveKit room-join JWT for the agent from the API key/secret in .env.

    The agent is just another room participant: it needs a token granting join on the one
    room the inbound SIP call is routed into. The caller joins the same room via the SIP
    dispatch rule (see scripts/setup_livekit_sip.py), and the two meet there.
    """
    return (
        livekit_api.AccessToken(settings.livekit_api_key, settings.livekit_api_secret)
        .with_identity(identity)
        .with_name("Clinic Scheduling Agent")
        .with_grants(livekit_api.VideoGrants(room_join=True, room=room_name))
        .to_jwt()
    )


def _build_livekit_transport(settings: Settings) -> LiveKitTransport:
    """LiveKit SIP-compatible transport (MODE=telephony, the Phase-5 path).

    Same TransportParams as local (audio in/out, pinned 24 kHz out) so the pipeline chain and
    Cartesia rate are unchanged — LiveKit resamples between this room rate and the SIP media
    at its own boundary. Only the ingress/egress moves from the laptop to a LiveKit room.
    """
    token = _livekit_join_token(settings, TELEPHONY_ROOM_NAME, identity="clinic-agent")
    logger.info(
        f"[telephony] joining LiveKit room {TELEPHONY_ROOM_NAME!r} at {settings.livekit_url} "
        f"as 'clinic-agent'; waiting for the inbound SIP caller"
    )
    return LiveKitTransport(
        url=settings.livekit_url,
        token=token,
        room_name=TELEPHONY_ROOM_NAME,
        params=LiveKitParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_out_sample_rate=OUTPUT_SAMPLE_RATE,
        ),
    )


async def run_agent() -> None:
    """Build and run the clinic voice-agent pipeline (local mic or LiveKit SIP telephony)."""
    _configure_logging()  # non-blocking sink before anything hits the audio loop

    settings = load_settings()
    require_phase1_keys(settings)  # fail fast on missing keys before opening the mic

    # --- Transport (mode switch) ---------------------------------------------------------
    # Only the transport layer changes between modes; everything below (ASR->LLM->TTS, tool
    # calls, mic gate, barge-in, greeting/disclosure) is identical. MODE defaults to "local".
    telephony = settings.mode == "telephony"
    if telephony:
        require_telephony_keys(settings)  # fail fast before we try to join the room
        transport: BaseTransport = _build_livekit_transport(settings)
    else:
        transport = _build_local_transport()
    logger.info(f"[mode] MODE={settings.mode!r} → {'LiveKit SIP telephony' if telephony else 'local mic/speaker'}")

    # --- Services ------------------------------------------------------------------------
    vad = VADProcessor(vad_analyzer=SileroVADAnalyzer())
    stt = DeepgramSTTService(api_key=settings.deepgram_api_key)
    # LLM: Anthropic/Claude (active provider). The Phase-2 tool schema + handlers
    # are provider-agnostic — Pipecat's AnthropicLLMAdapter converts build_tools_schema()
    # to Anthropic's input_schema format and routes tool_use/tool_result through the
    # same register_scheduling_functions handlers, so nothing below the service changes.
    llm = AnthropicLLMService(
        api_key=settings.anthropic_api_key,
        settings=AnthropicLLMService.Settings(model=settings.anthropic_model),
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

    # --- Observability (Phase 6) ---------------------------------------------------------
    # One collector per call: MetricsTap processors (inserted in the pipeline below) report the
    # per-turn latency boundaries into it, the scheduling-tool handlers report tool outcomes, and
    # finalize() writes the P50/P95/P99 session summary at hangup. Runs alongside the console logs.
    metrics = LatencyCollector(mode=settings.mode)
    logger.info(f"[metrics] structured latency log → {metrics.log_path} (call_id={metrics.call_id})")

    # --- Scheduling-API tools (Phase 2) --------------------------------------------------
    # The LLM decides when to call these; the client is the HTTP plumbing to the mock API.
    scheduling_client = SchedulingClient(settings.scheduling_api_base_url)
    tools = build_tools_schema()
    register_scheduling_functions(llm, scheduling_client, collector=metrics)
    logger.info(
        f"[tools] registered scheduling functions (check_availability, hold_slot, "
        f"confirm_booking) → {settings.scheduling_api_base_url}"
    )

    # --- Conversation context (Phase-2 system prompt + tools) ----------------------------
    # The prompt embeds today's clinic-local date plus a 14-day date->weekday table, which is what
    # lets relative phrases ("next Tuesday") resolve to the concrete date the scheduling API
    # filters on. That makes the prompt TIME-SENSITIVE, and the telephony worker is long-lived: it
    # sits idle for days waiting for inbound calls, so a prompt built once at process start is
    # already wrong after the first midnight -- the model would resolve "tomorrow" against a stale
    # table and book the wrong day. Built here so nothing downstream ever sees an empty context,
    # then rebuilt at the start of each call by _reset_context_for_call() below.
    context = LLMContext(
        messages=[{"role": "system", "content": build_phase2_system_prompt()}],
        tools=tools,
    )

    def _reset_context_for_call() -> None:
        """Rebuild the system prompt with a current date table, at the start of each call.

        `messages` is a read-only property on LLMContext, so the supported way to replace the
        conversation is set_messages(). Replacing (rather than appending) also drops any prior
        call's turns, which matters on telephony where one process can serve a second caller.
        NOTE: this resets the *conversation* only -- the LatencyCollector is still per-process,
        so a second call still reports into the first call's collector. Per-call session state is
        Phase 10 (the in-house event loop); this is the narrow date-correctness fix.
        """
        context.set_messages([{"role": "system", "content": build_phase2_system_prompt()}])
    # Barge-in (Phase 4): the user aggregator's built-in turn controller broadcasts an
    # interruption on user-turn-start, which cancels the bot's in-flight TTS/LLM. We gate WHEN a
    # turn starts with MinWordsUserTurnStartStrategy: while the bot speaks it needs >= min_words
    # transcribed words (so short sounds / echo blips don't interrupt); while the bot is silent a
    # single word starts a normal turn. This pairs with the audio-level BotSpeechInputGate below.
    min_words = barge_in_min_words()
    user_params = LLMUserAggregatorParams(
        user_turn_strategies=UserTurnStrategies(
            start=[MinWordsUserTurnStartStrategy(min_words=min_words, use_interim=True)],
        ),
    )
    aggregators = LLMContextAggregatorPair(context, user_params=user_params)
    logger.info(
        f"[barge-in] turn gate: MinWordsUserTurnStartStrategy(min_words={min_words}); "
        f"interruptions enabled"
    )

    # --- Pipeline ------------------------------------------------------------------------
    # The mic gate sits BEFORE the VAD so the bot's own echo never reaches VAD/STT.
    mic_gate = BotSpeechInputGate(name="mic-gate")
    pipeline = Pipeline(
        [
            transport.input(),
            mic_gate,
            vad,
            stt,
            StageLogger(log_asr=True, name="asr-logger"),
            # metrics-asr sits BEFORE the user aggregator so it sees the final TranscriptionFrame
            # (and UserStoppedSpeakingFrame) before the aggregator consumes them.
            MetricsTap(metrics, name="metrics-asr"),
            aggregators.user(),
            llm,
            StageLogger(log_llm=True, name="llm-logger"),
            MetricsTap(metrics, name="metrics-llm"),
            tts,
            # metrics-tts sits between TTS and the transport so it timestamps the first audio
            # frame of the reply (turn end) before it leaves for the speaker/SIP egress.
            MetricsTap(metrics, name="metrics-tts"),
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
        # Telephony waits (possibly minutes) for an inbound call with no audio flowing. Pipecat's
        # default 300s idle timeout would cancel the worker before the caller ever dials, so
        # disable idle cancellation on the telephony path — the agent must wait indefinitely for
        # the inbound SIP call. Local mic keeps the default safety timeout.
        idle_timeout_secs=None if telephony else 300,
    )

    # --- Greeting trigger (mode-dependent event) -----------------------------------------
    # The greeting is spoken deterministically (exact AI-disclosure wording every run; on
    # telephony it also carries the call-recording consent — see greeting_for()). WHICH event
    # fires it differs by mode because the two transports reach "a caller is present" at
    # different moments:
    #   - local:     the mic opens the instant the pipeline starts, so the caller is already
    #                "there" at on_pipeline_started. Greet immediately.
    #   - telephony: the agent joins the LiveKit room at pipeline start, but the room is EMPTY
    #                until the inbound SIP call connects. Greeting at pipeline-start would talk
    #                to an empty room, so we wait for on_first_participant_joined (the caller
    #                actually arriving) before speaking. This is the one behavioral difference
    #                the room-based transport forces; the greeting text/flow are otherwise the
    #                same.
    greeting = greeting_for(settings.mode)

    async def _speak_greeting() -> None:
        # Both modes converge here at "a caller is present", which is the correct moment to
        # rebuild the date-sensitive system prompt (see _reset_context_for_call above).
        _reset_context_for_call()
        logger.info("TTS  ▶ synthesis triggered for greeting/disclosure")
        await worker.queue_frames([TTSSpeakFrame(greeting)])

    if telephony:

        @transport.event_handler("on_first_participant_joined")
        async def _greet_on_join(_transport: BaseTransport, participant_id: str) -> None:
            logger.info(f"[telephony] caller joined room (participant={participant_id})")
            await _speak_greeting()

        @transport.event_handler("on_participant_disconnected")
        async def _finalize_on_hangup(_transport: BaseTransport, participant_id: str) -> None:
            # Caller hung up: write the session summary right now so the P50/P95/P99 console line
            # lands at the moment the call ends (finalize() is idempotent — the finally-block
            # call below is a no-op after this).
            logger.info(f"[telephony] caller left room (participant={participant_id})")
            metrics.finalize()

    else:

        @worker.event_handler("on_pipeline_started")
        async def _greet(_worker: PipelineWorker, _frame: StartFrame) -> None:
            # pipecat calls this handler as (worker, frame) — both args are required.
            await _speak_greeting()

    logger.info(
        "Starting clinic voice agent (Phase 2: booking via scheduling-API tools). "
        + ("Waiting for an inbound call; Ctrl-C to stop." if telephony
           else "Speak into your mic; Ctrl-C to stop.")
    )
    try:
        await WorkerRunner().run(worker)
    finally:
        stats = mic_gate.barge_in_stats()
        logger.info(
            f"[barge-in] session totals: candidates={stats['bargein_candidates']}, "
            f"real={stats['bargein_true']}, false-positive={stats['bargein_false']}, "
            f"suppressed_echo_frames={stats['suppressed_frames']}"
        )
        metrics.finalize()  # write the latency/outcome session summary (idempotent)
        await scheduling_client.aclose()  # close the HTTP client even on Ctrl-C / error


def main() -> None:
    asyncio.run(run_agent())


if __name__ == "__main__":
    main()

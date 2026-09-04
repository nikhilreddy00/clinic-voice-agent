"""Phase 10 — CallSession: the drain loop that ties the engine together.

One ``CallSession`` == one call. It owns a single ``asyncio.Queue`` of events, the adapter set,
and the loop that does the only thing this engine does:

    event = await queue.get()  ->  record  ->  meter  ->  reduce()  ->  execute actions

Everything else is an adapter turning an action into I/O and pushing the result back as an
event.

**Every piece of per-call state is constructed here.** That is the concrete defect this phase
exists to fix: through Phase 9 the ``LatencyCollector``, ``SchedulingClient``, ``LLMContext``,
and mic gate were function-locals of ``pipeline.run_agent()``, built once per *process*. On the
telephony path one process serves whatever calls arrive, so caller #2 inherited caller #1's
message history and wrote turns into an already-finalized metrics summary. Here two sessions
in one process share nothing but the event loop — which is what Phase 11 needs in order to put
N calls in one worker.

Adapters are wired with plain callbacks rather than a registry: ``media -> TurnEngine -> STT``
for audio in, and ``TTS -> media`` for audio out. Audio never becomes an event.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import replace
from datetime import datetime, timezone

from loguru import logger

from livekit import api as livekit_api

from ..config import TELEPHONY_ROOM_NAME, Settings
from ..metrics import LatencyCollector
from ..prompts import build_phase2_system_prompt
from ..scheduling_tools import SchedulingClient, build_tools_schema
from . import events as ev
from . import telemetry
from .actions import (
    Action,
    CancelLLM,
    CancelSpeech,
    ClassifyIntent,
    EndCall,
    InvokeTool,
    LoadCallerMemory,
    Speak,
    StartLLM,
    TransferToHuman,
)
from .adapters.classifier import IntentClassifier
from .adapters.llm import AnthropicLLM, shared_anthropic_client
from .adapters.media import LiveKitMedia, LocalMedia, MediaAdapter
from .adapters.stt import DeepgramSTT
from .adapters.tools import ToolExecutor
from .adapters.tts import CartesiaTTS
from .adapters.turn import TurnEngine
from .llm_router import LLMRouter
from .recorder import TraceRecorder
from .reducer import reduce
from .state import CallState, Phase


def _livekit_join_token(settings: Settings, room_name: str, identity: str) -> str:
    """Mint a LiveKit room-join JWT for the agent from the API key/secret in .env."""
    return (
        livekit_api.AccessToken(settings.livekit_api_key, settings.livekit_api_secret)
        .with_identity(identity)
        .with_name("Clinic Scheduling Agent")
        .with_grants(livekit_api.VideoGrants(room_join=True, room=room_name))
        .to_jwt()
    )


# Above this, a stall is long enough to be a real cause of a gap between LLM text deltas
# rather than ordinary scheduling jitter. Sampling itself is ~free; the logging is not, so
# the threshold keeps a live call from writing a line 20x a second.
_LOOP_LAG_WARN_MS = float(os.getenv("CLINIC_LOOP_LAG_WARN_MS", "50"))


class CallSession:
    """One voice call, end to end."""

    def __init__(self, settings: Settings, *, call_id: str | None = None, record: bool = True) -> None:
        self.settings = settings
        self.state = CallState()
        self._queue: asyncio.Queue[ev.Event] = asyncio.Queue()
        self._seq = 0
        self._closed = asyncio.Event()

        self.metrics = LatencyCollector(mode=settings.mode, call_id=call_id)
        self.call_id = self.metrics.call_id
        self._recorder = TraceRecorder(self.call_id, enabled=record)

        # --- adapters -------------------------------------------------------------------
        # Built through overridable factories rather than inline, so a subclass can swap any
        # one of them. That seam is what the Phase-11 Tier-A load test needs: driving 1,000
        # synthetic sessions means fake STT/LLM/TTS adapters that inject realistic latency
        # distributions, exercising this exact loop with no vendor spend. Tests use it too.
        #
        # Construction order matters: media owns playback, so TTS needs it; STT owns the
        # socket, so the TurnEngine needs it.
        # Pinned once per call so every per-intent prompt shares one date table, and so a long
        # call cannot silently re-ground itself across midnight mid-conversation.
        self._now = datetime.now(timezone.utc)
        self._router = LLMRouter()
        self._llm = self._build_llm()
        self._classifier = self._build_classifier()
        self._tools = self._build_tools()
        self.media: MediaAdapter = self._build_media()
        self._tts = self._build_tts()
        self._stt = self._build_stt()
        self._turn = self._build_turn()
        # Phase-14 finding 2: is the loop blocked while LLM deltas queue up, or does the
        # wire deliver them in bursts? Off by default — it is a diagnostic, not a feature.
        self._loop_lag = (
            telemetry.LoopLagMonitor(warn_over_ms=_LOOP_LAG_WARN_MS)
            if os.getenv("CLINIC_LOOP_LAG") == "1"
            else None
        )

    def _build_llm(self):
        """The system prompt is built HERE, per call, not per process.

        It embeds today's clinic-local date plus a 14-day date→weekday table, so a long-lived
        telephony worker idle since yesterday would otherwise resolve "tomorrow" against a
        stale table and book the wrong day.
        """
        return AnthropicLLM(
            api_key=self.settings.anthropic_api_key,
            model=self.settings.anthropic_model,
            system_prompt=build_phase2_system_prompt(self._now),
            tools=build_tools_schema(),
            emit=self.emit,
            router=self._router,
            now=self._now,
            client=shared_anthropic_client(self.settings.anthropic_api_key),
        )

    def _build_classifier(self):
        """Intent classifier — always the fast tier, always off the critical path."""
        return IntentClassifier(
            api_key=self.settings.anthropic_api_key,
            spec=self._router.classifier_spec(),
            emit=self.emit,
            client=shared_anthropic_client(self.settings.anthropic_api_key),
        )

    def _build_tools(self):
        # call_id scopes the client's Idempotency-Keys to this call, so a retry after a
        # timed-out voice turn replays the original booking instead of creating a second
        # appointment, while two concurrent callers never collide on a key.
        return ToolExecutor(
            SchedulingClient(self.settings.scheduling_api_base_url, call_id=self.call_id),
            emit=self.emit,
            collector=self.metrics,
        )

    def _build_media(self) -> MediaAdapter:
        if self.settings.mode == "telephony":
            token = _livekit_join_token(self.settings, TELEPHONY_ROOM_NAME, "clinic-agent")
            return LiveKitMedia(
                emit=self.emit,
                on_audio=self._on_audio,
                url=self.settings.livekit_url,
                token=token,
                room_name=TELEPHONY_ROOM_NAME,
            )
        return LocalMedia(emit=self.emit, on_audio=self._on_audio)

    def _build_tts(self):
        return CartesiaTTS(
            api_key=self.settings.cartesia_api_key,
            voice_id=self.settings.cartesia_voice_id,
            emit=self.emit,
            play=self.media.play,
            end_utterance=self.media.end_utterance,
            clear_playback=self.media.clear,
        )

    def _build_stt(self):
        return DeepgramSTT(
            api_key=self.settings.deepgram_api_key,
            emit=self.emit,
            on_partial=lambda text, now: self._turn.on_partial_transcript(text, now),
        )

    def _build_turn(self) -> TurnEngine:
        return TurnEngine(emit=self.emit, forward_audio=self._stt.send_audio)

    async def _on_audio(self, pcm: bytes, now: float) -> None:
        await self._turn.feed(pcm, now)

    # --- event intake -------------------------------------------------------------------

    def emit(self, event: ev.Event) -> None:
        """Accept an event from an adapter. Synchronous so callbacks can call it directly.

        ``seq`` is stamped here rather than by the producer: adapters run as several concurrent
        tasks, and the order events are accepted onto this queue *is* the order the reducer
        sees them, which is exactly what a replay has to reproduce.
        """
        self._seq += 1
        self._queue.put_nowait(replace(event, seq=self._seq))

    # --- lifecycle ----------------------------------------------------------------------

    async def run(self) -> None:
        """Start the adapters and drain events until the call closes."""
        logger.info(
            f"[session] call_id={self.call_id} mode={self.settings.mode!r} "
            f"engine=core (in-house event loop)"
        )
        logger.info(f"[metrics] structured latency log → {self.metrics.log_path}")
        if self._recorder.enabled:
            logger.info(f"[trace] event trace → {self._recorder.path}")

        # CallStarted MUST be enqueued before any adapter can emit. Starting the media adapter
        # can produce CallerPresent synchronously — the local mic is live the moment its stream
        # opens, and on telephony a SIP call can already be bridged into the room before the
        # agent finishes connecting. If CallerPresent were reduced first, the greeting would be
        # chosen against the default mode and a telephony caller would never hear the
        # call-recording consent line. That is a governance failure, not a cosmetic ordering nit.
        self.emit(ev.CallStarted(t=time.monotonic(), call_id=self.call_id, mode=self.settings.mode))

        if self._loop_lag is not None:
            self._loop_lag.start()
            logger.info(f"[loop] lag sampling on; warning over {_LOOP_LAG_WARN_MS:.0f} ms")

        await self._stt.start()
        await self._tts.start()
        await self.media.start()

        try:
            await self._drain()
        finally:
            await self._teardown()

    async def _drain(self) -> None:
        while True:
            event = await self._queue.get()
            self._recorder.record(event)
            telemetry.record_event(self.metrics, event)
            self._notify_turn_engine(event)

            self.state, actions = reduce(self.state, event)
            for action in actions:
                await self._execute(action)

            if self.state.phase is Phase.CLOSED:
                return

    def _notify_turn_engine(self, event: ev.Event) -> None:
        """Keep the mic gate in step with playback.

        Driven from the same event stream the reducer sees rather than from the TTS adapter, so
        the gate opens and closes at exactly the moments the state machine believes it does.
        """
        if isinstance(event, ev.BotStartedSpeaking):
            self._turn.on_bot_started(event.t)
        elif isinstance(event, ev.BotStoppedSpeaking):
            self._turn.on_bot_stopped(event.t)

    async def _execute(self, action: Action) -> None:
        if isinstance(action, StartLLM):
            self._llm.start(
                action.request_id,
                action.messages,
                tier=action.tier,
                intent=action.intent,
                routing_reason=action.routing_reason,
                context_note=action.context_note,
            )
        elif isinstance(action, ClassifyIntent):
            self._classifier.classify(action.utterance)
        elif isinstance(action, TransferToHuman):
            # No live transfer exists in this build — Phase 15 implements the warm handoff over
            # LiveKit SIP. Logged at WARNING because a silent no-op here would look like a
            # working escalation in the logs of a call where nobody was actually reached.
            level = logger.error if action.urgent else logger.warning
            level(
                f"[transfer] NOT IMPLEMENTED — would transfer to a human "
                f"(reason={action.reason!r}, urgent={action.urgent}): {action.summary}"
            )
            self.state = replace(self.state, escalated=True)
        elif isinstance(action, CancelLLM):
            self._llm.cancel(action.request_id)
        elif isinstance(action, Speak):
            if action.deterministic:
                logger.info(f"TTS  ▶ speaking scripted line: {action.text!r}")
            await self._tts.speak(action.utterance_id, action.text, final=action.final)
        elif isinstance(action, CancelSpeech):
            await self._tts.cancel(action.utterance_id)
        elif isinstance(action, InvokeTool):
            self._tools.invoke(action.tool_call_id, action.name, action.arguments)
        elif isinstance(action, LoadCallerMemory):
            # Fire-and-forget, in parallel with the greeting: the caller must never wait on a
            # database to hear the AI disclosure, and a lookup that fails just means the agent
            # greets exactly as it did before Phase 13.
            self._tools.load_caller_memory(action.phone)
        elif isinstance(action, EndCall):
            logger.info(f"[session] ending call ({action.reason})")
            self._closed.set()

    def hangup(self, reason: str = "shutdown") -> None:
        """Close the call from outside the loop (Ctrl-C, worker shutdown)."""
        self.emit(ev.Hangup(t=time.monotonic(), reason=reason))

    async def _teardown(self) -> None:
        stats = self._turn.stats()
        logger.info(
            f"[barge-in] session totals: candidates={stats['bargein_candidates']}, "
            f"real={stats['bargein_true']}, false-positive={stats['bargein_false']}, "
            f"suppressed_echo_frames={stats['suppressed_frames']}"
        )
        logger.info(
            f"[session] outcome={self.state.outcome} turns={self.state.turn_index} "
            f"interruptions={self.state.interruptions} events={self._seq} "
            f"intent={self.state.intent.value if self.state.intent else 'unclassified'}"
            + (f" EMERGENCY({self.state.emergency_category})" if self.state.emergency else "")
        )
        if self._loop_lag is not None:
            await self._loop_lag.stop()
            if summary := self._loop_lag.summary():
                logger.info(summary)
        self.metrics.finalize()
        self._recorder.close()  # flush the buffered tail and release the descriptor

        for closer in (self._stt.aclose, self._tts.aclose, self.media.aclose,
                       self._tools.aclose, self._llm.aclose, self._classifier.aclose):
            try:
                await closer()
            except Exception as exc:  # noqa: BLE001 - one failed close must not skip the rest
                logger.warning(f"[session] teardown error in {closer.__qualname__}: {exc}")

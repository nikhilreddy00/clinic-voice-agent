"""Phase 10 — Phase-6 latency metrics, re-sourced from events.

The Phase-6 :class:`~clinic_agent.metrics.LatencyCollector` is kept exactly as it is; only its
input changes. It used to be fed by three ``MetricsTap`` frame processors spliced into the
Pipecat pipeline at the STT/LLM/TTS boundaries. Here it is a plain subscriber on the one event
stream, which is strictly better: the boundaries are named events rather than frame classes
observed at the right position, so the failure mode that made Phase 6 record ``turns=0`` —
watching ``UserStoppedSpeakingFrame`` when the pipeline actually emits
``VADUserStoppedSpeakingFrame`` — cannot happen. There is one ``SpeechStopped`` event and it
means one thing.

Stage boundaries are unchanged, so the numbers stay comparable across engines:

    SpeechStopped                   -> turn starts (VAD silence)
    FinalTranscript                 -> ASR done
    first LLMTextDelta / LLMToolUse -> LLM first output (or first tool call)
    LLMCompleted                    -> LLM reply complete
    BotStartedSpeaking              -> first audio out, turn finalizes

Tool outcomes are recorded inside ``scheduling_tools.execute_tool`` on both engines, so they
are not duplicated here.
"""

from __future__ import annotations

import asyncio
import time

from loguru import logger

from ..metrics import LatencyCollector
from . import events as ev


def record_event(collector: LatencyCollector, event: ev.Event) -> None:
    """Map one event to its LatencyCollector hook. Pure dispatch — unit-testable, no I/O."""
    if isinstance(event, ev.SpeechStopped):
        collector.on_user_stopped(event.t)
    elif isinstance(event, ev.FinalTranscript):
        if event.text.strip():
            collector.on_transcript(event.t, event.confidence)
    elif isinstance(event, ev.LLMToolUse):
        collector.on_llm_first_output(event.t, is_tool_call=True)
    elif isinstance(event, ev.LLMTextDelta):
        collector.on_llm_first_output(event.t)
    elif isinstance(event, ev.LLMCompleted):
        collector.on_llm_end(event.t)
    elif isinstance(event, ev.BotStartedSpeaking):
        collector.on_output_audio(event.t)


# --- event-loop lag -------------------------------------------------------------------------
#
# Phase 14, finding 2. On the 2026-09-04 calls the LLM's text deltas arrived in bursts — one
# character, a 100-350 ms gap, then 50-100 characters at once — and `adapters/llm.py` emits one
# event per SDK delta with no batching anywhere in between. Either the wire delivers like that,
# or this loop is blocked while the deltas queue up in the socket. Those have completely
# different fixes, and speculative LLM start is only worth building if it is the former.
#
# The sampler is the same one `loadtest/tier_a.py` has used since Phase 11 — it lives here now
# so a live call and the load test report the same number the same way, rather than the agent
# growing a second, subtly different copy.


class LoopLagMonitor:
    """Measures how late the event loop wakes a task that asked for a fixed sleep.

    The most honest single number for orchestrator saturation. Latency percentiles include
    provider time and so move slowly; lag is pure scheduling delay and moves first.

    ``warn_over_ms`` logs the outliers as they happen. That is the point on a live call: a
    percentile cannot be lined up against a delta gap in a trace, but a timestamped
    ``[loop] blocked 180 ms`` line can.
    """

    def __init__(self, interval: float = 0.05, *, warn_over_ms: float | None = None) -> None:
        self.interval = interval
        self.warn_over_ms = warn_over_ms
        self.samples: list[float] = []
        self._task: asyncio.Task | None = None
        self._stop = False

    async def _run(self) -> None:
        while not self._stop:
            start = time.monotonic()
            await asyncio.sleep(self.interval)
            lag = (time.monotonic() - start - self.interval) * 1000
            self.samples.append(lag)
            if self.warn_over_ms is not None and lag > self.warn_over_ms:
                logger.warning(f"[loop] blocked {lag:.0f} ms")

    def start(self) -> None:
        self._stop = False
        self.samples.clear()
        self._task = asyncio.create_task(self._run(), name="loop-lag")

    async def stop(self) -> None:
        self._stop = True
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    def summary(self) -> str:
        """One human line. Empty string if nothing was sampled."""
        if not self.samples:
            return ""
        ordered = sorted(self.samples)
        p = lambda q: ordered[min(len(ordered) - 1, int(len(ordered) * q))]  # noqa: E731
        return (
            f"[loop] lag over {len(ordered)} samples: "
            f"p50 {p(0.50):.1f} ms  p95 {p(0.95):.1f} ms  max {ordered[-1]:.0f} ms"
        )

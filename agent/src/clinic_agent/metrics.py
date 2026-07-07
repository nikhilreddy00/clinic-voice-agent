"""Phase 6 — per-turn latency + call-outcome observability (structured JSON sink).

This module captures the four voice-to-voice latencies the README reports, plus per-turn ASR
confidence, per-call tool outcomes, and the call outcome at hangup. It writes a newline-delimited
JSON log (`logs/calls.jsonl`, one object per event) that the scheduling API's `/metrics` endpoint
aggregates for the dashboard. It runs ALONGSIDE the existing human-readable console logs
(`ASR ▶ / LLM ▶ / TOOL ▶ / TTS ▶`) — it never replaces or mutates them, and the taps never drop
or change a frame.

A *turn* spans from the caller's VAD silence to the first audio frame of the bot's reply:

    VADUserStoppedSpeakingFrame     -> t_user_stop  (VAD silence, turn starts)
    final TranscriptionFrame        -> t_transcript (ASR done)
    first LLMTextFrame / tool call  -> t_llm_first  (LLM first token OR tool call fired)
    LLMFullResponseEndFrame         -> t_llm_end    (LLM reply complete; last before audio wins)
    first OutputAudioRawFrame        -> t_audio_out  (first audio out, turn finalizes)

    asr_ms = t_transcript - t_user_stop
    llm_ms = t_llm_first  - t_transcript      (on tool-call turns this ENDS at the first tool
                                               call, not the final response — e2e_ms is the
                                               honest number for those turns)
    tts_ms = t_audio_out  - t_llm_end         (null if TTS streaming started before the LLM
                                               finished — an overlap, not a measurable gap)
    e2e_ms = t_audio_out  - t_user_stop       (voice-to-voice; always exact)

Deltas use time.monotonic() (like the rest of the pipeline); record timestamps use wall-clock ISO.
The deterministic greeting has no preceding UserStoppedSpeakingFrame, so it emits no turn.
"""

from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from pipecat.frames.frames import (
    Frame,
    FunctionCallInProgressFrame,
    LLMFullResponseEndFrame,
    LLMTextFrame,
    OutputAudioRawFrame,
    TranscriptionFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

# The four per-turn latency stages, in pipeline order. Used as dict keys everywhere so the
# agent-side collector and the API-side aggregator stay in lockstep.
STAGES = ("asr", "llm", "tts", "e2e")


def resolve_log_dir() -> Path:
    """Directory holding calls.jsonl, shared between the agent (writer) and API (reader).

    `CLINIC_LOG_DIR` overrides (set to a shared volume path under Docker). Default resolves to
    the repo-root `logs/` regardless of the process's cwd, so the agent (run from agent/) and the
    scheduling API (run from scheduling_api/) both land on the same file in local dev.
    """
    env = os.getenv("CLINIC_LOG_DIR")
    if env:
        return Path(env)
    # metrics.py -> clinic_agent -> src -> agent -> <repo root>
    return Path(__file__).resolve().parents[3] / "logs"


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def percentiles(values: list[float]) -> dict:
    """Nearest-rank P50/P95/P99 (rounded to whole ms) over a list of millisecond values.

    Nearest-rank (not interpolated) keeps small-N percentiles honest and matches the API-side
    aggregator. Returns None percentiles for an empty list so callers can render "n/a".
    """
    if not values:
        return {"p50": None, "p95": None, "p99": None, "count": 0}
    ordered = sorted(values)

    def rank(p: float) -> int:
        return max(0, math.ceil(p / 100 * len(ordered)) - 1)

    return {
        "p50": round(ordered[rank(50)]),
        "p95": round(ordered[rank(95)]),
        "p99": round(ordered[rank(99)]),
        "count": len(ordered),
    }


def _confidence_stats(values: list[float]) -> dict:
    if not values:
        return {"avg": None, "min": None, "max": None, "count": 0}
    return {
        "avg": round(sum(values) / len(values), 3),
        "min": round(min(values), 3),
        "max": round(max(values), 3),
        "count": len(values),
    }


class LatencyCollector:
    """Accumulates per-turn latencies + tool/outcome signals for one call and writes JSONL.

    One instance per call. The MetricsTap processors and the scheduling-tool handlers report
    events into it; `finalize()` (called once at hangup) writes the call summary line and the
    human-readable console session line. All writes are append-only to a single file that only
    this process writes (one agent process per voice session), so there is no write contention.
    """

    def __init__(self, *, mode: str, call_id: str | None = None) -> None:
        self.mode = mode
        self.call_id = call_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        self.log_path = resolve_log_dir() / "calls.jsonl"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        self._reset_turn()

        # Cross-turn aggregates for this call.
        self._samples: dict[str, list[float]] = {s: [] for s in STAGES}
        self._confidences: list[float] = []
        self._confidence_seen = False
        self._turn_count = 0
        self._tool_total = 0
        self._tool_success = 0
        self._booked = False
        self._escalated = False
        self._finalized = False

    # --- per-turn state -----------------------------------------------------------------
    def _reset_turn(self) -> None:
        self._t_user_stop: float | None = None
        self._t_transcript: float | None = None
        self._t_llm_first: float | None = None
        self._t_llm_end: float | None = None
        self._confidence: float | None = None
        self._had_tool_call = False

    # --- event hooks (called by the taps) -----------------------------------------------
    def on_user_stopped(self, now: float) -> None:
        """VAD silence — a new turn begins (resets any half-finished pending turn)."""
        self._reset_turn()
        self._t_user_stop = now

    def on_transcript(self, now: float, confidence: float | None) -> None:
        if self._t_user_stop is None or self._t_transcript is not None:
            return  # no pending turn, or already have the first final transcript
        self._t_transcript = now
        self._confidence = confidence
        if confidence is not None:
            self._confidence_seen = True

    def on_llm_first_output(self, now: float, *, is_tool_call: bool = False) -> None:
        if is_tool_call:
            self._had_tool_call = True
        if self._t_user_stop is None or self._t_transcript is None:
            return
        if self._t_llm_first is None:
            self._t_llm_first = now

    def on_llm_end(self, now: float) -> None:
        if self._t_user_stop is not None:
            self._t_llm_end = now  # last LLM-response-end before audio wins

    def on_output_audio(self, now: float) -> None:
        """First audio frame of the reply — finalize and emit the turn."""
        if self._t_user_stop is None or self._t_transcript is None:
            return  # greeting audio, or a turn with no transcript (e.g. barge-in) — skip

        asr_ms = (self._t_transcript - self._t_user_stop) * 1000
        e2e_ms = (now - self._t_user_stop) * 1000
        llm_ms = (self._t_llm_first - self._t_transcript) * 1000 if self._t_llm_first else None
        tts_ms = (now - self._t_llm_end) * 1000 if self._t_llm_end else None
        # If TTS began streaming before the LLM finished, tts_ms is negative — an overlap, not a
        # measurable latency. Record null + a flag rather than a misleading negative number.
        tts_overlap = tts_ms is not None and tts_ms < 0
        if tts_overlap:
            tts_ms = None

        for stage, value in (("asr", asr_ms), ("llm", llm_ms), ("tts", tts_ms), ("e2e", e2e_ms)):
            if value is not None:
                self._samples[stage].append(value)
        if self._confidence is not None:
            self._confidences.append(self._confidence)
        self._turn_count += 1

        self._write(
            {
                "event": "turn",
                "call_id": self.call_id,
                "ts": _iso_now(),
                "asr_ms": _round_or_none(asr_ms),
                "llm_ms": _round_or_none(llm_ms),
                "tts_ms": _round_or_none(tts_ms),
                "e2e_ms": _round_or_none(e2e_ms),
                "asr_confidence": self._confidence,
                "had_tool_call": self._had_tool_call,
                "tts_streaming_overlap": tts_overlap,
            }
        )
        self._reset_turn()

    def record_tool(
        self, endpoint: str, http_status: int | None, latency_ms: float, success: bool
    ) -> None:
        self._tool_total += 1
        if success:
            self._tool_success += 1
        if endpoint == "/confirm-booking" and success:
            self._booked = True  # booked overrides escalated at outcome resolution
        self._write(
            {
                "event": "tool",
                "call_id": self.call_id,
                "ts": _iso_now(),
                "endpoint": endpoint,
                "http_status": http_status,
                "latency_ms": round(latency_ms),
                "success": success,
            }
        )

    def mark_escalation(self) -> None:
        """A no-availability path fired (agent hand-off). Overridden by a later successful book."""
        self._escalated = True

    # --- teardown -----------------------------------------------------------------------
    def _outcome(self) -> str:
        if self._booked:
            return "booked"
        if self._escalated:
            return "escalated"
        return "abandoned"

    def finalize(self) -> None:
        """Write the call summary (JSON + human console line). Idempotent — safe to call twice."""
        if self._finalized:
            return
        self._finalized = True

        outcome = self._outcome()
        latency = {s: percentiles(self._samples[s]) for s in STAGES}
        confidence = _confidence_stats(self._confidences)

        self._write(
            {
                "event": "call_summary",
                "call_id": self.call_id,
                "ts": _iso_now(),
                "mode": self.mode,
                "outcome": outcome,
                "turns": self._turn_count,
                "latency_ms": latency,
                "asr_confidence": confidence,
                "tool_calls": {"total": self._tool_total, "success": self._tool_success},
            }
        )
        logger.info(self._human_summary(outcome, latency))
        if self._turn_count > 0 and not self._confidence_seen:
            logger.warning(
                "[session] ASR confidence was never present on any transcript this session — "
                "Deepgram may not be surfacing it; dashboard confidence stats will be empty."
            )

    def _human_summary(self, outcome: str, latency: dict) -> str:
        def stage(name: str, key: str) -> str:
            p = latency[key]
            fmt = lambda v: f"{v}ms" if v is not None else "n/a"  # noqa: E731
            return f"{name} p50={fmt(p['p50'])} p95={fmt(p['p95'])}"

        return (
            f"[session] outcome={outcome} turns={self._turn_count} | "
            f"{stage('ASR', 'asr')} | {stage('LLM', 'llm')} | "
            f"{stage('TTS', 'tts')} | {stage('E2E', 'e2e')}"
        )

    def _write(self, obj: dict) -> None:
        try:
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(obj) + "\n")
        except OSError as exc:
            # Observability must never take down the call: log and carry on.
            logger.warning(f"[metrics] could not write {self.log_path}: {exc}")


def _round_or_none(value: float | None) -> int | None:
    return round(value) if value is not None else None


def _extract_confidence(frame: TranscriptionFrame) -> float | None:
    """Pull Deepgram's per-utterance confidence off a final TranscriptionFrame, defensively.

    Deepgram puts it at result.channel.alternatives[0].confidence. If the provider doesn't
    surface it on this frame, degrade to None — a missing confidence must never drop or corrupt
    the turn record.
    """
    try:
        return float(frame.result.channel.alternatives[0].confidence)
    except (AttributeError, IndexError, TypeError, ValueError):
        return None


def record_frame_event(collector: LatencyCollector, frame: Frame, now: float) -> None:
    """Map a single pipeline frame to the matching LatencyCollector hook (pure — unit-testable).

    Kept out of the FrameProcessor so it can be tested against real frame instances without any
    pipeline plumbing — this is exactly the boundary that regressed when the VAD frame class was
    wrong (VADProcessor emits VADUserStoppedSpeakingFrame, NOT UserStoppedSpeakingFrame).
    """
    if isinstance(frame, VADUserStoppedSpeakingFrame):
        collector.on_user_stopped(now)  # VAD silence — turn starts
    elif isinstance(frame, TranscriptionFrame) and frame.text.strip():
        collector.on_transcript(now, _extract_confidence(frame))
    elif isinstance(frame, FunctionCallInProgressFrame):
        collector.on_llm_first_output(now, is_tool_call=True)
    elif isinstance(frame, LLMTextFrame):
        collector.on_llm_first_output(now)
    elif isinstance(frame, LLMFullResponseEndFrame):
        collector.on_llm_end(now)
    elif isinstance(frame, OutputAudioRawFrame):
        collector.on_output_audio(now)


class MetricsTap(FrameProcessor):
    """Pass-through frame tap that reports latency-boundary events to a shared LatencyCollector.

    Placed at three pipeline positions, each of which sees a distinct set of boundary frames:
      - after STT  : VADUserStoppedSpeakingFrame, final TranscriptionFrame
      - after LLM  : first LLMTextFrame / FunctionCallInProgressFrame, LLMFullResponseEndFrame
      - after TTS  : first OutputAudioRawFrame
    One class handles all three; each instance only encounters its local frames. Never mutates or
    drops a frame — every frame is forwarded unchanged.
    """

    def __init__(self, collector: LatencyCollector, **kwargs) -> None:
        super().__init__(**kwargs)
        self._c = collector

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        record_frame_event(self._c, frame, time.monotonic())
        await self.push_frame(frame, direction)

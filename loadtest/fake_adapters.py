"""Phase 11 — synthetic adapters for the Tier-A load test.

Tier A answers one question: **does the orchestrator scale, or does it?** To answer that, the
vendors have to stop being variables. These adapters replace STT/LLM/TTS/tools/media with
fixed latency distributions, so any growth in measured voice-to-voice latency as concurrency
rises is the loop, the queue, the VAD, and the GIL — not Deepgram having a slow afternoon.

The distributions are the project's own measured numbers (``README.md`` latency table and the
Phase-10 live LLM check), not invented ones, so a load-test result is dimensionally comparable
to a real call:

    ASR final     87 ms p50 / 171 ms p95
    LLM TTFT     600 ms p50 (Phase-10 measured 780 ms on a cold first call)
    TTS TTFB     134 ms p50 / 1078 ms p95

Lognormal rather than gaussian, because provider latency has a tail and a symmetric
distribution would quietly under-report p95 — the number that actually matters here.

Everything is seeded, so a sweep is reproducible and two runs can be compared.
"""

from __future__ import annotations

import asyncio
import math
import random
import time
from dataclasses import dataclass

from clinic_agent.core import events as ev

# --- latency model ------------------------------------------------------------------------


@dataclass(frozen=True)
class LatencyProfile:
    """Provider latencies in milliseconds, as (p50, p95) pairs."""

    asr_ms: tuple[float, float] = (87.0, 171.0)
    llm_ttft_ms: tuple[float, float] = (600.0, 1400.0)
    llm_token_ms: tuple[float, float] = (12.0, 30.0)
    tts_ttfb_ms: tuple[float, float] = (134.0, 1078.0)
    tool_ms: tuple[float, float] = (140.0, 320.0)


def _lognormal(p50: float, p95: float, rng: random.Random) -> float:
    """Draw from a lognormal fitted to the given median and 95th percentile.

    mu = ln(p50); sigma = ln(p95/p50) / 1.645 (the standard normal 95th-percentile z-score).
    """
    if p95 <= p50:
        return p50 / 1000.0
    mu = math.log(p50)
    sigma = math.log(p95 / p50) / 1.6448536269514722
    return math.exp(rng.gauss(mu, sigma)) / 1000.0


class Sampler:
    def __init__(self, profile: LatencyProfile, seed: int) -> None:
        self.profile = profile
        self.rng = random.Random(seed)

    def draw(self, name: str) -> float:
        p50, p95 = getattr(self.profile, name)
        return _lognormal(p50, p95, self.rng)


# --- scripted conversation ------------------------------------------------------------------
# A representative booking: intake turns, an availability lookup, a hold, and a confirm. Tool
# turns matter — they are the ones that put two LLM round trips inside a single caller turn.

SCRIPT: list[list[dict]] = [
    [{"kind": "text", "text": "Happy to help. Can I get your name? "}],
    [{"kind": "text", "text": "Thanks. What day works for you? "}],
    [{"kind": "tool", "name": "check_availability", "args": {"date": "2026-08-14"}}],
    [{"kind": "text", "text": "Thursday at nine is open. Does that work? "}],
    [{"kind": "tool", "name": "hold_slot", "args": {"slot_id": 12}}],
    [{"kind": "text", "text": "Holding that for you. Shall I book it? "}],
    [{"kind": "tool", "name": "confirm_booking", "args": {"hold_id": "h1"}}],
    [{"kind": "text", "text": "You're all set. Anything else? "}],
]

TOOL_RESULTS = {
    "check_availability": {"ok": True, "count": 2, "slots": [{"slot_id": 12}, {"slot_id": 13}]},
    "hold_slot": {"ok": True, "hold_id": "h1", "slot_id": 12},
    "confirm_booking": {"ok": True, "confirmation_id": "LOAD1234"},
}


class FakeLLM:
    """Streams a scripted reply after a realistic time-to-first-token."""

    def __init__(self, emit, sampler: Sampler) -> None:
        self._emit = emit
        self._s = sampler
        self._tasks: dict[str, asyncio.Task] = {}
        self._step = 0

    # Keyword-for-keyword with AnthropicLLM.start. Spelled out rather than swallowed by
    # **kwargs on purpose: when the real adapter grows an argument, this must fail loudly at
    # the seam instead of quietly accepting it. Phase 13 added `context_note` and this fake was
    # not updated, which broke every load-test session for a whole phase (see tier_a's
    # failed-session guard).
    def start(self, request_id: str, messages, *, tier=None, intent=None,
              routing_reason="", context_note="") -> None:
        self._tasks[request_id] = asyncio.create_task(self._run(request_id))

    def cancel(self, request_id: str) -> None:
        task = self._tasks.pop(request_id, None)
        if task and not task.done():
            task.cancel()

    async def _run(self, request_id: str) -> None:
        step = SCRIPT[self._step % len(SCRIPT)]
        self._step += 1
        try:
            await asyncio.sleep(self._s.draw("llm_ttft_ms"))
            for block in step:
                if block["kind"] == "text":
                    # Stream in word-sized deltas so the reducer's sentence chunking runs for
                    # real — that path is part of what is being load tested.
                    for word in block["text"].split(" "):
                        self._emit(
                            ev.LLMTextDelta(
                                t=time.monotonic(), request_id=request_id, text=word + " "
                            )
                        )
                        await asyncio.sleep(self._s.draw("llm_token_ms"))
                else:
                    self._emit(
                        ev.LLMToolUse(
                            t=time.monotonic(),
                            request_id=request_id,
                            tool_call_id=f"{request_id}-{block['name']}",
                            name=block["name"],
                            arguments=dict(block["args"]),
                        )
                    )
            stop = "tool_use" if any(b["kind"] == "tool" for b in step) else "end_turn"
            self._emit(
                ev.LLMCompleted(t=time.monotonic(), request_id=request_id, stop_reason=stop)
            )
        except asyncio.CancelledError:
            raise
        finally:
            self._tasks.pop(request_id, None)

    async def aclose(self) -> None:
        for request_id in list(self._tasks):
            self.cancel(request_id)


class FakeTools:
    """Canned scheduling results after a realistic API round trip."""

    def __init__(self, emit, sampler: Sampler) -> None:
        self._emit = emit
        self._s = sampler
        self._tasks: dict[str, asyncio.Task] = {}

    def invoke(self, tool_call_id: str, name: str, arguments: dict) -> None:
        self._tasks[tool_call_id] = asyncio.create_task(self._run(tool_call_id, name))

    def load_caller_memory(self, phone: str) -> None:
        """Phase 13's pre-greeting ANI lookup, faked as an unknown caller.

        The real one is fire-and-forget and swallows its own failures, so a missing method here
        would not crash a session -- it would just silently skip a real per-call code path and
        make the load test measure a slightly different program than the one that ships.
        Every returning-caller session is a cold one under load until this fake grows a
        recognised-caller variant.
        """
        self._emit(ev.CallerMemoryLoaded(t=time.monotonic(), known=False,
                                         upcoming_appointments=0))

    async def post_call_metrics(self, payload: dict) -> dict:
        """Phase 16's teardown POST, faked as an instant success.

        It costs a real HTTP round trip per session at teardown, so the load test has to have
        SOMETHING here or it measures a cheaper program than the one that ships. Zero latency is
        the deliberate simplification: teardown is off the critical path and already bounded by
        `CallSession.METRICS_POST_TIMEOUT_S`.
        """
        return {"ok": True, "call_id": payload.get("call_id"), "turns": 0, "tools": 0}

    async def _run(self, tool_call_id: str, name: str) -> None:
        latency = self._s.draw("tool_ms")
        try:
            await asyncio.sleep(latency)
            result = TOOL_RESULTS.get(name, {"ok": True})
            self._emit(
                ev.ToolCompleted(
                    t=time.monotonic(),
                    tool_call_id=tool_call_id,
                    name=name,
                    result=result,
                    ok=True,
                    latency_ms=latency * 1000,
                    http_status=200,
                )
            )
        except asyncio.CancelledError:
            raise
        finally:
            self._tasks.pop(tool_call_id, None)

    async def aclose(self) -> None:
        for task in list(self._tasks.values()):
            task.cancel()
        self._tasks.clear()


class FakeMedia:
    """Playback with no device: emits the same speaking events the real adapter does.

    Playback duration is derived from the text length at a realistic speaking rate, because
    turn cadence — and therefore how many sessions are concurrently *busy* — depends on it. A
    media fake that completes instantly would report a fleet that is idle most of the time and
    flatter the result.
    """

    SPEAKING_RATE_CPS = 15.0  # characters per second, ≈ 165 wpm

    def __init__(self, emit) -> None:
        self._emit = emit
        self._speaking: str | None = None
        self._pending: dict[str, float] = {}
        self._tasks: set[asyncio.Task] = set()

    async def start(self) -> None:
        pass

    async def play(self, utterance_id: str, pcm: bytes) -> None:
        self._pending[utterance_id] = self._pending.get(utterance_id, 0.0) + len(pcm)
        if self._speaking != utterance_id:
            self._speaking = utterance_id
            self._emit(ev.BotStartedSpeaking(t=time.monotonic(), utterance_id=utterance_id))

    async def end_utterance(self, utterance_id: str) -> None:
        duration = self._pending.pop(utterance_id, 0.0) / self.SPEAKING_RATE_CPS
        task = asyncio.create_task(self._finish(utterance_id, duration))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _finish(self, utterance_id: str, duration: float) -> None:
        try:
            await asyncio.sleep(duration)
        except asyncio.CancelledError:
            return
        if self._speaking == utterance_id:
            self._speaking = None
            self._emit(
                ev.BotStoppedSpeaking(
                    t=time.monotonic(), utterance_id=utterance_id, completed=True
                )
            )

    async def clear(self, utterance_id: str) -> None:
        self._pending.pop(utterance_id, None)
        if self._speaking == utterance_id:
            self._speaking = None
            self._emit(
                ev.BotStoppedSpeaking(
                    t=time.monotonic(), utterance_id=utterance_id, completed=False
                )
            )

    async def aclose(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        self._tasks.clear()


class FakeTTS:
    """Synthesis with a realistic time-to-first-byte, feeding the fake media adapter."""

    def __init__(self, media: FakeMedia, sampler: Sampler) -> None:
        self._media = media
        self._s = sampler
        self._first: set[str] = set()

    async def start(self) -> None:
        pass

    async def speak(self, utterance_id: str, text: str, *, final: bool) -> None:
        if utterance_id not in self._first:
            self._first.add(utterance_id)
            await asyncio.sleep(self._s.draw("tts_ttfb_ms"))
        if text:
            await self._media.play(utterance_id, b"\x00" * max(1, len(text)))
        if final:
            self._first.discard(utterance_id)
            await self._media.end_utterance(utterance_id)

    async def cancel(self, utterance_id: str) -> None:
        self._first.discard(utterance_id)
        await self._media.clear(utterance_id)

    async def aclose(self) -> None:
        pass


class FakeSTT:
    """Consumes audio and finalizes a scripted transcript after a realistic ASR delay."""

    def __init__(self, emit, sampler: Sampler) -> None:
        self._emit = emit
        self._s = sampler
        self._tasks: set[asyncio.Task] = set()
        self.audio_frames = 0

    async def start(self) -> None:
        pass

    async def send_audio(self, pcm: bytes) -> None:
        self.audio_frames += 1

    def transcribe(self, text: str) -> None:
        """Called by the driver at the caller's end of turn."""
        task = asyncio.create_task(self._finalize(text))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _finalize(self, text: str) -> None:
        try:
            await asyncio.sleep(self._s.draw("asr_ms"))
        except asyncio.CancelledError:
            return
        self._emit(ev.FinalTranscript(t=time.monotonic(), text=text, confidence=0.96))

    async def aclose(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        self._tasks.clear()


class FakeClassifier:
    """No-op classifier for load runs: intent scoping is not what Tier A measures."""

    def classify(self, utterance: str) -> None:
        pass

    async def aclose(self) -> None:
        pass

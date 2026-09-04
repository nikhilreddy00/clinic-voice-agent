"""Phase 11 — Tier-A load test: orchestrator scale.

Drives N concurrent :class:`~clinic_agent.core.session.CallSession` instances through a real
booking script inside a real :class:`~clinic_agent.core.worker.Worker`, with the vendors
replaced by seeded latency distributions (``fake_adapters``). Everything under test is the
production code path: the event queue, ``reduce()``, action dispatch, the turn engine, metrics,
and trace recording.

Because provider latency is held fixed, **any growth in voice-to-voice latency as concurrency
rises is the orchestrator**. That is the whole design of the experiment.

Two modes, measuring genuinely different ceilings:

* ``--mode loop`` (default) — no audio. Bounds the event loop, the reducer, and the queue.
* ``--mode audio`` — synthetic PCM fed through the real ``TurnEngine`` at real-time pace, so
  ``frame_rms`` and Silero inference run for real, 50 frames/second/session. This is where the
  practical per-worker ceiling lives, and it is much lower than the loop-only number. Reporting
  only the first would be the flattering half of the truth.

The headline signal is **event-loop lag**: a background task that asks to sleep 50 ms and
records how late it actually wakes. It rises before latency percentiles do and is the clearest
statement of "this worker is past its limit."

    uv run python loadtest/tier_a.py --mode loop  --concurrency 1,10,50,100,250,500,1000
    uv run python loadtest/tier_a.py --mode audio --concurrency 1,5,10,20,40
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import resource
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "agent" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from loguru import logger  # noqa: E402

from clinic_agent.config import Settings  # noqa: E402
from clinic_agent.core import events as ev  # noqa: E402
from clinic_agent.core.audio import INPUT_SAMPLE_RATE  # noqa: E402
from clinic_agent.core.session import CallSession  # noqa: E402
from clinic_agent.core.state import Phase  # noqa: E402
from clinic_agent.core.telemetry import LoopLagMonitor  # noqa: E402
from clinic_agent.core.worker import Worker  # noqa: E402
from fake_adapters import (  # noqa: E402
    FakeClassifier,
    FakeLLM,
    FakeMedia,
    FakeSTT,
    FakeTools,
    FakeTTS,
    LatencyProfile,
    Sampler,
)
import chart  # noqa: E402

CALLER_TURNS = [
    "Hi, I'd like to book an appointment.",
    "Dana Reyes.",
    "Sometime tomorrow morning would be great.",
    "Yes, that one works for me.",
    "Yes, please go ahead and book it.",
    "No, that's everything. Thank you.",
]

# One 20 ms frame of low-level noise. Content is irrelevant to the measurement; what matters is
# that the real RMS and the real Silero inference run on every frame.
_FRAME = bytes(int(INPUT_SAMPLE_RATE * 0.02) * 2)


def load_settings_for_test() -> Settings:
    return Settings(
        mode="telephony",
        deepgram_api_key="loadtest",
        anthropic_api_key="loadtest",
        anthropic_model="claude-haiku-4-5-20251001",
        groq_api_key="",
        groq_model="",
        cartesia_api_key="loadtest",
        cartesia_voice_id="loadtest",
        livekit_url="ws://loadtest",
        livekit_api_key="loadtest",
        livekit_api_secret="loadtest-secret-value-long-enough",
        livekit_phone_number="",
        scheduling_api_base_url="http://127.0.0.1:8000",
    )


class LoadSession(CallSession):
    """A production CallSession with seeded synthetic vendors."""

    def __init__(self, settings: Settings, call_id: str, sampler: Sampler, *, record: bool):
        self._sampler = sampler
        super().__init__(settings, call_id=call_id, record=record)

    def _build_llm(self):
        return FakeLLM(self.emit, self._sampler)

    def _build_tools(self):
        return FakeTools(self.emit, self._sampler)

    def _build_media(self):
        return FakeMedia(self.emit)

    def _build_tts(self):
        return FakeTTS(self.media, self._sampler)

    def _build_stt(self):
        return FakeSTT(self.emit, self._sampler)

    def _build_classifier(self):
        return FakeClassifier()


# --- instrumentation ------------------------------------------------------------------------


# LoopLagMonitor now lives in the engine (core/telemetry.py) so a live call and this
# harness report loop lag the same way. Imported above.


def percentile(values: list[float], p: float) -> float | None:
    """Nearest-rank percentile — matches ``clinic_agent.metrics.percentiles``."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(0, math.ceil(p / 100 * len(ordered)) - 1)
    return round(ordered[rank], 1)


# A p95 computed from a handful of samples is not a p95. At concurrency 1 with 6 turns there
# are six e2e samples, so "p95" is the sixth-largest of six — it misses the tool-call turns
# entirely and reads LOW, which then makes every higher level look like a regression. Low
# concurrency levels are therefore repeated until they have a comparable number of samples.
MIN_SAMPLES_FOR_PERCENTILES = 60
MAX_REPEATS = 12


@dataclass
class LevelResult:
    concurrency: int
    mode: str
    repeats: int = 1
    undersampled: bool = False
    sessions_started: int = 0
    sessions_completed: int = 0
    sessions_failed: int = 0
    turns_completed: int = 0
    turns_timed_out: int = 0
    booked: int = 0
    wall_secs: float = 0.0
    cpu_secs: float = 0.0
    rss_mb: float = 0.0
    e2e: dict = field(default_factory=dict)
    loop_lag_ms: dict = field(default_factory=dict)
    audio_frames: int = 0

    @property
    def cpu_per_session(self) -> float:
        return self.cpu_secs / self.sessions_started if self.sessions_started else 0.0


# --- the driver -----------------------------------------------------------------------------


async def drive_call(session: LoadSession, turns: int, *, audio: bool, result: LevelResult) -> None:
    """Run one scripted caller through ``turns`` turns of conversation."""
    if not await wait_for_listening(session, timeout=30.0):
        return

    for i in range(turns):
        if session.state.phase is Phase.CLOSED:
            break
        line = CALLER_TURNS[i % len(CALLER_TURNS)]

        session.emit(ev.SpeechStarted(t=time.monotonic()))
        if audio:
            # ~1.2 s of speech at real-time pace: 60 frames through the real TurnEngine, so
            # RMS and Silero inference are genuinely on the critical path.
            for _ in range(60):
                await session._turn.feed(_FRAME, time.monotonic())
                await asyncio.sleep(0.02)
        session.emit(ev.SpeechStopped(t=time.monotonic()))
        session._stt.transcribe(line)

        # Wait for the turn to START before waiting for it to end. Without this the driver
        # sees the session still in LISTENING — the transcript is still inside the synthetic
        # ASR delay — and immediately calls the turn complete, so every turn is recorded as
        # instantaneous and no e2e sample is ever produced.
        if not await wait_until(session, lambda s: s is not Phase.LISTENING, timeout=30.0):
            result.turns_timed_out += 1
            break
        if not await wait_for_listening(session, timeout=30.0):
            result.turns_timed_out += 1
            break
        result.turns_completed += 1

    session.hangup("loadtest_complete")


async def wait_until(session: CallSession, predicate, timeout: float) -> bool:
    """Poll until ``predicate(phase)`` holds, the call closes, or the timeout expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        phase = session.state.phase
        if phase is Phase.CLOSED:
            return False
        if predicate(phase):
            return True
        await asyncio.sleep(0.005)
    return False


async def wait_for_listening(session: CallSession, timeout: float) -> bool:
    """Wait until the bot has handed the floor back (or the call ended)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if session.state.phase is Phase.CLOSED:
            return False
        if session.state.phase is Phase.LISTENING and session._queue.empty():
            return True
        await asyncio.sleep(0.005)
    return False


async def run_level(
    concurrency: int, *, turns: int, mode: str, seed: int, record: bool, min_samples: int
) -> LevelResult:
    """Run one concurrency level, repeating it until the percentiles are trustworthy."""
    repeats = min(
        MAX_REPEATS, max(1, -(-min_samples // max(1, concurrency * turns)))  # ceil division
    )
    result = LevelResult(concurrency=concurrency, mode=mode, repeats=repeats)
    e2e: list[float] = []
    lag_samples: list[float] = []

    rusage_before = resource.getrusage(resource.RUSAGE_SELF)
    t0 = time.monotonic()
    for round_index in range(repeats):
        await run_round(
            concurrency,
            turns=turns,
            mode=mode,
            seed=seed + round_index * 101,
            record=record,
            result=result,
            e2e=e2e,
            lag_samples=lag_samples,
            round_index=round_index,
        )

    result.wall_secs = round(time.monotonic() - t0, 2)
    rusage_after = resource.getrusage(resource.RUSAGE_SELF)
    result.cpu_secs = round(
        (rusage_after.ru_utime - rusage_before.ru_utime)
        + (rusage_after.ru_stime - rusage_before.ru_stime),
        2,
    )
    # ru_maxrss is a process-wide high-water mark, so this is peak RSS observed *so far*, not
    # this level's footprint. Flat peak across rising concurrency is the meaningful reading.
    result.rss_mb = round(rusage_after.ru_maxrss / (1024 * 1024), 1)

    result.e2e = {
        "p50": percentile(e2e, 50),
        "p95": percentile(e2e, 95),
        "p99": percentile(e2e, 99),
        "count": len(e2e),
    }
    result.loop_lag_ms = {
        "p50": percentile(lag_samples, 50),
        "p95": percentile(lag_samples, 95),
        "max": round(max(lag_samples), 1) if lag_samples else None,
        "count": len(lag_samples),
    }
    result.undersampled = len(e2e) < min_samples
    return result


async def run_round(
    concurrency: int,
    *,
    turns: int,
    mode: str,
    seed: int,
    record: bool,
    result: LevelResult,
    e2e: list[float],
    lag_samples: list[float],
    round_index: int,
) -> None:
    """One pass of ``concurrency`` simultaneous calls."""
    audio = mode == "audio"
    settings = load_settings_for_test()

    worker = Worker(
        settings,
        capacity=concurrency,
        prewarm=min(concurrency, 8),
        session_factory=lambda s, call_id: LoadSession(
            s, call_id, Sampler(LatencyProfile(), seed + hash(call_id) % 10_000), record=record
        ),
    )
    await worker.start()

    lag = LoopLagMonitor()
    lag.start()

    drivers: list[asyncio.Task] = []
    sessions: list[CallSession] = []
    for i in range(concurrency):
        call_id = f"load-{mode}-{concurrency}-r{round_index}-{i}"
        if not await worker.accept(call_id):
            continue
        session = worker.session(call_id)
        sessions.append(session)
        result.sessions_started += 1
        session.emit(ev.CallerPresent(t=time.monotonic(), participant_id=f"caller-{i}"))
        drivers.append(
            asyncio.create_task(drive_call(session, turns, audio=audio, result=result))
        )

    await asyncio.gather(*drivers, return_exceptions=True)

    for session in sessions:
        e2e.extend(session.metrics.samples("e2e"))
        if session.state.booked:
            result.booked += 1
        if audio:
            result.audio_frames += getattr(session._stt, "audio_frames", 0)

    await lag.stop()
    lag_samples.extend(lag.samples)

    await worker.drain(timeout=15.0)
    result.sessions_completed += worker.stats.completed
    result.sessions_failed += worker.stats.failed


def find_knee(
    results: list[LevelResult], *, degradation: float = 1.5, lag_budget_ms: float = 50.0
) -> int | None:
    """First concurrency level where the orchestrator itself starts costing latency.

    Measured as degradation **relative to the single-session baseline**, not against an
    absolute product budget. The synthetic providers are not the real ones — their fixed
    delays already put the baseline above the Phase-14 target — so an absolute threshold would
    report a knee at one session and say nothing about scaling. What Tier A can honestly
    measure is the *added* cost of concurrency, and that is what this finds.

    Loop lag is the second criterion, and often trips first: it is pure scheduling delay, so it
    rises as soon as the loop is oversubscribed, while a latency percentile still has the
    provider delays averaged into it.
    """
    if not results:
        return None
    baseline_p95 = results[0].e2e.get("p95")
    for r in results[1:]:
        p95 = r.e2e.get("p95")
        lag_p95 = r.loop_lag_ms.get("p95") or 0.0
        if baseline_p95 and p95 is not None and p95 > baseline_p95 * degradation:
            return r.concurrency
        if lag_p95 > lag_budget_ms:
            return r.concurrency
    return None


def find_lag_inflection(results: list[LevelResult], *, factor: float = 2.5) -> int | None:
    """First level where loop lag grows faster than concurrency does.

    A softer signal than :func:`find_knee`, and on this hardware the only one that fires: voice
    latency never degraded, but scheduling headroom did. Lag growing superlinearly across a
    doubling of concurrency is the worker saying it is approaching its limit while callers
    still cannot hear any difference — which is exactly when you want to know.
    """
    previous = None
    for r in results:
        lag = r.loop_lag_ms.get("p95")
        if lag is None:
            continue
        if previous is not None:
            prev_conc, prev_lag = previous
            conc_growth = r.concurrency / prev_conc if prev_conc else 1.0
            lag_growth = lag / prev_lag if prev_lag else 1.0
            if lag_growth > conc_growth * factor / 2 and lag > 2.0:
                return r.concurrency
        previous = (r.concurrency, lag)
    return None


def rechart(json_path: Path) -> int:
    """Redraw a chart from a saved sweep. A sweep is minutes of compute; presentation is not."""
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    results = [
        LevelResult(**{k: v for k, v in level.items() if k in LevelResult.__dataclass_fields__})
        for level in payload["levels"]
    ]
    knee = find_knee(results)
    lag_inflection = find_lag_inflection(results)
    payload["knee_concurrency"] = knee
    payload["lag_inflection_concurrency"] = lag_inflection
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    marker = knee or lag_inflection
    svg = chart.render(
        json_path.with_suffix(".svg"),
        title=f"Tier-A: concurrency vs voice-to-voice latency ({payload['mode']} mode)",
        subtitle=(
            f"synthetic providers at fixed latency, so growth is orchestrator overhead · "
            f"{payload['turns_per_call']} turns/call · {payload.get('cpu_count')} cores · "
            f"seed {payload['seed']}"
        ),
        x_values=[float(r.concurrency) for r in results],
        series={
            "e2e p95 (ms)": [r.e2e.get("p95") for r in results],
            "e2e p50 (ms)": [r.e2e.get("p50") for r in results],
            "event-loop lag p95 (ms)": [r.loop_lag_ms.get("p95") for r in results],
        },
        y_label="milliseconds",
        knee=float(marker) if marker else None,
        knee_label=(
            f"knee ≈ {knee}" if knee else (f"lag inflection ≈ {lag_inflection}" if lag_inflection else "")
        ),
    )
    print(f"knee: {knee or 'not reached'} · lag inflection: {lag_inflection or 'none'}")
    print(f"chart: {svg}")
    return 0


async def main() -> int:
    parser = argparse.ArgumentParser(description="Tier-A orchestrator load test")
    parser.add_argument("--concurrency", default="1,10,50,100,250,500,1000")
    parser.add_argument("--turns", type=int, default=6)
    parser.add_argument("--mode", choices=("loop", "audio"), default="loop")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--min-samples", type=int, default=MIN_SAMPLES_FOR_PERCENTILES,
        help="repeat low-concurrency levels until they have at least this many e2e samples",
    )
    parser.add_argument("--record-traces", action="store_true",
                        help="write per-call traces (off by default: it is I/O under test)")
    parser.add_argument("--out", default=str(Path(__file__).parent / "results"))
    parser.add_argument(
        "--rechart", default=None,
        help="re-render the chart from an existing results JSON instead of running a sweep",
    )
    args = parser.parse_args()

    if args.rechart:
        return rechart(Path(args.rechart))

    logger.remove()
    logger.add(sys.stderr, level=os.getenv("CLINIC_LOG_LEVEL", "WARNING"), enqueue=True)

    levels = [int(x) for x in args.concurrency.split(",") if x.strip()]
    results: list[LevelResult] = []

    print(f"Tier-A load test — mode={args.mode} turns={args.turns} seed={args.seed}")
    print(f"{'conc':>6} {'rnds':>5} {'sess':>6} {'e2e n':>7} {'e2e p50':>9} {'e2e p95':>9} "
          f"{'lag p95':>9} {'cpu/sess':>9} {'peak MB':>8} {'wall s':>7}")
    print("-" * 92)

    for level in levels:
        result = await run_level(
            level, turns=args.turns, mode=args.mode, seed=args.seed,
            record=args.record_traces, min_samples=args.min_samples,
        )
        results.append(result)
        flag = " *undersampled" if result.undersampled else ""
        if result.sessions_failed:
            flag += f"  *** {result.sessions_failed} SESSIONS FAILED ***"
        print(
            f"{result.concurrency:>6} {result.repeats:>5} {result.sessions_started:>6} "
            f"{result.e2e['count']:>7} {str(result.e2e['p50']):>9} {str(result.e2e['p95']):>9} "
            f"{str(result.loop_lag_ms['p95']):>9} {result.cpu_per_session:>9.3f} "
            f"{result.rss_mb:>8.1f} {result.wall_secs:>7.1f}{flag}"
        )

    knee = find_knee(results)
    lag_inflection = find_lag_inflection(results)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "turns_per_call": args.turns,
        "seed": args.seed,
        "cpu_count": os.cpu_count(),
        "min_samples": args.min_samples,
        "knee_concurrency": knee,
        "lag_inflection_concurrency": lag_inflection,
        "levels": [asdict(r) for r in results],
    }
    json_path = out_dir / f"tierA-{args.mode}-{stamp}.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    x = [float(r.concurrency) for r in results]
    svg_path = chart.render(
        out_dir / f"tierA-{args.mode}-{stamp}.svg",
        title=f"Tier-A: concurrency vs voice-to-voice latency ({args.mode} mode)",
        subtitle=(
            f"synthetic providers at fixed latency, so growth is orchestrator overhead · "
            f"{args.turns} turns/call · {os.cpu_count()} cores · seed {args.seed}"
        ),
        x_values=x,
        series={
            "e2e p95 (ms)": [r.e2e.get("p95") for r in results],
            "e2e p50 (ms)": [r.e2e.get("p50") for r in results],
            "event-loop lag p95 (ms)": [r.loop_lag_ms.get("p95") for r in results],
        },
        y_label="milliseconds",
        knee=float(knee or lag_inflection) if (knee or lag_inflection) else None,
        knee_label=(
            f"knee ≈ {knee}" if knee else (f"lag inflection ≈ {lag_inflection}" if lag_inflection else "")
        ),
    )

    print(f"\nknee (latency degradation): {knee if knee else 'not reached in this sweep'}")
    print(f"loop-lag inflection:        {lag_inflection if lag_inflection else 'none'}")
    print(f"results: {json_path}")
    print(f"chart:   {svg_path}")

    # A load test that measured nothing must not exit 0. `FakeLLM.start()` drifted out of sync
    # with the real adapter when Phase 13 added `context_note`, so every session raised on its
    # first turn -- and this harness printed a tidy table of `None`s, wrote a chart, and exited
    # successfully for a whole phase. Undersampling is a warning; sessions dying is a failure.
    failed = sum(r.sessions_failed for r in results)
    if failed:
        print(f"\nFAILED: {failed} session(s) raised — these numbers measure nothing. "
              f"Most likely a fake adapter in loadtest/fake_adapters.py has drifted out of "
              f"sync with the real one in agent/src/clinic_agent/core/adapters/.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

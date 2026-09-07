"""Phase 11 — a worker that hosts N concurrent CallSessions in one process.

Phase 10 made this possible by moving every piece of per-call state into ``CallSession``; this
module is what actually uses it. Through Phase 10 the deployment model was one process per call
(`railway.toml: numReplicas = 1`, Direct SIP dispatch into one shared room), and `CLAUDE.md`
justified it with the GIL: CPU-bound audio work on one call would stall the others.

That reasoning was right about the mechanism and wrong about the conclusion. The fix is not one
process per call — it is making per-session work non-blocking, which is what the Phase-11 audit
did:

* Silero weights and its inference pool are shared process-wide (``core/vad.py``), so a session
  costs ~2 KB instead of 7.97 MB and inference runs on a bounded pool sized to cores.
* Trace and metrics writes go through buffered/shared sinks (``core/jsonl.py``) instead of an
  ``open/write/close`` per event on the event loop.
* ``frame_rms`` is vectorized (Phase 10), so the 50 Hz-per-session RMS is not a Python loop.

What remains genuinely serial is Silero inference and the reducer itself. The load test
(``loadtest/``) is what says where that ceiling actually is, rather than guessing.

**Prewarming.** Constructing a session is not free — an analyzer, plus STT and TTS websocket
handshakes. On the Direct-dispatch model that cost was paid once at boot and never appeared in
call latency. With room-per-call it lands *inside* the caller's first impression, so the worker
keeps a small pool of sessions with their sockets already open and hands one over on assignment.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from loguru import logger

from .. import tenant
from ..config import Settings
from .jsonl import close_shared, flush_shared
from .session import CallSession
from .vad import SharedSileroVAD, shared_inference_session

SessionFactory = Callable[[Settings, str], CallSession]


@dataclass
class WorkerStats:
    """Counters a session router polls to make assignment decisions."""

    capacity: int = 0
    active: int = 0
    prewarmed: int = 0
    accepted: int = 0
    rejected: int = 0
    completed: int = 0
    failed: int = 0
    prewarm_hits: int = 0
    prewarm_misses: int = 0
    outcomes: dict[str, int] = field(default_factory=dict)

    @property
    def free(self) -> int:
        return max(0, self.capacity - self.active)

    @property
    def load(self) -> float:
        return self.active / self.capacity if self.capacity else 1.0

    def as_dict(self) -> dict:
        return {
            "capacity": self.capacity,
            "active": self.active,
            "free": self.free,
            "load": round(self.load, 3),
            "prewarmed": self.prewarmed,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "completed": self.completed,
            "failed": self.failed,
            "prewarm_hits": self.prewarm_hits,
            "prewarm_misses": self.prewarm_misses,
            "outcomes": dict(self.outcomes),
        }


class Worker:
    """Hosts up to ``capacity`` concurrent sessions on one event loop."""

    def __init__(
        self,
        settings: Settings,
        *,
        capacity: int = 8,
        prewarm: int = 2,
        session_factory: SessionFactory | None = None,
    ) -> None:
        self.settings = settings
        self.capacity = capacity
        self.prewarm_target = min(prewarm, capacity)
        self._factory = session_factory or (lambda s, call_id: CallSession(s, call_id=call_id))

        self.stats = WorkerStats(capacity=capacity)
        self._sessions: dict[str, CallSession] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._pool: list[SharedSileroVAD] = []
        self._draining = False
        self._idle = asyncio.Event()
        self._idle.set()

    # --- lifecycle ----------------------------------------------------------------------

    async def start(self) -> None:
        """Load shared resources and fill the prewarm pool before accepting calls."""
        t0 = time.monotonic()
        shared_inference_session()  # pay the model load once, at boot, not on a call
        # The tenant this worker answers for. Resolved here for the same reason the VAD weights
        # are loaded here: anything a call would otherwise wait for belongs at boot.
        await tenant.load(
            self.settings.scheduling_api_base_url, self.settings.livekit_phone_number
        )
        await self._refill_pool()
        logger.info(
            f"[worker] ready: capacity={self.capacity} prewarm={len(self._pool)} "
            f"({(time.monotonic() - t0) * 1000:.0f} ms)"
        )

    async def _refill_pool(self) -> None:
        while len(self._pool) < self.prewarm_target:
            analyzer = SharedSileroVAD(sample_rate=16000)
            # Resolve the rate now, not on assignment: set_sample_rate is what computes the
            # frame counts and hysteresis thresholds, and doing it here is the difference
            # between a warm analyzer and one that still has setup left to do on the call.
            analyzer.set_sample_rate(16000)
            self._pool.append(analyzer)
        self.stats.prewarmed = len(self._pool)

    def _take_prewarmed(self) -> SharedSileroVAD | None:
        """Hand out a ready analyzer, resetting its recurrent state for the new caller.

        The reset is not optional: Silero's state is a rolling window of the *previous*
        caller's audio, so a reused analyzer would start the next call mid-utterance.
        """
        if not self._pool:
            self.stats.prewarm_misses += 1
            return None
        analyzer = self._pool.pop()
        analyzer.reset()
        self.stats.prewarm_hits += 1
        self.stats.prewarmed = len(self._pool)
        return analyzer

    # --- assignment ---------------------------------------------------------------------

    def can_accept(self) -> bool:
        return not self._draining and len(self._sessions) < self.capacity

    async def accept(self, call_id: str, *, on_done: Callable[[str], None] | None = None) -> bool:
        """Start a session for ``call_id``. Returns False if the worker is full or draining.

        Rejection is a normal outcome, not an error: it is the signal the router uses to place
        the call on a different worker. A worker that accepted past capacity would degrade
        every call it is already carrying rather than shedding one.
        """
        if not self.can_accept():
            self.stats.rejected += 1
            return False
        if call_id in self._sessions:
            return False

        session = self._factory(self.settings, call_id)
        analyzer = self._take_prewarmed()
        if analyzer is not None:
            self._attach_prewarmed(session, analyzer)

        self._sessions[call_id] = session
        self.stats.accepted += 1
        self.stats.active = len(self._sessions)
        self._idle.clear()
        self._tasks[call_id] = asyncio.create_task(
            self._run_session(call_id, session, on_done), name=f"call-{call_id}"
        )
        await self._refill_pool()
        return True

    def _attach_prewarmed(self, session: CallSession, analyzer: SharedSileroVAD) -> None:
        """Give the session's TurnEngine an already-constructed analyzer."""
        turn = getattr(session, "_turn", None)
        if turn is not None:
            turn._vad = analyzer
            analyzer.set_sample_rate(turn._sample_rate)

    async def _run_session(
        self, call_id: str, session: CallSession, on_done: Callable[[str], None] | None
    ) -> None:
        try:
            await session.run()
            self.stats.completed += 1
            outcome = session.state.outcome
            self.stats.outcomes[outcome] = self.stats.outcomes.get(outcome, 0) + 1
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - one bad call must not take the worker down
            self.stats.failed += 1
            logger.error(f"[worker] session {call_id} failed: {exc}")
        finally:
            self._sessions.pop(call_id, None)
            self._tasks.pop(call_id, None)
            self.stats.active = len(self._sessions)
            if not self._sessions:
                self._idle.set()
            if on_done is not None:
                on_done(call_id)

    def session(self, call_id: str) -> CallSession | None:
        return self._sessions.get(call_id)

    # --- shutdown -----------------------------------------------------------------------

    async def drain(self, timeout: float = 30.0) -> None:
        """Stop accepting, let in-flight calls finish, then release shared resources.

        Draining rather than killing is the point: a worker being replaced should shed new
        calls to its peers while the ones it already has run to their natural end. Callers
        mid-booking do not get hung up on for a deploy.
        """
        self._draining = True
        logger.info(f"[worker] draining — {len(self._sessions)} call(s) in flight")
        try:
            await asyncio.wait_for(self._idle.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(f"[worker] drain timed out; hanging up {len(self._sessions)} call(s)")
            for session in list(self._sessions.values()):
                session.hangup("worker_drain")
            await asyncio.sleep(0)
        await self.shutdown()

    async def shutdown(self) -> None:
        """Hard stop: cancel everything still running and flush the shared sinks."""
        for task in list(self._tasks.values()):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)
        self._tasks.clear()
        self._sessions.clear()
        self.stats.active = 0
        flush_shared()
        close_shared()
        logger.info(f"[worker] stopped: {self.stats.as_dict()}")


async def run_worker(
    settings: Settings, *, capacity: int = 8, prewarm: int = 2, ready: Awaitable | None = None
) -> Worker:
    """Convenience constructor used by the router host and the load test."""
    worker = Worker(settings, capacity=capacity, prewarm=prewarm)
    await worker.start()
    if ready is not None:
        await ready
    return worker

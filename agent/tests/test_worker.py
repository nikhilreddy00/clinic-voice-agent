"""Phase 11 — the multi-session worker.

The claim under test is the one `CLAUDE.md` used to say was impossible: several live calls in
one process, sharing nothing. Phase 10 made every piece of per-call state live in
``CallSession``; these tests check that the worker built on top of it actually holds the line
on capacity, isolation, prewarming, and draining.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from clinic_agent.config import Settings
from clinic_agent.core import events as ev
from clinic_agent.core.session import CallSession
from clinic_agent.core.state import Phase
from clinic_agent.core.vad import SharedSileroVAD, shared_executor, shared_inference_session
from clinic_agent.core.worker import Worker

from test_session import FakeLLM, FakeMedia, FakeSTT, FakeTools, FakeTTS, _settle, settings


class QuietSession(CallSession):
    """A CallSession with faked vendors, for exercising the worker rather than the adapters."""

    def _build_llm(self):
        return FakeLLM(self.emit, [])

    def _build_tools(self):
        return FakeTools(self.emit, {})

    def _build_media(self):
        return FakeMedia(self.emit)

    def _build_tts(self):
        return FakeTTS(self.media)

    def _build_stt(self):
        return FakeSTT()


def make_worker(tmp_path, monkeypatch, **kwargs) -> Worker:
    monkeypatch.setenv("CLINIC_LOG_DIR", str(tmp_path))
    return Worker(
        settings(),
        session_factory=lambda s, call_id: QuietSession(s, call_id=call_id),
        **kwargs,
    )


# --- shared resources ---------------------------------------------------------------------


def test_silero_weights_and_pool_are_shared_process_wide():
    """The per-session cost that made one-call-per-process look mandatory.

    Measured: Pipecat's analyzer is 7.97 MB and 23.4 ms per instance, so 1,000 sessions would
    duplicate ~8 GB of identical weights. The ONNX session is stateless, so it can be shared.
    """
    a, b = SharedSileroVAD(sample_rate=16000), SharedSileroVAD(sample_rate=16000)
    assert a._model._session is b._model._session
    assert a._executor is b._executor is shared_executor()
    assert shared_inference_session() is a._model._session


def test_shared_vad_keeps_recurrent_state_per_session():
    """Sharing the weights must not share the conversation — that would be a real audio bug."""
    a, b = SharedSileroVAD(sample_rate=16000), SharedSileroVAD(sample_rate=16000)
    a.set_sample_rate(16000)
    b.set_sample_rate(16000)
    assert a._model._state is not b._model._state


def test_reset_clears_state_so_a_pooled_analyzer_is_safe_to_reuse():
    """Without the reset, caller #2 starts with a window of caller #1's audio.

    ``_last_reset_time`` is primed first because the inherited analyzer also resets on a 5 s
    timer, and at t=0 that timer is already due — so a naive version of this test would watch
    Pipecat's periodic reset and prove nothing about ours.
    """
    vad = SharedSileroVAD(sample_rate=16000)
    vad.set_sample_rate(16000)
    vad._last_reset_time = time.time()

    tone = b"\x00\x30" * 512  # 512 int16 samples, the frame size Silero requires at 16 kHz
    vad.voice_confidence(tone)
    assert vad._model._context.shape[1] == 64  # a window of the previous caller's audio

    vad.reset()
    assert vad._model._context.shape[1] == 0
    assert vad._vad_state.name == "QUIET"


def test_reset_before_a_sample_rate_is_known_does_not_crash():
    """A prewarmed analyzer is reset on assignment, possibly before its rate is resolved."""
    SharedSileroVAD().reset()  # must not raise ZeroDivisionError


# --- capacity ------------------------------------------------------------------------------


async def test_worker_hosts_many_sessions_at_once(tmp_path, monkeypatch):
    worker = make_worker(tmp_path, monkeypatch, capacity=5, prewarm=2)
    await worker.start()

    assert all([await worker.accept(f"call-{i}") for i in range(5)])
    assert worker.stats.active == 5
    assert worker.stats.accepted == 5

    await worker.shutdown()


async def test_worker_refuses_past_capacity_instead_of_degrading_everyone(tmp_path, monkeypatch):
    """Rejection is the router's signal to place the call elsewhere, not an error."""
    worker = make_worker(tmp_path, monkeypatch, capacity=2, prewarm=1)
    await worker.start()

    assert await worker.accept("a") and await worker.accept("b")
    assert await worker.accept("c") is False
    assert worker.stats.rejected == 1
    assert worker.can_accept() is False

    await worker.shutdown()


async def test_capacity_frees_up_when_a_call_ends(tmp_path, monkeypatch):
    worker = make_worker(tmp_path, monkeypatch, capacity=1, prewarm=1)
    await worker.start()
    await worker.accept("a")

    session = worker.session("a")
    await _settle(session)
    session.hangup("caller_left")
    for _ in range(500):
        await asyncio.sleep(0)
        if worker.can_accept():
            break

    assert worker.stats.active == 0
    assert worker.stats.completed == 1
    assert await worker.accept("b") is True

    await worker.shutdown()


async def test_duplicate_call_id_is_not_started_twice(tmp_path, monkeypatch):
    worker = make_worker(tmp_path, monkeypatch, capacity=4, prewarm=1)
    await worker.start()

    assert await worker.accept("a") is True
    assert await worker.accept("a") is False
    assert worker.stats.active == 1

    await worker.shutdown()


# --- isolation -----------------------------------------------------------------------------


async def test_concurrent_sessions_share_no_conversation_state(tmp_path, monkeypatch):
    """The defect that forced one process per call: caller #2 inheriting caller #1's history."""
    worker = make_worker(tmp_path, monkeypatch, capacity=3, prewarm=2)
    await worker.start()
    for i in range(3):
        await worker.accept(f"call-{i}")

    sessions = [worker.session(f"call-{i}") for i in range(3)]
    for i, session in enumerate(sessions):
        await _settle(session)
        session.emit(ev.FinalTranscript(t=time.monotonic(), text=f"caller {i} speaking"))
    for session in sessions:
        await _settle(session)

    for i, session in enumerate(sessions):
        spoken = [m["content"] for m in session.state.messages if m["role"] == "user"]
        assert spoken == [f"caller {i} speaking"]
        assert session.call_id == f"call-{i}"
        assert session.metrics is not sessions[(i + 1) % 3].metrics

    await worker.shutdown()


async def test_one_failing_session_does_not_take_down_the_worker(tmp_path, monkeypatch):
    monkeypatch.setenv("CLINIC_LOG_DIR", str(tmp_path))

    class ExplodingSession(QuietSession):
        async def _drain(self):
            raise RuntimeError("boom")

    def factory(s: Settings, call_id: str):
        cls = ExplodingSession if call_id == "bad" else QuietSession
        return cls(s, call_id=call_id)

    worker = Worker(settings(), capacity=4, prewarm=1, session_factory=factory)
    await worker.start()
    await worker.accept("bad")
    await worker.accept("good")

    for _ in range(500):
        await asyncio.sleep(0)
        if worker.stats.failed:
            break

    assert worker.stats.failed == 1
    assert worker.session("good") is not None  # the healthy call is untouched

    await worker.shutdown()


# --- prewarming ----------------------------------------------------------------------------


async def test_prewarmed_analyzers_are_handed_over_and_the_pool_refills(tmp_path, monkeypatch):
    """Cold start must not land inside the caller's first impression."""
    worker = make_worker(tmp_path, monkeypatch, capacity=4, prewarm=2)
    await worker.start()
    assert worker.stats.prewarmed == 2

    pooled = list(worker._pool)
    await worker.accept("a")

    assert worker.stats.prewarm_hits == 1
    assert worker.session("a")._turn._vad in pooled
    assert worker.stats.prewarmed == 2  # refilled behind the call

    await worker.shutdown()


async def test_running_out_of_prewarmed_analyzers_is_recorded_not_fatal(tmp_path, monkeypatch):
    worker = make_worker(tmp_path, monkeypatch, capacity=4, prewarm=0)
    await worker.start()

    assert await worker.accept("a") is True
    assert worker.stats.prewarm_misses == 1
    assert worker.session("a")._turn._vad is not None  # built on demand instead

    await worker.shutdown()


# --- shutdown ------------------------------------------------------------------------------


async def test_drain_lets_in_flight_calls_finish(tmp_path, monkeypatch):
    """A deploy should not hang up on a caller mid-booking."""
    worker = make_worker(tmp_path, monkeypatch, capacity=2, prewarm=1)
    await worker.start()
    await worker.accept("a")
    session = worker.session("a")
    await _settle(session)

    drainer = asyncio.create_task(worker.drain(timeout=5.0))
    await asyncio.sleep(0)
    assert worker.can_accept() is False  # sheds new calls immediately

    session.hangup("caller_left")
    await asyncio.wait_for(drainer, timeout=5)

    assert worker.stats.completed == 1
    assert session.state.phase is Phase.CLOSED


async def test_drain_gives_up_and_hangs_up_after_the_timeout(tmp_path, monkeypatch):
    """A call that never ends cannot hold a deploy open forever."""
    worker = make_worker(tmp_path, monkeypatch, capacity=2, prewarm=1)
    await worker.start()
    await worker.accept("a")
    await _settle(worker.session("a"))

    await asyncio.wait_for(worker.drain(timeout=0.05), timeout=5)
    assert worker.stats.active == 0


async def test_outcomes_are_tallied_for_the_fleet_view(tmp_path, monkeypatch):
    worker = make_worker(tmp_path, monkeypatch, capacity=2, prewarm=1)
    await worker.start()
    await worker.accept("a")
    session = worker.session("a")
    await _settle(session)
    session.hangup("caller_left")

    for _ in range(500):
        await asyncio.sleep(0)
        if worker.stats.completed:
            break

    assert worker.stats.outcomes == {"abandoned": 1}
    assert worker.stats.as_dict()["free"] == 2

    await worker.shutdown()


@pytest.mark.parametrize("capacity,active,expected_free", [(8, 0, 8), (8, 3, 5), (8, 8, 0)])
def test_stats_free_capacity(capacity, active, expected_free):
    from clinic_agent.core.worker import WorkerStats

    stats = WorkerStats(capacity=capacity, active=active)
    assert stats.free == expected_free
    assert stats.load == pytest.approx(active / capacity)

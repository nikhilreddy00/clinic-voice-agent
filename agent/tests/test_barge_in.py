"""Unit tests for the barge-in mic-gate logic (Phase 4).

These cover the pure decision state machine in clinic_agent.barge_in — the sustained-speech vs
echo-tail distinction, the false/true-positive accounting, and the RMS helper. They deliberately
do NOT import clinic_agent.pipeline (that pulls in the pyaudio transport); end-to-end barge-in is
audio-timing behavior and can only be verified on a live mic (see the Phase-4 notes).

Run with the agent venv, either directly (no pytest needed):
    agent/.venv/bin/python agent/tests/test_barge_in.py
or, if pytest is installed (uv sync --extra dev):
    agent/.venv/bin/python -m pytest agent/tests/test_barge_in.py
"""

from __future__ import annotations

import array
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from clinic_agent.barge_in import BargeInConfig, MicGateLogic, frame_rms  # noqa: E402

# Default gate: 600 ms sustained @ 20 ms/frame = 30 voiced frames to open; RMS voiced >= 500.
CFG = BargeInConfig()
FRAME_MS = 20.0
VOICED = 800.0     # above the 500 RMS threshold
QUIET = 100.0      # below it
FRAMES_TO_OPEN = int(CFG.sustained_ms / FRAME_MS)  # 30


def _feed(logic: MicGateLogic, now: float, rms: float, n: int) -> list[str]:
    return [logic.on_audio(now, rms, FRAME_MS) for _ in range(n)]


def test_idle_passes_all_audio():
    """With the bot silent, every frame passes regardless of energy."""
    logic = MicGateLogic(CFG)
    assert _feed(logic, now=1.0, rms=VOICED, n=5) == ["PASS"] * 5
    assert _feed(logic, now=1.0, rms=QUIET, n=5) == ["PASS"] * 5
    assert logic.suppressed_frames == 0


def test_short_blip_during_bot_speech_is_suppressed():
    """A voiced burst shorter than sustained_ms (echo tail / cough) never opens the gate."""
    logic = MicGateLogic(CFG)
    logic.on_bot_started(now=1.0)
    decisions = _feed(logic, now=1.0, rms=VOICED, n=FRAMES_TO_OPEN - 1)  # 29 < 30
    assert decisions == ["SUPPRESS"] * (FRAMES_TO_OPEN - 1)
    assert logic.bargein_candidates == 0
    assert logic.suppressed_frames == FRAMES_TO_OPEN - 1


def test_quiet_frame_resets_the_voiced_run():
    """A quiet frame breaks the run, so two sub-threshold bursts don't accumulate into a barge-in."""
    logic = MicGateLogic(CFG)
    logic.on_bot_started(now=1.0)
    _feed(logic, 1.0, VOICED, FRAMES_TO_OPEN - 5)
    assert logic.on_audio(1.0, QUIET, FRAME_MS) == "SUPPRESS"  # resets run
    decisions = _feed(logic, 1.0, VOICED, FRAMES_TO_OPEN - 1)
    assert "OPEN" not in decisions
    assert logic.bargein_candidates == 0


def test_sustained_speech_opens_the_gate():
    """>= sustained_ms of continuous voiced audio opens the gate (OPEN once, then PASS)."""
    logic = MicGateLogic(CFG)
    logic.on_bot_started(now=1.0)
    decisions = _feed(logic, now=1.0, rms=VOICED, n=FRAMES_TO_OPEN + 3)
    assert decisions[: FRAMES_TO_OPEN - 1] == ["SUPPRESS"] * (FRAMES_TO_OPEN - 1)
    assert decisions[FRAMES_TO_OPEN - 1] == "OPEN"
    assert decisions[FRAMES_TO_OPEN:] == ["PASS"] * 3
    assert logic.bargein_candidates == 1
    assert logic.candidate_pending is True


def test_true_positive_when_interruption_follows_open():
    """Gate opens, a real interruption is observed → counted true, candidate cleared."""
    logic = MicGateLogic(CFG)
    logic.on_bot_started(now=1.0)
    _feed(logic, 1.0, VOICED, FRAMES_TO_OPEN)
    assert logic.on_interruption(now=1.1) == "true_positive"
    assert logic.bargein_true == 1
    assert logic.candidate_pending is False
    # A later bot-start must NOT also count it as a false positive.
    assert logic.on_bot_started(now=2.0) is None
    assert logic.bargein_false == 0


def test_false_positive_when_no_interruption_follows():
    """Echo opens the gate but no real turn arrives → next bot-start resolves it as false."""
    logic = MicGateLogic(CFG)
    logic.on_bot_started(now=1.0)
    _feed(logic, 1.0, VOICED, FRAMES_TO_OPEN)
    logic.on_bot_stopped(now=1.1)
    assert logic.on_bot_started(now=2.0) == "false_positive"
    assert logic.bargein_false == 1
    assert logic.bargein_true == 0


def test_shutdown_resolves_dangling_candidate_as_false():
    logic = MicGateLogic(CFG)
    logic.on_bot_started(now=1.0)
    _feed(logic, 1.0, VOICED, FRAMES_TO_OPEN)
    logic.resolve_pending_on_shutdown()
    assert logic.bargein_false == 1


def test_hangover_suppresses_echo_tail_then_reopens():
    """After the bot stops, a normal-stop hangover keeps suppressing briefly, then audio passes."""
    logic = MicGateLogic(CFG)
    logic.on_bot_started(now=1.0)
    logic.on_bot_stopped(now=1.0)  # gate never opened -> hangover applies
    # Within the 0.4 s hangover: still muted, short audio suppressed.
    assert logic.on_audio(now=1.0 + 0.1, rms=VOICED, frame_ms=FRAME_MS) == "SUPPRESS"
    # After the hangover: back to passthrough.
    assert logic.on_audio(now=1.0 + 0.5, rms=VOICED, frame_ms=FRAME_MS) == "PASS"


def test_active_bargein_skips_hangover():
    """If the caller barged in (gate open), the bot stopping must NOT re-mute their speech."""
    logic = MicGateLogic(CFG)
    logic.on_bot_started(now=1.0)
    _feed(logic, 1.0, VOICED, FRAMES_TO_OPEN)  # gate_open = True
    logic.on_bot_stopped(now=1.1)              # barge-in in progress -> no hangover
    assert logic.muted(now=1.1) is False
    assert logic.on_audio(now=1.1, rms=VOICED, frame_ms=FRAME_MS) == "PASS"


def test_frame_rms():
    assert frame_rms(b"") == 0.0
    assert frame_rms(array.array("h", [0] * 100).tobytes()) == 0.0
    # Constant amplitude -> RMS equals that amplitude.
    assert abs(frame_rms(array.array("h", [1000] * 100).tobytes()) - 1000.0) < 1e-6
    # Odd trailing byte is tolerated (guarded), not a crash.
    assert frame_rms(array.array("h", [500] * 10).tobytes() + b"\x01") > 0.0


def _run_all() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())

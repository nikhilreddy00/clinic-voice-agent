"""Barge-in mic-gate logic (Phase 4).

Pure, dependency-free state machine + an RMS helper for the mic gate, split out from
``pipeline.py`` so it can be unit-tested WITHOUT importing the pyaudio-backed audio transport
(importing ``pipeline`` pulls in PyAudio). ``pipeline.BotSpeechInputGate`` is the thin Pipecat
``FrameProcessor`` that wraps :class:`MicGateLogic`.

Why this exists (Phase-4 barge-in hardening):
  The Phase-1 gate was half-duplex — it dropped ALL mic audio while the bot spoke, so the
  laptop's own TTS echo (there is no acoustic echo cancellation on a local mic) never reached
  VAD/STT and could not create a feedback loop. The cost: the caller could not interrupt.

  This gate keeps the echo protection but allows genuine barge-in. While the bot speaks it
  suppresses input by default, but SUSTAINED voiced input — at least ``sustained_ms`` of
  continuous above-threshold audio — opens the gate and passes audio through, so a real
  interruption reaches STT and Pipecat's ``MinWordsUserTurnStartStrategy`` (the turn-level
  barge-in gate that actually broadcasts the interruption). A short ``hangover_secs`` window
  after the bot stops still swallows the echo tail. Short blips / the echo tail stay
  suppressed; sustained speech barges in.

Honest limitation: with no AEC on the local mic, sustained bot echo CAN itself open the gate.
The min-words turn strategy is the second line of defense, and every gate-open during bot
speech that does NOT lead to a real interruption is counted as a FALSE POSITIVE (a trust
metric worth reporting). Fully robust barge-in needs an AEC-equipped transport (WebRTC /
telephony), which arrives in a later phase.
"""

from __future__ import annotations

import array
import math
import os
from dataclasses import dataclass


def frame_rms(audio: bytes) -> float:
    """Root-mean-square amplitude of 16-bit PCM ``audio`` (host-native byte order).

    Returns 0.0 for empty input. Local dev targets (macOS/Linux, x86-64/arm64) are all
    little-endian, matching PyAudio's native int16 delivery, so no byte-swap is needed.
    """
    if not audio:
        return 0.0
    usable = len(audio) - (len(audio) % 2)  # guard against a stray odd byte
    if usable <= 0:
        return 0.0
    samples = array.array("h")
    samples.frombytes(audio[:usable])
    acc = 0
    for s in samples:
        acc += s * s
    return math.sqrt(acc / len(samples))


def barge_in_min_words() -> int:
    """Min transcribed words to trigger a barge-in while the bot is speaking (env override)."""
    return int(os.getenv("CLINIC_BARGEIN_MIN_WORDS", "3"))


@dataclass(frozen=True)
class BargeInConfig:
    """Tunable thresholds for the mic gate (behavioral, not secrets)."""

    hangover_secs: float = 0.4      # keep muting this long after the bot stops (echo tail)
    sustained_ms: float = 600.0     # continuous voiced audio needed to open the gate mid-speech
    voice_rms_threshold: float = 500.0  # int16 RMS above which a frame counts as "voiced"

    @classmethod
    def from_env(cls) -> "BargeInConfig":
        return cls(
            hangover_secs=float(os.getenv("CLINIC_BARGEIN_HANGOVER_SECS", "0.4")),
            sustained_ms=float(os.getenv("CLINIC_BARGEIN_SUSTAINED_MS", "600")),
            voice_rms_threshold=float(os.getenv("CLINIC_BARGEIN_RMS", "500")),
        )


class MicGateLogic:
    """Pure state machine for the barge-in mic gate.

    Fed timestamped events (bot start/stop, per-frame audio energy, observed interruptions) and
    returns a per-frame decision plus running metrics. No Pipecat/audio deps so it is unit
    testable. ``now`` is a monotonic seconds float supplied by the caller.

    Audio decision values: ``"PASS"`` (forward), ``"SUPPRESS"`` (drop as echo/short blip),
    ``"OPEN"`` (forward — this is the first frame of a sustained-speech barge-in candidate).
    """

    def __init__(self, config: BargeInConfig | None = None) -> None:
        self.cfg = config or BargeInConfig()
        self.bot_speaking = False
        self.unmute_at = 0.0
        self.voiced_run_ms = 0.0
        self.gate_open = False
        self.candidate_pending = False
        # Metrics (resume trust factors).
        self.suppressed_frames = 0
        self.bargein_candidates = 0   # gate-opens during bot speech
        self.bargein_true = 0         # candidates followed by a real interruption
        self.bargein_false = 0        # candidates that were echo (no interruption followed)

    def muted(self, now: float) -> bool:
        return self.bot_speaking or now < self.unmute_at

    def on_bot_started(self, now: float) -> str | None:
        """Bot began speaking. Resolves any still-pending candidate as a false positive."""
        resolved = self._resolve_pending_as_false()
        self.bot_speaking = True
        self.voiced_run_ms = 0.0
        self.gate_open = False
        return resolved

    def on_bot_stopped(self, now: float) -> None:
        """Bot stopped speaking. Apply the echo-tail hangover UNLESS a real barge-in is live.

        If ``gate_open`` the caller is mid-utterance (they barged in), so re-muting for the
        hangover would clip their speech — skip it and keep passing audio.
        """
        self.bot_speaking = False
        self.unmute_at = 0.0 if self.gate_open else now + self.cfg.hangover_secs

    def on_interruption(self, now: float) -> str | None:
        """An InterruptionFrame / UserStartedSpeaking was observed. Confirms a pending candidate."""
        if self.candidate_pending:
            self.candidate_pending = False
            self.bargein_true += 1
            return "true_positive"
        return None

    def on_audio(self, now: float, rms: float, frame_ms: float) -> str:
        """Decide what to do with one input audio frame; updates gate/candidate state."""
        if not self.muted(now):
            self.voiced_run_ms = 0.0
            self.gate_open = False
            return "PASS"

        # Muted: accumulate a run of consecutive voiced frames; a quiet frame breaks it.
        if rms >= self.cfg.voice_rms_threshold:
            self.voiced_run_ms += frame_ms
        else:
            self.voiced_run_ms = 0.0

        if not self.gate_open and self.voiced_run_ms >= self.cfg.sustained_ms:
            self.gate_open = True
            self.candidate_pending = True
            self.bargein_candidates += 1
            return "OPEN"
        if self.gate_open:
            return "PASS"

        self.suppressed_frames += 1
        return "SUPPRESS"

    def resolve_pending_on_shutdown(self) -> None:
        """Count any never-confirmed candidate as a false positive (call at teardown)."""
        self._resolve_pending_as_false()

    def stats(self) -> dict:
        return {
            "suppressed_frames": self.suppressed_frames,
            "bargein_candidates": self.bargein_candidates,
            "bargein_true": self.bargein_true,
            "bargein_false": self.bargein_false,
        }

    def _resolve_pending_as_false(self) -> str | None:
        if self.candidate_pending:
            self.candidate_pending = False
            self.bargein_false += 1
            return "false_positive"
        return None

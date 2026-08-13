"""Phase 10 — TurnEngine: who has the floor.

Sits between the media adapter and STT. Consumes raw 16 kHz PCM and emits only *decisions*
(:class:`SpeechStarted`, :class:`SpeechStopped`, :class:`UserInterrupted`) — audio itself never
reaches the reducer, which is what keeps call traces small and replayable.

It combines the two gates Phase 4 established, unchanged in behavior:

* **Audio level** — :class:`~clinic_agent.barge_in.MicGateLogic`, ported forward verbatim.
  While the bot speaks, input is suppressed by default; ~600 ms of sustained voiced audio
  opens the gate, and a 400 ms hangover after playback swallows the echo tail. On a laptop mic
  with no acoustic echo cancellation this is what stops the bot hearing itself.
* **Turn level** — a minimum word count on interim transcripts, the direct replacement for
  Pipecat's ``MinWordsUserTurnStartStrategy``. Opening the audio gate only lets speech reach
  STT; it takes ``min_words`` transcribed words while the bot is speaking to actually declare a
  barge-in. Short sounds and echo blips get past the first gate and are stopped by this one.

Every gate-open during bot speech that is never confirmed by a real interruption is counted as
a false positive — the same trust metric Phase 4 reported.

VAD runs in ``SileroVADAnalyzer``, kept from Pipecat. It is a plain ONNX wrapper with an
``analyze_audio()`` coroutine that already offloads inference to a thread, not a pipeline
component — re-implementing it would buy nothing and lose a well-tuned model.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from loguru import logger

from pipecat.audio.vad.vad_analyzer import VADState

from ...barge_in import BargeInConfig, MicGateLogic, barge_in_min_words, frame_rms
from .. import events as ev
from ..audio import INPUT_SAMPLE_RATE, duration_ms
from ..vad import SharedSileroVAD

EmitFn = Callable[[ev.Event], None]
ForwardFn = Callable[[bytes], Awaitable[None]]


class TurnEngine:
    """Turn-taking and barge-in decisions for one call."""

    def __init__(
        self,
        emit: EmitFn,
        forward_audio: ForwardFn,
        *,
        config: BargeInConfig | None = None,
        min_words: int | None = None,
        sample_rate: int = INPUT_SAMPLE_RATE,
        vad: SharedSileroVAD | None = None,
    ) -> None:
        self._emit = emit
        self._forward = forward_audio
        self._gate = MicGateLogic(config or BargeInConfig.from_env())
        self._min_words = min_words if min_words is not None else barge_in_min_words()
        self._sample_rate = sample_rate

        # Phase 11: SharedSileroVAD shares the ONNX weights and inference pool across every
        # session in the process — measured at 7.97 MB/0.46 ms per session against Pipecat's
        # 7.97 MB/23.4 ms, i.e. ~8 GB of duplicated weights avoided at 1,000 sessions — while
        # producing bit-identical confidences. `vad` is injectable so a prewarmed session can
        # hand over an already-constructed analyzer.
        self._vad = vad or SharedSileroVAD(sample_rate=sample_rate)
        self._vad.set_sample_rate(sample_rate)
        self._vad_state = VADState.QUIET

        self._bot_speaking = False
        self._interrupt_sent = False  # one interruption per bot utterance, not one per frame

    # --- audio in -----------------------------------------------------------------------

    async def feed(self, pcm: bytes, now: float) -> None:
        """Process one input audio frame: gate it, then run VAD on what survives."""
        if not pcm:
            return

        decision = self._gate.on_audio(now, frame_rms(pcm), duration_ms(pcm, self._sample_rate))
        if decision == "OPEN":
            logger.info(
                f"[barge-in] sustained input during bot speech → opening mic "
                f"(candidate #{self._gate.bargein_candidates})"
            )
        elif decision == "SUPPRESS":
            if self._gate.suppressed_frames % 50 == 1:  # ~1 s heartbeat, not one line per frame
                logger.info(
                    f"[mic-gate] suppressing echo while bot speaks "
                    f"(dropped {self._gate.suppressed_frames} frames)"
                )
            return

        await self._forward(pcm)
        await self._run_vad(pcm, now)

    async def _run_vad(self, pcm: bytes, now: float) -> None:
        state = await self._vad.analyze_audio(pcm)
        if state == self._vad_state:
            return

        previous, self._vad_state = self._vad_state, state
        if state == VADState.SPEAKING:
            self._emit(ev.SpeechStarted(t=now))
        elif state == VADState.QUIET and previous in (VADState.SPEAKING, VADState.STOPPING):
            # The honest start of the voice-to-voice latency clock.
            self._emit(ev.SpeechStopped(t=now))

    # --- transcripts --------------------------------------------------------------------

    def on_partial_transcript(self, text: str, now: float) -> None:
        """Turn-level barge-in gate: enough real words while the bot speaks is an interruption."""
        if not self._bot_speaking or self._interrupt_sent:
            return
        if len(text.split()) < self._min_words:
            return
        self._interrupt_sent = True
        if self._gate.on_interruption(now) == "true_positive":
            logger.info(
                f"[barge-in] real interruption — caller barged in over the bot "
                f"(total real={self._gate.bargein_true})"
            )
        self._emit(ev.UserInterrupted(t=now))

    # --- playback state (driven by the session from bot speaking events) ----------------

    def on_bot_started(self, now: float) -> None:
        if self._gate.on_bot_started(now) == "false_positive":
            logger.info(
                f"[barge-in] false-positive interruption — echo opened the gate but no real "
                f"turn followed (total false={self._gate.bargein_false})"
            )
        self._bot_speaking = True
        self._interrupt_sent = False
        logger.info("[mic-gate] bot speaking → guarding mic (sustained speech can still barge in)")

    def on_bot_stopped(self, now: float) -> None:
        self._gate.on_bot_stopped(now)
        self._bot_speaking = False

    # --- teardown -----------------------------------------------------------------------

    def stats(self) -> dict:
        """Resolve any dangling candidate and return the barge-in counters (teardown log)."""
        self._gate.resolve_pending_on_shutdown()
        return self._gate.stats()

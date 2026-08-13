"""Phase 11 — a VAD analyzer that can afford N sessions in one process.

Pipecat's ``SileroVADAnalyzer`` is correct and well-tuned, and Phase 10 kept it. It is also
built on the assumption that a process runs **one** call, and two of its costs are per-instance:

* it loads its own copy of the ONNX model — measured at **7.9 MB and ~24 ms per instance** on
  this machine, so 1,000 sessions would be ~7.9 GB of byte-identical weights and 24 s of
  cold start;
* it allocates its own ``ThreadPoolExecutor(max_workers=1)``, so N concurrently-speaking
  sessions means N OS threads all contending for the GIL.

Neither is inherent. ``onnxruntime.InferenceSession`` is stateless and thread-safe — Silero's
recurrent state lives in the small ``_state`` / ``_context`` arrays that are passed *in* as
tensors on every call. So the weights can be loaded once per process and the per-stream state
stays per-session, which is what this module does. Everything else (the confidence/volume
smoothing, the start/stop hysteresis) is inherited from Pipecat unchanged, because that is the
tuned part and there is nothing to gain by reimplementing it.

The executor is likewise shared and bounded: inference is CPU work, so the useful ceiling is
core count, not session count.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams

_MODEL_RESET_STATES_TIME = 5.0  # matches Pipecat; keeps recurrent state from drifting

_shared_session = None
_shared_executor: ThreadPoolExecutor | None = None


def _model_path() -> str:
    from importlib import resources

    return str(resources.files("pipecat.audio.vad.data").joinpath("silero_vad.onnx"))


def shared_inference_session():
    """Process-wide Silero ONNX session, loaded on first use.

    Thread-safety: ``InferenceSession.run`` is safe to call concurrently; the per-stream
    recurrent state is passed in as an input tensor, never held on the session.
    """
    global _shared_session
    if _shared_session is None:
        import onnxruntime

        opts = onnxruntime.SessionOptions()
        # One thread per run, with parallelism coming from the executor instead. Letting ORT
        # spawn its own pools per session is how a worker ends up with hundreds of threads
        # fighting over a handful of cores.
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        _shared_session = onnxruntime.InferenceSession(
            _model_path(), providers=["CPUExecutionProvider"], sess_options=opts
        )
        logger.info("[vad] loaded shared Silero ONNX session (one per process)")
    return _shared_session


def shared_executor() -> ThreadPoolExecutor:
    """Bounded, process-wide pool for VAD inference.

    Sized to cores because this is CPU-bound: giving it one thread per session would add
    context-switching to a workload that is already core-limited. ``CLINIC_VAD_THREADS``
    overrides for load-test sweeps.
    """
    global _shared_executor
    if _shared_executor is None:
        workers = int(os.getenv("CLINIC_VAD_THREADS", "0")) or min(8, (os.cpu_count() or 4))
        _shared_executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="vad")
        logger.info(f"[vad] shared inference pool: {workers} thread(s)")
    return _shared_executor


class _SharedStateModel:
    """Per-stream Silero state on top of the shared session.

    A drop-in for Pipecat's ``SileroOnnxModel``: same call signature, same reset semantics,
    but holding only the ~1 KB of recurrent state instead of the whole model.
    """

    def __init__(self) -> None:
        self._session = shared_inference_session()
        self.sample_rates = [8000, 16000]
        self.reset_states()

    def reset_states(self, batch_size: int = 1) -> None:
        self._state = np.zeros((2, batch_size, 128), dtype="float32")
        self._context = np.zeros((batch_size, 0), dtype="float32")
        self._last_sr = 0
        self._last_batch_size = 0

    def __call__(self, x, sr: int):
        if np.ndim(x) == 1:
            x = np.expand_dims(x, 0)
        num_samples = 512 if sr == 16000 else 256
        if np.shape(x)[-1] != num_samples:
            raise ValueError(f"expected {num_samples} samples at {sr} Hz, got {np.shape(x)[-1]}")

        batch_size = np.shape(x)[0]
        context_size = 64 if sr == 16000 else 32
        if self._last_sr and self._last_sr != sr:
            self.reset_states(batch_size)
        if self._last_batch_size and self._last_batch_size != batch_size:
            self.reset_states(batch_size)
        if not np.shape(self._context)[1]:
            self._context = np.zeros((batch_size, context_size), dtype="float32")

        x = np.concatenate((self._context, x), axis=1)
        out, state = self._session.run(
            None, {"input": x, "state": self._state, "sr": np.array(sr, dtype="int64")}
        )
        self._state = state
        self._context = x[..., -context_size:]
        self._last_sr = sr
        self._last_batch_size = batch_size
        return out


class SharedSileroVAD(SileroVADAnalyzer):
    """Silero VAD sharing the ONNX weights and inference pool across every session.

    Inherits Pipecat's analyzer wholesale — the state machine and smoothing are the tuned part
    — and swaps only the two things that were per-instance for no reason.
    """

    def __init__(self, *, sample_rate: int | None = None, params: VADParams | None = None):
        # Deliberately skips SileroVADAnalyzer.__init__, which loads a private copy of the
        # model. Everything it sets up beyond that is initialized here.
        super(SileroVADAnalyzer, self).__init__(sample_rate=sample_rate, params=params)
        self._model = _SharedStateModel()
        self._last_reset_time = 0.0
        self._executor = shared_executor()

    def reset(self) -> None:
        """Clear recurrent state so a pooled analyzer can be reused by the next call.

        Both halves matter. The model's recurrent state is a rolling window of the *previous*
        caller's audio, so without the reset a reused analyzer starts the next call believing
        it is mid-utterance. ``set_params`` re-arms the start/stop hysteresis counters, and is
        skipped when no sample rate has been set yet — the base class derives its frame counts
        from the rate and would divide by zero.
        """
        self._model.reset_states()
        self._last_reset_time = time.time()
        if self.sample_rate:
            self.set_params(self._params)

"""Pipecat pipeline for the clinic voice agent.

=======================  SKELETON — DO NOT IMPLEMENT IN PHASE 0  =======================

This file intentionally contains NO working pipeline logic yet. It documents the intended
Pipecat wiring with real import paths (verified against current Pipecat docs) so Phase 1 can
fill it in without re-deriving the structure.

The real imports below are kept COMMENTED so this module imports cleanly without pipecat-ai
installed (agent deps are not installed in Phase 0). Uncomment and implement in Phase 1.

Reference pipeline order (from Pipecat docs):
    transport.input() -> stt -> user_aggregator -> llm -> tts -> transport.output()
                       -> assistant_aggregator
"""

from __future__ import annotations

# --- Phase 1 imports (uncomment when implementing) ---------------------------------------
# from pipecat.audio.vad.silero import SileroVADAnalyzer
# from pipecat.pipeline.pipeline import Pipeline
# from pipecat.pipeline.worker import PipelineParams, PipelineWorker
# from pipecat.workers.runner import WorkerRunner
# from pipecat.processors.aggregators.llm_context import LLMContext
# from pipecat.processors.aggregators.llm_response_universal import (
#     LLMContextAggregatorPair,
#     LLMUserAggregatorParams,
# )
# from pipecat.services.deepgram.stt import DeepgramSTTService   # ASR
# from pipecat.services.groq.llm import GroqLLMService           # LLM (Llama on Groq)
# from pipecat.services.cartesia.tts import CartesiaTTSService   # TTS
# from pipecat.transports.base_transport import BaseTransport, TransportParams

# from .config import load_settings
# from .prompts import GREETING, SYSTEM_PROMPT


async def run_agent() -> None:
    """Build and run the voice-agent pipeline.

    TODO(Phase 1): implement the live ASR->LLM->TTS loop.
      1. settings = load_settings()
      2. Build transport (WebRTC for local dev; Twilio/LiveKit SIP in Phase 7).
      3. stt = DeepgramSTTService(api_key=settings.deepgram_api_key)
      4. llm = GroqLLMService(api_key=settings.groq_api_key, model=settings.groq_model)
      5. tts = CartesiaTTSService(api_key=..., voice_id=...)   # or Piper for free-tier
      6. context = LLMContext() seeded with SYSTEM_PROMPT; aggregators = LLMContextAggregatorPair(...)
      7. pipeline = Pipeline([transport.input(), stt, user_agg, llm, tts,
                              transport.output(), assistant_agg])
      8. worker = PipelineWorker(pipeline, params=PipelineParams(enable_metrics=True))
      9. On connect: deliver GREETING (AI disclosure). Run via WorkerRunner.

    TODO(Phase 2): register scheduling-API tool calls (availability / hold / confirm).
    TODO(Phase 3): enforce the state machine in docs/build_spec.md (validation, fallbacks).
    TODO(Phase 5): wire per-turn latency / ASR-confidence / tool-success observability.
    """
    raise NotImplementedError("Pipeline is a Phase 0 skeleton; implement starting in Phase 1.")


if __name__ == "__main__":
    # TODO(Phase 1): use `from pipecat.runner.run import main` entrypoint instead.
    raise SystemExit("clinic_agent.pipeline is a skeleton — nothing to run in Phase 0.")

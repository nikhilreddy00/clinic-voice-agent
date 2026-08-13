"""Phase 10 — Phase-6 latency metrics, re-sourced from events.

The Phase-6 :class:`~clinic_agent.metrics.LatencyCollector` is kept exactly as it is; only its
input changes. It used to be fed by three ``MetricsTap`` frame processors spliced into the
Pipecat pipeline at the STT/LLM/TTS boundaries. Here it is a plain subscriber on the one event
stream, which is strictly better: the boundaries are named events rather than frame classes
observed at the right position, so the failure mode that made Phase 6 record ``turns=0`` —
watching ``UserStoppedSpeakingFrame`` when the pipeline actually emits
``VADUserStoppedSpeakingFrame`` — cannot happen. There is one ``SpeechStopped`` event and it
means one thing.

Stage boundaries are unchanged, so the numbers stay comparable across engines:

    SpeechStopped                   -> turn starts (VAD silence)
    FinalTranscript                 -> ASR done
    first LLMTextDelta / LLMToolUse -> LLM first output (or first tool call)
    LLMCompleted                    -> LLM reply complete
    BotStartedSpeaking              -> first audio out, turn finalizes

Tool outcomes are recorded inside ``scheduling_tools.execute_tool`` on both engines, so they
are not duplicated here.
"""

from __future__ import annotations

from ..metrics import LatencyCollector
from . import events as ev


def record_event(collector: LatencyCollector, event: ev.Event) -> None:
    """Map one event to its LatencyCollector hook. Pure dispatch — unit-testable, no I/O."""
    if isinstance(event, ev.SpeechStopped):
        collector.on_user_stopped(event.t)
    elif isinstance(event, ev.FinalTranscript):
        if event.text.strip():
            collector.on_transcript(event.t, event.confidence)
    elif isinstance(event, ev.LLMToolUse):
        collector.on_llm_first_output(event.t, is_tool_call=True)
    elif isinstance(event, ev.LLMTextDelta):
        collector.on_llm_first_output(event.t)
    elif isinstance(event, ev.LLMCompleted):
        collector.on_llm_end(event.t)
    elif isinstance(event, ev.BotStartedSpeaking):
        collector.on_output_audio(event.t)

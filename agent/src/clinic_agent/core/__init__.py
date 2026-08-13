"""Phase 10 — the in-house orchestration loop.

Replaces Pipecat's ``Pipeline`` / ``FrameProcessor`` chain with an event loop this project
owns. Media transport (LiveKit SIP / the local audio device) and the vendor models are still
bought; what changed is that turn boundaries, cancellation, the tool loop, and call state are
now explicit, pure, and replayable instead of emergent behavior of a linear frame pipeline.

    events.py    typed events (the only things the reducer sees)
    actions.py   what the reducer asks adapters to do
    state.py     immutable CallState + the turn-phase FSM
    reducer.py   pure reduce(state, event) -> (state, actions)   <- the engine
    session.py   CallSession: one queue, one drain loop, all per-call state
    recorder.py  trace recording + deterministic replay
    telemetry.py Phase-6 latency metrics, re-sourced from events
    adapters/    media, turn-taking, STT, LLM, TTS, tools

Run it with ``python -m clinic_agent.core`` (``MODE=local`` or ``MODE=telephony``). The Pipecat
pipeline remains at ``clinic_agent.pipeline`` as the fallback path.
"""

from .reducer import reduce
from .session import CallSession
from .state import CallState, Phase

__all__ = ["CallSession", "CallState", "Phase", "reduce"]

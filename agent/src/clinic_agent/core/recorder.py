"""Phase 10 — call trace recording and deterministic replay.

Every event that reaches the reducer is appended to ``logs/traces/<call_id>.jsonl``. Because
the reducer is pure and every event here is plain data, that file is a complete, executable
description of the call: :func:`replay` runs it back through ``reduce()`` in milliseconds with
no audio, no network, and no API keys.

This is what the Phase-10 exit criterion is, and it becomes the Tier-1 eval in Phase 16 — the
one that runs on every commit. A regression in turn-taking, barge-in handling, or the tool
loop shows up as a diff in the replayed action sequence, from a trace captured on a real
phone call, without placing a phone call.

Traces contain caller utterances, so they are PHI-shaped. All data in this project is
synthetic (see CLAUDE.md), and ``logs/`` is git-ignored; Phase 17 adds the tagged redaction
boundary that makes this safe for real deployments.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

from loguru import logger

from ..metrics import resolve_log_dir
from .actions import Action
from .events import Event, event_from_dict
from .jsonl import JsonlWriter
from .reducer import reduce
from .state import CallState


def trace_dir() -> Path:
    """Directory holding per-call trace files (sibling of the Phase-6 calls.jsonl sink)."""
    return resolve_log_dir() / "traces"


class TraceRecorder:
    """Appends a call's reducer-facing events to one JSONL file.

    One instance per :class:`~clinic_agent.core.session.CallSession`. Writes are best-effort:
    a full disk or a read-only volume must never take down a live call, so a failed write is
    logged once and recording is disabled for the rest of the call.
    """

    # Traces are write-only until the call ends, so they buffer. Two reasons, both about
    # running many sessions in one worker: it cuts the per-record cost ~25x (0.9 us vs 22.4 us
    # for open/append/close), and it means a session holds a file descriptor only while
    # flushing — otherwise a worker hits the FD limit long before it runs out of CPU.
    BUFFER_LINES = 64

    def __init__(self, call_id: str, *, enabled: bool = True) -> None:
        self.call_id = call_id
        self.path = trace_dir() / f"{call_id}.jsonl"
        self._writer = JsonlWriter(self.path, buffer_lines=self.BUFFER_LINES) if enabled else None
        self.count = 0

    @property
    def enabled(self) -> bool:
        return self._writer is not None and self._writer.enabled

    def record(self, event: Event) -> None:
        if self._writer is None:
            return
        self._writer.write(event.to_dict())
        self.count += 1

    def close(self) -> None:
        """Flush the tail of the trace and release the descriptor. Called at call teardown."""
        if self._writer is not None:
            self._writer.close()


def load_trace(path: str | Path) -> list[Event]:
    """Read a trace file back into events, ordered by ``seq``.

    Sorting rather than trusting file order is deliberate: ``seq`` is stamped when an event is
    accepted onto the session queue, which is the order the reducer sees, and that is what a
    replay must reproduce. File order is a consequence of it, not the definition of it.
    """
    events: list[Event] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                events.append(event_from_dict(json.loads(line)))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{line_no}: bad trace line ({exc})") from exc
    return sorted(events, key=lambda e: e.seq)


def replay(events: Iterable[Event], state: CallState | None = None) -> tuple[CallState, list[Action]]:
    """Fold a whole event stream through the reducer. Returns the final state and every action."""
    state = state or CallState()
    actions: list[Action] = []
    for event in events:
        state, produced = reduce(state, event)
        actions.extend(produced)
    return state, actions


def replay_steps(
    events: Iterable[Event], state: CallState | None = None
) -> Iterator[tuple[Event, CallState, list[Action]]]:
    """Like :func:`replay` but yields each step — for pinpointing where a replay diverged."""
    state = state or CallState()
    for event in events:
        state, produced = reduce(state, event)
        yield event, state, produced

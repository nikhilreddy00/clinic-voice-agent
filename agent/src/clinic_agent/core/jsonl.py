"""Phase 11 — JSONL sinks that survive N sessions per process.

Phase 10 wrote both the metrics sink and the per-call trace with an ``open(...); write; close``
per record, directly on the event loop. That is fine for one call and wrong for a worker
hosting many: measured on this machine, per record —

    open/append/close   22.4 us
    persistent handle    2.1 us     (~10x cheaper)
    buffered, 64 lines   0.9 us     (~25x cheaper)

Twenty-two microseconds of blocking syscall is nothing next to a 900 ms turn, which is exactly
why it went unnoticed. It stops being nothing when it happens on the one thread that also has
to hand the audio device its next 20 ms frame, several thousand times a second across sessions.

Two access patterns, so two policies:

* **The metrics sink** (``logs/calls.jsonl``) is one file that every session in the process
  appends to, and the dashboard reads it live. Sessions therefore *share* one writer — one file
  descriptor per process rather than per call — and it stays unbuffered by default so
  freshness is unchanged. The event loop is single-threaded, so shared append needs no lock.
* **Traces** are one file per call, and nothing reads them until the call is over. They buffer,
  because a worker at high concurrency would otherwise hold one open descriptor per session and
  run into the file-descriptor limit long before it ran out of CPU.

Both are best-effort: a full disk or a read-only volume must never take down a live call.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from loguru import logger


class JsonlWriter:
    """Append-only JSONL sink with an optional line buffer.

    ``buffer_lines=1`` writes through immediately (still on a persistent handle, so it costs
    one syscall rather than three). Larger values amortize; ``flush_interval`` bounds how long
    a record can sit unwritten so a buffered sink is never indefinitely stale.
    """

    def __init__(
        self, path: str | Path, *, buffer_lines: int = 1, flush_interval: float = 2.0
    ) -> None:
        self.path = Path(path)
        self.buffer_lines = max(1, buffer_lines)
        self.flush_interval = flush_interval
        self.enabled = True
        self.written = 0

        self._handle = None
        self._buffer: list[str] = []
        self._last_flush = time.monotonic()

        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning(f"[jsonl] disabled — cannot create {self.path.parent}: {exc}")
            self.enabled = False

    def write(self, obj: dict) -> None:
        if not self.enabled:
            return
        self._buffer.append(json.dumps(obj, default=str))
        due = len(self._buffer) >= self.buffer_lines or (
            time.monotonic() - self._last_flush >= self.flush_interval
        )
        if due:
            self.flush()

    def flush(self) -> None:
        if not self.enabled or not self._buffer:
            return
        try:
            if self._handle is None:
                self._handle = self.path.open("a", encoding="utf-8")
            self._handle.write("\n".join(self._buffer) + "\n")
            self._handle.flush()
            self.written += len(self._buffer)
        except OSError as exc:
            # Observability must never take down the call.
            logger.warning(f"[jsonl] disabled — could not write {self.path}: {exc}")
            self.enabled = False
            self._handle = None
        finally:
            self._buffer.clear()
            self._last_flush = time.monotonic()

    def close(self) -> None:
        self.flush()
        if self._handle is not None:
            try:
                self._handle.close()
            except OSError:
                pass
            self._handle = None


# --- shared sinks ------------------------------------------------------------------------
# Keyed by resolved path so every session in a worker appends through one descriptor. Shared
# writers are never closed by an individual session — the process owns them (see close_shared).

_shared: dict[str, JsonlWriter] = {}


def shared_writer(path: str | Path, *, buffer_lines: int = 1) -> JsonlWriter:
    """Get (or create) the process-wide writer for ``path``."""
    key = str(Path(path).resolve())
    writer = _shared.get(key)
    if writer is None:
        writer = JsonlWriter(path, buffer_lines=buffer_lines)
        _shared[key] = writer
    return writer


def flush_shared() -> None:
    for writer in _shared.values():
        writer.flush()


def close_shared() -> None:
    """Flush and close every shared writer. Called at worker shutdown, not per call."""
    for writer in _shared.values():
        writer.close()
    _shared.clear()

#!/usr/bin/env python3
"""Export recorded call traces to the configured OTLP backend.

    agent/.venv/bin/python agent/scripts/export_traces_otel.py                 # eval/traces/*
    agent/.venv/bin/python agent/scripts/export_traces_otel.py logs/traces/X.jsonl
    agent/.venv/bin/python agent/scripts/export_traces_otel.py --dry-run       # print, send nothing

This exists because `core/otel.spans_from_events` derives spans from a FINISHED event stream
rather than opening them live. A call does not have to have been instrumented in advance to end
up in the APM — which is what makes it possible to populate and verify a Grafana dashboard
without placing a phone call. Live calls bill Cartesia, Deepgram and LiveKit (CLAUDE.md), so
"just make a call to check the dashboard" is not a verification step here.

Times are shifted so a trace recorded days ago lands in the dashboard's default window. The
DURATIONS are the recorded ones, untouched — only the origin moves. `--at` overrides.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "agent" / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / "agent" / ".env")

from clinic_agent.core.otel import (  # noqa: E402
    export_spans,
    otel_enabled,
    spans_from_events,
)
from clinic_agent.core.recorder import load_trace  # noqa: E402


def show(span, depth: int = 0) -> None:
    attrs = ", ".join(f"{k}={v}" for k, v in list(span.attributes.items())[:3])
    print(f"{'  ' * depth}{span.name:<24} {span.duration_ms:8.0f} ms  {attrs}")
    for child in span.children:
        show(child, depth + 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("traces", nargs="*", type=Path,
                        help="trace files (default: the eval/traces corpus)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the span trees and send nothing")
    parser.add_argument("--at", type=float, default=None,
                        help="unix seconds to land the FIRST trace at (default: an hour ago)")
    parser.add_argument("--spacing", type=float, default=120.0,
                        help="seconds between consecutive traces (default: 120)")
    args = parser.parse_args()

    paths = args.traces or sorted((REPO_ROOT / "eval" / "traces").glob("*.jsonl"))
    if not paths:
        print("no traces to export", file=sys.stderr)
        return 2

    if not args.dry_run and not otel_enabled():
        print("CLINIC_OTEL_ENDPOINT is not set — nothing to export to.\n"
              "Set it (and CLINIC_OTEL_HEADERS) in agent/.env, or pass --dry-run.",
              file=sys.stderr)
        return 2

    origin = args.at if args.at is not None else time.time() - 3600
    exported = 0

    for i, path in enumerate(paths):
        root = spans_from_events(load_trace(path))
        if root is None:
            print(f"[skip] {path.stem}: no events")
            continue

        # The recorded `t` values are monotonic seconds from an arbitrary process start, so they
        # mean nothing as wall clock. Anchor each trace's first event to a real timestamp and
        # let every span keep its own offset — the shape and every duration survive exactly.
        base = origin + i * args.spacing
        offset = base - root.start

        def to_ns(monotonic: float, _offset: float = offset) -> int:
            return int((monotonic + _offset) * 1e9)

        turns = root.attributes.get("call.turns", 0)
        print(f"[{i + 1}/{len(paths)}] {path.stem}  {root.duration_ms / 1000:6.1f}s  "
              f"{turns} turns  {root.attributes.get('call.tool_calls', 0)} tool calls")
        if args.dry_run:
            show(root, depth=1)
            continue

        if export_spans(root, to_ns):
            exported += 1
        else:
            print(f"       ! {path.stem} did not flush — the backend rejected it")

    if args.dry_run:
        print(f"\ndry run — {len(paths)} traces rendered, nothing sent")
        return 0

    endpoint = os.getenv("CLINIC_OTEL_ENDPOINT", "").strip().strip('"')
    print(f"\nexported {exported}/{len(paths)} traces to {endpoint}")
    print("Grafana: Explore -> Tempo -> Search, service.name = clinic-voice-agent")
    return 0 if exported == len(paths) else 1


if __name__ == "__main__":
    raise SystemExit(main())

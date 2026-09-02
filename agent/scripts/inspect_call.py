#!/usr/bin/env python3
"""Read a call trace and say what actually happened.

    agent/.venv/bin/python agent/scripts/inspect_call.py            # newest call
    agent/.venv/bin/python agent/scripts/inspect_call.py <call_id>  # a specific one

A call can sound flawless and have booked nothing. The console log will not tell you — the
event stream will, and this prints the parts that matter: what the caller said, which tools ran
and whether they succeeded, where the engine held a turn instead of interrupting, and the
verdict at the end.
"""

from __future__ import annotations

import glob
import json
import os
import sys

TRACES = os.path.join(os.path.dirname(__file__), "..", "..", "logs", "traces")


def load(arg: str | None) -> tuple[str, list[dict]]:
    if arg:
        path = arg if os.path.exists(arg) else os.path.join(TRACES, f"{arg}.jsonl")
    else:
        files = glob.glob(os.path.join(TRACES, "*.jsonl"))
        if not files:
            sys.exit(f"no traces in {os.path.abspath(TRACES)}")
        path = max(files, key=os.path.getmtime)
    return path, [json.loads(line) for line in open(path)]


def main() -> None:
    path, events = load(sys.argv[1] if len(sys.argv) > 1 else None)
    print(f"{os.path.basename(path)}  ({len(events)} events)\n")

    t0 = events[0]["t"]
    text_by_request: dict[str, str] = {}
    tools: list[tuple[str, bool]] = []
    held = released = nudges = 0
    booked = False

    for e in events:
        kind, t = e.get("kind"), e["t"] - t0
        if kind == "CallerPresent":
            print(f"{t:7.1f}  caller on the line (ANI {e.get('phone') or 'withheld'})")
        elif kind == "CallerMemoryLoaded":
            print(f"{t:7.1f}  memory: known={e.get('known')} upcoming={e.get('upcoming_appointments')}")
        elif kind == "FinalTranscript":
            print(f"{t:7.1f}  CALLER: {e.get('text')!r}")
        elif kind == "TurnHeld":
            held += 1
            if e.get("released"):
                released += 1
                print(f"{t:7.1f}    (patience ran out on an unfinished sentence: {e.get('tail')!r})")
        elif kind == "LLMTextDelta":
            text_by_request[e["request_id"]] = text_by_request.get(e["request_id"], "") + e["text"]
        elif kind == "LLMCompleted":
            said = text_by_request.get(e["request_id"], "").strip()
            if said:
                print(f"{t:7.1f}  AGENT : {said!r}")
        elif kind == "LLMToolUse":
            print(f"{t:7.1f}    -> {e.get('name')}({json.dumps(e.get('arguments'))[:110]})")
        elif kind == "ToolCompleted":
            ok = bool(e.get("ok"))
            tools.append((e.get("name"), ok))
            booked = booked or (e.get("name") == "confirm_booking" and ok)
            detail = "" if ok else f"  {str(e.get('result', {}).get('error'))[:70]!r}"
            print(f"{t:7.1f}    <- {e.get('name')} {'ok' if ok else 'FAILED'}{detail}")
        elif kind == "Hangup":
            print(f"{t:7.1f}  hangup ({e.get('reason')})")

    _unused_nudges = sum(
        1 for e in events
        if e.get("kind") == "LLMStarted"
    ) - sum(1 for e in events if e.get("kind") == "FinalTranscript")
    del _unused_nudges

    print("\n" + "-" * 60)
    if not tools:
        print("NO TOOL CALLS AT ALL — nothing was booked, changed, or looked up.")
    else:
        print("tools: " + ", ".join(f"{n}{'' if ok else ' (FAILED)'}" for n, ok in tools))
    print(f"booking committed: {'YES' if booked else 'no'}")
    print(f"turns held for a pause: {held}   of which cut off anyway: {released}")
    if released:
        print("  ^ the caller was talked over mid-sentence this many times")


if __name__ == "__main__":
    main()

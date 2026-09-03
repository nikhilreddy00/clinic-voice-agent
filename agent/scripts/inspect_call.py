#!/usr/bin/env python3
"""Read a call trace and say what actually happened.

    agent/.venv/bin/python agent/scripts/inspect_call.py            # newest call
    agent/.venv/bin/python agent/scripts/inspect_call.py <call_id>  # a specific one

A call can sound flawless and have booked nothing. The console log will not tell you — the
event stream will, and this prints the parts that matter: what the caller said, which tools ran
and whether they succeeded, where the engine held a turn instead of interrupting, the
per-stage latency breakdown, and the verdict at the end.

The latency table is the point of the stage split. A single voice-to-voice number cannot tell
you whether a slow call is the model, the endpointer, or the speech queue, and those have
completely different fixes. Measured on the 2026-09-03 calls, LLM TTFT was 52% of every turn —
which is why prompt caching (not endpointing, not TTS) was the thing worth changing.
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


def _pct(values: list[float], pct: float) -> float | None:
    """Linear-interpolated percentile. None on an empty sample rather than a misleading 0."""
    if not values:
        return None
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct / 100
    low = int(k)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (k - low)


def _latency(events: list[dict]) -> None:
    """Per-stage p50/p95 across the call.

    A turn is bounded by SpeechStopped -> BotStartedSpeaking, which is what the caller actually
    experiences: the moment they stop talking to the moment they hear a voice. Turns missing a
    boundary (an interrupted turn, the greeting) contribute to the stages they do have and are
    left out of the ones they do not, so a partial turn cannot silently deflate a percentile.
    """
    turns: list[dict] = []
    current: dict | None = None
    for e in events:
        kind, t = e.get("kind"), e["t"]
        if kind == "SpeechStopped":
            current = {"stop": t}
        elif current is None:
            continue
        elif kind == "FinalTranscript" and "final" not in current:
            current["final"] = t
        elif kind == "LLMStarted" and "llm" not in current:
            current["llm"] = t
        elif kind == "LLMTextDelta" and "token" not in current:
            current["token"] = t
        elif kind == "BotStartedSpeaking":
            current["speak"] = t
            turns.append(current)
            current = None

    stages = [
        ("endpointing  (speech stop -> transcript)", "stop", "final"),
        ("dispatch     (transcript -> LLM start)", "final", "llm"),
        ("LLM TTFT     (LLM start -> first token)", "llm", "token"),
        ("speech queue (first token -> speaking)", "token", "speak"),
        ("E2E          (speech stop -> speaking)", "stop", "speak"),
    ]
    print(f"latency over {len(turns)} complete turns"
          + ("" if turns else "  — no turn had both a speech-stop and a bot-start"))
    if not turns:
        return
    print(f"  {'stage':44s} {'n':>3s} {'p50':>8s} {'p95':>8s} {'max':>8s}")
    for label, a, b in stages:
        samples = [(x[b] - x[a]) * 1000 for x in turns if a in x and b in x]
        if not samples:
            print(f"  {label:44s}   -")
            continue
        print(f"  {label:44s} {len(samples):3d} "
              f"{_pct(samples, 50):7.0f}ms {_pct(samples, 95):7.0f}ms {max(samples):7.0f}ms")

    classify = [e["latency_ms"] for e in events
                if e.get("kind") == "IntentClassified" and e.get("latency_ms")]
    if classify:
        # Off the critical path (it runs in parallel), so it is reported apart from the stack
        # above — but it gates intent scoping from the NEXT turn, so a slow one still costs.
        print(f"  {'classifier   (parallel; scopes the NEXT turn)':44s} {len(classify):3d} "
              f"{_pct(classify, 50):7.0f}ms {_pct(classify, 95):7.0f}ms {max(classify):7.0f}ms")


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

    print("\n" + "-" * 60)
    _latency(events)
    print("-" * 60)
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

#!/usr/bin/env python3
"""Regenerate the landing page's call-replay data from a real result file.

Why a generator instead of pasting JSON once: the numbers and the confirmation code on that
page are the whole argument, and retyping them by hand is how a wrong one ships. This derives
them from the source file every time.

It reads either source this project produces:

* a **call trace** (``logs/traces/<call_id>.jsonl``) — the event stream of a real inbound phone
  call, which is what the page ships now that the engine has booked one end to end;
* an **eval result** (``eval/results/*.json``) — a text-in-the-loop case, which is what it used
  before a phone booking existed.

Run: python3 site/build_trace.py [path]
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAGE = Path(__file__).parent / "index.html"
# A real inbound call: greeting through confirmation, ending when the agent hung up by itself.
SOURCE = ROOT / "logs" / "traces" / "20260829T201039840380Z.jsonl"
CASE_ID = "hp_checkup_tomorrow"  # only consulted when the source is an eval result

BEGIN = "/* BEGIN GENERATED TRACE */"
END = "/* END GENERATED TRACE */"

# The eval result records utterances, assistant turns, and tool calls as three flat lists with
# no turn index linking them — the harness never needed one. These positions are INFERRED from
# the dialogue and corroborated by the assistant turns: availability is checked after the
# caller says any time works (turn 7), the slot is held when they accept (turn 8), and the
# booking is committed after the explicit yes (turn 9). The page therefore shows sequence and
# never a per-turn timestamp, because the source has none to show.
TOOL_AFTER_CALLER_TURN = {7: "check_availability", 8: "hold_slot", 9: "confirm_booking"}


def summarize(name: str, result: dict) -> dict:
    """Keep the part of a tool result a viewer can read in two seconds.

    confirm_booking's stored result carries null date_of_birth / new_patient / symptom_notes —
    a known eval-harness defect where those args are sent but not persisted. Showing nulls
    would misrepresent the call, so the summary takes only fields that are actually populated.
    """
    if name == "check_availability":
        return {"ok": result.get("ok"), "slots_found": result.get("count")}
    if name == "hold_slot":
        held = str(result.get("hold_id", ""))
        return {"ok": result.get("ok"), "hold_id": held[:8] + "…" if held else None}
    if name == "confirm_booking":
        return {
            "ok": result.get("ok"),
            "confirmation_id": result.get("confirmation_id"),
            "display_time": result.get("display_time"),
            "provider_name": result.get("provider_name"),
        }
    return {"ok": result.get("ok")}


def trim_args(args: dict) -> dict:
    """Shorten internal tokens for display.

    A hold_id is a 36-character UUID that would wrap across three lines and drown the fields a
    reader actually cares about. Truncating it also happens to say the right thing: the agent's
    own instructions class it as an internal token it must never speak aloud.
    """
    return {
        k: (v[:8] + "…" if k == "hold_id" and isinstance(v, str) and len(v) > 8 else v)
        for k, v in args.items()
    }


def build_steps_from_trace(path: Path) -> list[dict]:
    """Fold a call trace into replay steps, in the order the reducer actually saw them.

    Unlike the eval result, a trace records real ordering — so the beat the page is built
    around is preserved rather than reconstructed: ToolCompleted for the booking genuinely
    precedes the LLMCompleted that speaks the confirmation number aloud.

    Only the caller's words, the agent's words, and the tool calls come across. Everything else
    stays behind — including participant_id, which on the telephony path is the caller's real
    phone number.
    """
    events = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    events.sort(key=lambda e: e.get("seq", 0))

    args_by_id: dict[str, dict] = {}
    steps: list[dict] = []
    for e in events:
        kind = e.get("kind")
        if kind == "FinalTranscript" and e.get("text"):
            steps.append({"role": "caller", "text": e["text"]})
        elif kind == "LLMToolUse":
            args_by_id[e["tool_call_id"]] = e.get("arguments") or {}
        elif kind == "ToolCompleted":
            steps.append(
                {
                    "role": "tool",
                    "name": e["name"],
                    "args": trim_args(args_by_id.get(e["tool_call_id"], {})),
                    "result": summarize(e["name"], e.get("result") or {}),
                }
            )
        elif kind == "LLMCompleted" and (e.get("text") or "").strip():
            # A tool-only completion carries no text; its card already represents it.
            steps.append({"role": "agent", "text": e["text"].strip()})
    return steps


def build_steps_from_eval(path: Path) -> list[dict]:
    case = next(c for c in json.loads(path.read_text())["cases"] if c["id"] == CASE_ID)
    by_name = {t["name"]: t for t in case["tool_calls"]}
    steps: list[dict] = []

    for i, (caller, agent) in enumerate(
        zip(case["utterances"], case["assistant_turns"]), start=1
    ):
        steps.append({"role": "caller", "text": caller})
        tool_name = TOOL_AFTER_CALLER_TURN.get(i)
        if tool_name and tool_name in by_name:
            tool = by_name[tool_name]
            steps.append(
                {
                    "role": "tool",
                    "name": tool["name"],
                    "args": trim_args(tool.get("args") or {}),
                    "result": summarize(tool["name"], tool.get("result") or {}),
                }
            )
        steps.append({"role": "agent", "text": agent})
    return steps


def build_steps(path: Path) -> list[dict]:
    """Dispatch on what the file actually is, so either source just works."""
    if path.suffix == ".jsonl":
        return build_steps_from_trace(path)
    return build_steps_from_eval(path)


def main() -> int:
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else SOURCE
    if not source.exists():
        print(f"FAIL: source not found: {source}")
        return 1
    html = PAGE.read_text(encoding="utf-8")
    if BEGIN not in html or END not in html:
        print(f"FAIL: sentinels not found in {PAGE}")
        return 1

    steps = build_steps(source)
    block = "const CALL_REPLAY = " + json.dumps(steps, indent=2) + ";"
    pattern = re.compile(re.escape(BEGIN) + r".*?" + re.escape(END), re.DOTALL)
    # A function replacement, not a string: re.sub processes backslash escapes in a string
    # replacement, and JSON is full of them (…, \", \n). A literal block must bypass that.
    replacement = f"{BEGIN}\n{block}\n{END}"
    PAGE.write_text(pattern.sub(lambda _: replacement, html), encoding="utf-8")
    tools = sum(1 for s in steps if s["role"] == "tool")
    kind = "real phone call" if source.suffix == ".jsonl" else "eval case"
    print(f"OK: regenerated from {source.name} ({kind}) — {len(steps)} steps, {tools} tool calls")
    return 0


if __name__ == "__main__":
    sys.exit(main())

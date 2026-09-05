#!/usr/bin/env python3
"""Tier 1 — replay recorded calls through the live reducer and check the safety invariants.

    agent/.venv/bin/python eval/tier1_replay.py            # the whole corpus
    agent/.venv/bin/python eval/tier1_replay.py -v         # per-trace detail
    agent/.venv/bin/python eval/tier1_replay.py <path>...  # specific traces

Milliseconds, no API keys, no network, no audio. This is the tier that runs on every commit.

WHAT THIS IS NOT
----------------
It is NOT "assert the recorded action sequence still reproduces". That test rots on the first
legitimate fix: request ids derive from state counters, so a change that adds or removes a
request renumbers everything after it and every later delta reads as stale. CLAUDE.md records
exactly this (trace 20260904T173146047919Z replays to 2 utterances instead of 15 because the
nudge no longer fires on its first turn — correct behaviour, different trace).

So Tier 1 checks INVARIANTS instead. Each one is a property that must hold for any correct
reducer, is independent of request numbering, and was violated by a real defect on a real call:

    ungated_phi_tool      a PHI tool became a request before the caller was verified
    orphan_tool_use       a refused tool produced no result -> the turn never drains, silence
    thinking_spoken       the model's <thinking> block, hold UUID included, read to the caller
    unbacked_confirmation a confirmation number spoken that no tool ever returned
    nudge_discipline      the follow-through nudge fired on a turn it must not touch
    identity_pairing      patient_name and verified_dob describing two different people
    hold_id_retyped       confirm_booking carrying a hold_id the model typed rather than the
                          engine's
    nondeterminism        the same trace replayed twice produced different results

WHY THE CORPUS DOES NOT PROVE THE CHECKS
---------------------------------------
`eval/traces/` holds real recorded calls and they must all come back clean — that is the
regression signal. It is NOT evidence that the checks work: a check that always returns "no
violations" would pass every one of them. That evidence lives in `agent/tests/test_tier1.py`,
which feeds each invariant a hand-built event stream containing exactly the defect it exists to
catch and asserts it fires. An invariant suite that has never been seen to fail is decoration.

REPLAY DIVERGENCE IS EXPECTED, AND IT COSTS COVERAGE
----------------------------------------------------
Request ids derive from state counters, so a trace recorded by an older engine renumbers on
replay and its later `LLMToolUse`/`LLMTextDelta` events are dropped as stale. That is correct
behaviour, not a defect — but it means an old trace exercises only its opening. Each trace
therefore reports how many of its recorded tool calls the current reducer actually accepted,
and a trace that replays to nothing fails outright rather than passing vacuously.

All data here is synthetic (see CLAUDE.md). Phase 17 adds the tagged redaction boundary.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "agent" / "src"))

from clinic_agent.core import events as ev  # noqa: E402
from clinic_agent.core.actions import InvokeTool, Speak  # noqa: E402
from clinic_agent.core.reducer import _ACTION_NUDGE, reduce  # noqa: E402
from clinic_agent.core.recorder import load_trace  # noqa: E402
from clinic_agent.core.state import CallState  # noqa: E402
from clinic_agent.scheduling_tools import VERIFICATION_REQUIRED_TOOLS  # noqa: E402

CORPUS = Path(__file__).resolve().parent / "traces"

# Traces that are expected to fail a named invariant, and why. These are the recordings of the
# live defects; the wrong words are in the file and replaying them cannot remove the words.
# Keeping them in the corpus is what keeps the checks honest.
@dataclass(frozen=True)
class Violation:
    invariant: str
    detail: str
    seq: int | None = None

    def __str__(self) -> str:
        where = f" (seq {self.seq})" if self.seq is not None else ""
        return f"{self.invariant}{where}: {self.detail}"


@dataclass
class Step:
    """One reducer step, with the state on both sides so an invariant can see the transition."""

    event: ev.Event
    before: CallState
    after: CallState
    actions: list


def _steps(events) -> list[Step]:
    state = CallState()
    out: list[Step] = []
    for event in events:
        before = state
        state, actions = reduce(state, event)
        out.append(Step(event=event, before=before, after=state, actions=list(actions)))
    return out


# --- invariants -----------------------------------------------------------------------------
#
# Each takes the step list and returns the violations it found. Adding one is adding a function
# to INVARIANTS; nothing else changes.


def ungated_phi_tool(steps: list[Step]) -> list[Violation]:
    """No tool that reads or changes a patient's record becomes a request before verification.

    The gate lives in the reducer, not the prompt, precisely so this is checkable: an
    unverified request never becomes an `InvokeTool` action, so it never exists on the wire.
    """
    bad = []
    for step in steps:
        for action in step.actions:
            if isinstance(action, InvokeTool) and action.name in VERIFICATION_REQUIRED_TOOLS:
                if not step.before.identity_verified:
                    bad.append(Violation(
                        "ungated_phi_tool",
                        f"{action.name} invoked with identity_verified=False",
                        step.event.seq,
                    ))
    return bad


def orphan_tool_use(steps: list[Step]) -> list[Violation]:
    """Every tool the model asked for is either invoked or answered with a refusal result.

    A refused tool with no result means no ToolCompleted, which means the turn never drains and
    the call sits in silence for the rest of its life. Tool uses from a request that never
    completed (the caller hung up mid-turn) are exempt — there was no turn left to drain.
    """
    requested: dict[str, str] = {}      # tool_call_id -> name
    request_of: dict[str, str] = {}     # tool_call_id -> request_id
    completed_requests: set[str] = set()
    invoked: set[str] = set()
    answered: set[str] = set()

    for step in steps:
        e = step.event
        if isinstance(e, ev.LLMToolUse):
            # Only tool uses the reducer ACCEPTED. One belonging to a superseded request is
            # dropped by `_stale` and changes no state, so there is nothing to drain and
            # nothing to answer — counting it would flag replay divergence as a live defect.
            if len(step.after.turn_tool_uses) > len(step.before.turn_tool_uses):
                requested[e.tool_call_id] = e.name
                request_of[e.tool_call_id] = e.request_id
        elif isinstance(e, ev.LLMCompleted):
            completed_requests.add(e.request_id)
        for action in step.actions:
            if isinstance(action, InvokeTool):
                invoked.add(action.tool_call_id)
        for message in step.after.messages[len(step.before.messages):]:
            content = message.get("content")
            if isinstance(content, (list, tuple)):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        answered.add(block.get("tool_use_id", ""))

    return [
        Violation(
            "orphan_tool_use",
            f"{name} ({tid}) was neither invoked nor answered with a result",
        )
        for tid, name in requested.items()
        if tid not in invoked
        and tid not in answered
        and request_of.get(tid) in completed_requests
    ]


_THINKING = re.compile(r"</?thinking", re.IGNORECASE)
_UUID = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I)


def thinking_spoken(steps: list[Step]) -> list[Violation]:
    """Nothing the caller hears contains a thinking block or a hold UUID.

    Everything the reducer treats as text is spoken, so a leaked `<thinking>` is not a log
    artifact — it is the caller listening to the model's scratchpad, hold UUID and all.
    """
    bad = []
    for step in steps:
        for action in step.actions:
            if not isinstance(action, Speak):
                continue
            if _THINKING.search(action.text):
                bad.append(Violation("thinking_spoken", f"<thinking> in speech: {action.text[:80]!r}",
                                     step.event.seq))
            if _UUID.search(action.text):
                bad.append(Violation("thinking_spoken", f"hold UUID in speech: {action.text[:80]!r}",
                                     step.event.seq))
    return bad


# "your confirmation number is A754E2BC", "confirmation number 2096F951", "code is 88-56-60-88".
# The model reads these back with arbitrary punctuation, so the candidate is normalized to
# alphanumerics before it is compared against what the tools actually returned.
_SPOKEN_CODE = re.compile(
    r"confirmation\s+(?:number|code|id)\s*(?:is\s*)?[:\-]?\s*\*{0,2}([A-Z0-9][A-Z0-9\s\-]{5,20})",
    re.IGNORECASE,
)


def _normalize_code(raw: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", raw.upper())


def unbacked_confirmation(steps: list[Step]) -> list[Violation]:
    """A confirmation number spoken to the caller was returned by a tool that actually ran.

    The single worst failure this system can have: the caller hangs up believing they have an
    appointment. `had_tool_call: false` on every turn is the console tell; this is the
    machine-checkable version of it.
    """
    seen_codes: set[str] = set()
    bad = []
    for step in steps:
        e = step.event
        if isinstance(e, ev.ToolCompleted) and e.ok:
            for value in _walk_strings(e.result):
                code = _normalize_code(value)
                if len(code) >= 6:
                    seen_codes.add(code)
        for action in step.actions:
            if not isinstance(action, Speak):
                continue
            for match in _SPOKEN_CODE.finditer(action.text):
                code = _normalize_code(match.group(1))
                # Trailing prose gets swept into the capture ("is 88566088 Please arrive"), so
                # accept any prefix of the spoken run that a tool returned.
                if not code or any(code.startswith(k) or k.startswith(code) for k in seen_codes):
                    continue
                bad.append(Violation(
                    "unbacked_confirmation",
                    f"spoke confirmation {match.group(1).strip()!r}; no tool returned it",
                    step.event.seq,
                ))
    return bad


def _walk_strings(value):
    """Every string anywhere in a tool result, so a code is found wherever the API put it."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _walk_strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk_strings(item)


def nudge_discipline(steps: list[Step]) -> list[Violation]:
    """The follow-through nudge fires at most once per caller turn, and never on a turn that
    already ran a tool.

    Three live calls were truncated mid-sentence by a nudge that should not have fired, because
    starting the extra request closes the live TTS context. The cost of a false positive is not
    "one extra request" — it is the caller hearing "Good to hear from you, I'm happy to h—"
    followed by six seconds of silence on a confirmation read-back.
    """
    bad = []
    for step in steps:
        added = step.after.messages[len(step.before.messages):]
        for message in added:
            if message.get("content") != _ACTION_NUDGE:
                continue
            if step.before.turn_had_tool:
                bad.append(Violation("nudge_discipline", "nudged a turn whose tool already ran",
                                     step.event.seq))
            if step.before.nudged:
                bad.append(Violation("nudge_discipline", "nudged twice in one caller turn",
                                     step.event.seq))
    return bad


def identity_pairing(steps: list[Step]) -> list[Violation]:
    """`patient_name` and `verified_dob` always describe the same person.

    They are promoted together from the one verify_identity call the API accepted. Promoting
    them separately is how the engine came to report the caller as Nick while every PHI call
    carried Joe's date of birth — on a shared handset, the likeliest cross-person disclosure in
    the system.
    """
    bad = []
    for step in steps:
        after = step.after
        if after.patient_name and not after.verified_dob:
            bad.append(Violation(
                "identity_pairing",
                "patient_name set with no verified_dob — name and credential can diverge",
                step.event.seq,
            ))
        name_changed = after.patient_name != step.before.patient_name
        dob_changed = after.verified_dob != step.before.verified_dob
        if name_changed and after.patient_name and not dob_changed:
            bad.append(Violation(
                "identity_pairing",
                f"patient_name changed to {after.patient_name!r} without its date of birth",
                step.event.seq,
            ))
    return bad


def hold_id_retyped(steps: list[Step]) -> list[Violation]:
    """`confirm_booking` carries the engine's hold_id, never one the model retyped.

    A hold_id is a 36-character random UUID with no redundancy. Live, the model changed one hex
    digit of one and cost the call 30 seconds re-doing a hold that had already succeeded.
    """
    bad = []
    for step in steps:
        for action in step.actions:
            if not (isinstance(action, InvokeTool) and action.name == "confirm_booking"):
                continue
            live = step.before.active_hold_id
            sent = str(action.arguments.get("hold_id") or "")
            if live and sent != live:
                bad.append(Violation(
                    "hold_id_retyped",
                    f"confirm_booking sent {sent!r} while the engine held {live!r}",
                    step.event.seq,
                ))
    return bad


INVARIANTS = (
    ungated_phi_tool,
    orphan_tool_use,
    thinking_spoken,
    unbacked_confirmation,
    nudge_discipline,
    identity_pairing,
    hold_id_retyped,
)


def replay_coverage(steps: list[Step]) -> tuple[int, int]:
    """(accepted, recorded) tool calls — how much of an old trace the current reducer replays."""
    recorded = sum(1 for s in steps if isinstance(s.event, ev.LLMToolUse))
    accepted = sum(
        1 for s in steps
        if isinstance(s.event, ev.LLMToolUse)
        and len(s.after.turn_tool_uses) > len(s.before.turn_tool_uses)
    )
    return accepted, recorded


def check_events(events) -> list[Violation]:
    """Run every invariant over one event stream. The unit tests drive this directly."""
    steps = _steps(events)

    violations: list[Violation] = []
    for invariant in INVARIANTS:
        violations.extend(invariant(steps))

    # Determinism: the reducer is pure, so a second fold of the same events must produce the
    # same final state and the same actions. A clock read or a random id sneaking into reduce()
    # shows up here and nowhere else.
    again = _steps(events)
    if [s.actions for s in again] != [s.actions for s in steps]:
        violations.append(Violation("nondeterminism", "two replays produced different actions"))
    elif again[-1].after != steps[-1].after if steps else False:
        violations.append(Violation("nondeterminism", "two replays produced different final state"))

    return violations


def check_trace(path: Path) -> tuple[list[Violation], tuple[int, int]]:
    """Replay one recorded call: its violations, and how much of it the reducer still accepts."""
    events = load_trace(path)
    return check_events(events), replay_coverage(_steps(events))


def _report(path: Path, violations: list[Violation], coverage: tuple[int, int], *,
            verbose: bool) -> bool:
    """Print one trace's result. Returns True if the trace is clean and actually replayed."""
    accepted, recorded = coverage
    vacuous = recorded > 0 and accepted == 0

    ok = not violations and not vacuous
    note = f"  [{accepted}/{recorded} tool calls replayed]" if recorded else ""
    print(f"[{'PASS' if ok else 'FAIL'}] {path.stem}{note}")

    for v in violations:
        print(f"        \u2717 {v}")
    if vacuous:
        print("        \u2717 the reducer accepted none of this trace's tool calls — it replays to "
              "nothing and proves nothing. Re-record it against the current engine.")
    if verbose and accepted < recorded:
        print(f"        \u00b7 {recorded - accepted} tool calls dropped as stale (replay "
              "divergence, expected on a trace recorded by an older engine)")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("traces", nargs="*", type=Path, help="trace files (default: eval/traces)")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--json", type=Path, help="write the machine-readable result here")
    args = parser.parse_args()

    paths = args.traces or sorted(CORPUS.glob("*.jsonl"))
    if not paths:
        print(f"no traces found in {CORPUS}", file=sys.stderr)
        return 2

    results, failed = {}, 0
    for path in paths:
        violations, coverage = check_trace(path)
        results[path.stem] = {
            "violations": [
                {"invariant": v.invariant, "detail": v.detail, "seq": v.seq} for v in violations
            ],
            "tool_calls_replayed": coverage[0],
            "tool_calls_recorded": coverage[1],
        }
        if not _report(path, violations, coverage, verbose=args.verbose):
            failed += 1

    print(f"\ntier 1: {len(paths) - failed}/{len(paths)} traces clean, "
          f"{len(INVARIANTS) + 1} invariants each")
    if args.json:
        args.json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

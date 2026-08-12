"""Headless Phase-3 eval harness for the clinic voice agent.

Runs each scripted conversation in `test_cases.py` through the REAL Phase-2 dialogue brain —
`build_phase2_system_prompt()` + the scheduling-API tool schema + Anthropic/Claude + the real
`SchedulingClient` HTTP calls against the mock API — but WITHOUT the audio pipeline (no mic, no
ASR, no TTS). Simulated user utterances are fed as text straight into the LLM context, so the
suite runs fast and repeatably in CI while still exercising the actual booking/tool-calling
logic (same prompt, same tools, same API) rather than a re-implementation.

Test isolation (no changes to scheduling_api/):
  * The mock API is launched as a SUBPROCESS in its own uv env, pointed at a throwaway temp DB
    via CLINIC_DB_PATH.
  * The temp DB is reset (remove file + re-seed) BEFORE every case using scheduling_api's own
    stdlib-only db/seed modules, so a booking in one case can't cause a false 409 in the next.

Scoring is STRUCTURAL, not a transcript match: we inspect the tool-call trace (which tools ran,
with what args, and their results) and check it against each case's `Expected` outcome. See
test_cases.py for the outcome contract.

LLM provider: Anthropic/Claude (Haiku 4.5 by default) — the same active provider the live agent
uses. The eval drives Claude's Messages API directly (headless, no Pipecat): the Phase-2 system
prompt is passed as Anthropic's top-level `system` param, and the tool schema is converted with
the very same `AnthropicLLMAdapter.to_provider_tools_format()` the live pipeline uses, so the
eval and the agent stay on one tool definition. (Groq is a dormant fallback — see CLAUDE.md.)

Usage (from repo root, using the agent venv which has anthropic/httpx/dotenv):
    agent/.venv/bin/python eval/run_eval.py
    agent/.venv/bin/python eval/run_eval.py --only hp_flushot --only ad_pure_nonsense

Requires ANTHROPIC_API_KEY (read from agent/.env). This calls the real Anthropic API — not offline.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import httpx

# --- Repo paths & imports ----------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
AGENT_SRC = REPO_ROOT / "agent" / "src"
SCHEDULING_API_DIR = REPO_ROOT / "scheduling_api"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Make both service source trees importable from this (agent-venv) process. The agent package
# gives us the prompt + tool schema + HTTP client; scheduling_api's db/seed_data are stdlib-only
# and used ONLY to reset the temp DB between cases (its fastapi app runs in a subprocess).
sys.path.insert(0, str(AGENT_SRC))
sys.path.insert(0, str(SCHEDULING_API_DIR))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / "agent" / ".env")

from pipecat.adapters.services.anthropic_adapter import AnthropicLLMAdapter  # noqa: E402

from clinic_agent.prompts import build_phase2_system_prompt  # noqa: E402
from clinic_agent.scheduling_tools import SchedulingClient, build_tools_schema  # noqa: E402

import test_cases  # noqa: E402  (sibling module)
from providers import (  # noqa: E402  (sibling module)
    CANDIDATES,
    LLMTurn,
    OpenAICompatBackend,
    RunMetrics,
    build_backend,
)
from test_cases import ALL_CASES, EvalCase, Expected  # noqa: E402

# Per user turn, how many LLM<->tool round-trips we allow before giving up on that turn.
MAX_TOOL_STEPS = 8
# Retries for transient Anthropic API errors (rate limits / blips), with linear backoff.
API_RETRIES = 2
# Anthropic requires an explicit max_tokens. Replies + tool-call args are short.
MAX_TOKENS = 1024


# =========================================================================================
# Tool schema conversion (single source of truth = build_tools_schema())
# =========================================================================================
def anthropic_tools_from_schema() -> list[dict]:
    """Convert the Phase-2 pipecat FunctionSchemas into Anthropic tool dicts.

    Uses the exact adapter the live AnthropicLLMService uses, so the eval's tool definitions
    are byte-for-byte what the agent would send — if Phase 2's tools change, the eval follows.
    """
    return AnthropicLLMAdapter().to_provider_tools_format(build_tools_schema())


# =========================================================================================
# Mock scheduling API: subprocess lifecycle + per-case DB reset
# =========================================================================================
def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class MockApiServer:
    """Runs scheduling_api/ as a subprocess in its own uv env against a temp DB.

    Kept as a context manager so the process is always torn down. The temp DB path is exported
    via CLINIC_DB_PATH to BOTH the subprocess and this process (for resets).
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.port = _free_port()
        self.base_url = f"http://127.0.0.1:{self.port}"
        self._proc: subprocess.Popen | None = None

    def __enter__(self) -> "MockApiServer":
        env = {**os.environ, "CLINIC_DB_PATH": str(self.db_path)}
        # Own uv env for the API service (has fastapi/uvicorn); we only need HTTP from it.
        self._proc = subprocess.Popen(
            [
                "uv", "run", "uvicorn", "app.main:app",
                "--host", "127.0.0.1", "--port", str(self.port), "--log-level", "warning",
            ],
            cwd=str(SCHEDULING_API_DIR),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._wait_healthy()
        return self

    def _wait_healthy(self, timeout: float = 60.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._proc and self._proc.poll() is not None:
                raise RuntimeError("scheduling_api subprocess exited before becoming healthy")
            try:
                r = httpx.get(f"{self.base_url}/health", timeout=1.0)
                if r.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.4)
        raise RuntimeError(f"scheduling_api did not become healthy within {timeout}s")

    def __exit__(self, *exc) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()


def reset_db(db_path: Path) -> None:
    """Remove + re-seed the temp DB so every case starts from all-available slots.

    Reuses scheduling_api's own (stdlib-only) db module — the exact reset the API's test
    conftest uses. The subprocess opens a fresh connection per request, so replacing the file
    while it is idle between cases is safe.
    """
    os.environ["CLINIC_DB_PATH"] = str(db_path)
    # Import lazily and re-read the path: app.db caches DB_PATH at import time from the env.
    import importlib

    import app.db as api_db  # type: ignore

    importlib.reload(api_db)
    if db_path.exists():
        db_path.unlink()
    api_db.init_db()


def seeded_dates() -> set[str]:
    """The set of YYYY-MM-DD days the mock API seeds slots for (for date-constraint checks)."""
    import app.seed_data as seed  # type: ignore

    return {start[:10] for _, _, start, _ in seed.generate_slots()}


# =========================================================================================
# Conversation driver (headless LLM + tool loop)
# =========================================================================================
@dataclass
class Trace:
    """Everything observable from one scripted conversation, for scoring + the results file."""

    assistant_turns: list[str] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)  # {name, args, result}
    booking_result: dict | None = None  # successful confirm_booking response
    booking_args: dict | None = None     # args passed to that confirm_booking
    error: str | None = None


async def _execute_tool(name: str, args: dict, client: SchedulingClient, trace: Trace) -> dict:
    """Dispatch a tool call to the real SchedulingClient (same code path as the live agent)."""
    if name == "check_availability":
        result = await client.get_availability(
            date=args.get("date"), reason=args.get("reason_category"),
            provider_id=args.get("provider_id"),
        )
    elif name == "hold_slot":
        result = await client.hold_slot(slot_id=args.get("slot_id"))
    elif name == "confirm_booking":
        result = await client.confirm_booking(
            hold_id=args.get("hold_id"), patient_name=args.get("patient_name"),
            reason=args.get("reason"),
        )
    else:
        result = {"ok": False, "error": f"unknown tool {name!r}"}

    trace.tool_calls.append({"name": name, "args": args, "result": result})
    if name == "confirm_booking" and result.get("ok"):
        trace.booking_result = result
        trace.booking_args = args
    return result


async def _complete_with_retry(backend, system: str, history: list, tools) -> LLMTurn:
    """One model turn with retry/backoff on transient API errors.

    Provider-agnostic: `backend` owns the wire format, streaming, and usage accounting
    (see eval/providers.py). Phase 8 made this streaming so time-to-first-token is measurable —
    TTS cannot start until the first text token lands, so TTFT is the latency number that
    actually maps to how quickly a caller hears a voice.
    """
    last_exc: Exception | None = None
    for attempt in range(API_RETRIES + 1):
        try:
            return await backend.complete(system, history, tools)
        except Exception as exc:  # noqa: BLE001 - surface as a case error, not a crash
            last_exc = exc
            if attempt < API_RETRIES:
                await asyncio.sleep(1.5 * (attempt + 1))
    raise last_exc  # type: ignore[misc]


async def run_case(case: EvalCase, client: SchedulingClient, backend, tools,
                   system_prompt: str, *, metrics: RunMetrics | None = None) -> Trace:
    """Drive one scripted conversation end to end and return its trace.

    `metrics` accumulates latency / cache / cost for the Phase-8 bake-off; the plain `run_eval`
    path leaves it None and behaves exactly as before.
    """
    trace = Trace()
    # Provider-shaped conversation history. Both wire formats take a plain user message, but
    # assistant turns and tool results differ — the backend owns appending those.
    history: list = []

    try:
        for utterance in case.utterances:
            history.append({"role": "user", "content": utterance})
            # Resolve this user turn: keep looping while the model wants tools, until it
            # produces a plain-text reply (its spoken turn) or we hit the step cap.
            for _ in range(MAX_TOOL_STEPS):
                turn = await _complete_with_retry(backend, system_prompt, history, tools)
                if metrics is not None:
                    metrics.record(turn)
                backend.append_assistant(history, turn)

                if turn.tool_calls:
                    results = []
                    for call in turn.tool_calls:
                        payload = await _execute_tool(call.name, call.args, client, trace)
                        results.append((call, payload))
                    backend.append_tool_results(history, results)
                    continue  # feed tool results back to the model

                if turn.text:
                    trace.assistant_turns.append(turn.text)
                break
    except Exception as exc:  # noqa: BLE001
        trace.error = f"{type(exc).__name__}: {exc}"

    return trace


# =========================================================================================
# Scoring
# =========================================================================================
@dataclass
class CaseResult:
    case: EvalCase
    trace: Trace
    actual_outcome: str
    outcome_match: bool
    slot_filling_ok: bool | None   # None when the case wasn't expected to (or didn't) book
    passed: bool
    reasons: list[str]


def _derive_outcome(trace: Trace) -> str:
    """Classify what actually happened, independent of what was expected."""
    if trace.error:
        return "error"
    if trace.booking_result is not None:
        return "booked"
    saw_zero = any(
        tc["name"] == "check_availability" and tc["result"].get("ok")
        and tc["result"].get("count") == 0
        for tc in trace.tool_calls
    )
    if saw_zero:
        return "escalated"  # hit real no-availability
    # No booking and no zero-availability signal: treat as a handled/redirected conversation.
    return "gracefully_handled"


def score_case(case: EvalCase, trace: Trace, seed_dates: set[str]) -> CaseResult:
    exp = case.expected
    reasons: list[str] = []
    actual = _derive_outcome(trace)

    if trace.error:
        return CaseResult(case, trace, "error", False, None, False,
                          [f"conversation errored: {trace.error}"])

    slot_filling_ok: bool | None = None

    if exp.outcome == "booked":
        outcome_match = trace.booking_result is not None
        if not outcome_match:
            reasons.append("expected a booking but none was confirmed")
            slot_filling_ok = False
        else:
            slot_filling_ok = _check_slot_filling(exp, trace, seed_dates, reasons)

    elif exp.outcome == "escalated":
        booked = trace.booking_result is not None
        outcome_match = not booked
        if booked:
            reasons.append("expected escalation/no-booking but the agent booked an appointment")
        if exp.require_zero_availability:
            saw_zero = any(
                tc["name"] == "check_availability" and tc["result"].get("ok")
                and tc["result"].get("count") == 0
                for tc in trace.tool_calls
            )
            if not saw_zero:
                outcome_match = False
                reasons.append("expected a zero-availability result but never saw one")

    elif exp.outcome == "gracefully_handled":
        booked = trace.booking_result is not None
        outcome_match = not booked
        if booked:
            reasons.append("expected graceful handling but the agent booked an appointment")
        if exp.expect_followup_question:
            last = trace.assistant_turns[-1] if trace.assistant_turns else ""
            if not last.rstrip().endswith("?"):
                outcome_match = False
                reasons.append("expected a follow-up question but the final turn wasn't one")

    else:  # pragma: no cover - guarded by test_cases contract
        outcome_match = False
        reasons.append(f"unknown expected outcome {exp.outcome!r}")

    passed = outcome_match and (slot_filling_ok is not False)
    return CaseResult(case, trace, actual, outcome_match, slot_filling_ok, passed, reasons)


def _check_slot_filling(exp: Expected, trace: Trace, seed_dates: set[str],
                        reasons: list[str]) -> bool:
    """Did the booking capture name / reason / time correctly? (slot-filling accuracy)"""
    ok = True
    args = trace.booking_args or {}
    booked = trace.booking_result or {}

    name = (args.get("patient_name") or "").lower()
    if exp.name_contains and exp.name_contains.lower() not in name:
        ok = False
        reasons.append(f"name mismatch: booked as {args.get('patient_name')!r}, "
                       f"expected to contain {exp.name_contains!r}")

    reason = (args.get("reason") or "").lower()
    if exp.reason_any and not any(k.lower() in reason for k in exp.reason_any):
        ok = False
        reasons.append(f"reason mismatch: booked reason {args.get('reason')!r}, "
                       f"expected one of {exp.reason_any}")

    booked_date = (booked.get("start_time") or "")[:10]
    if exp.booked_on_date and booked_date != exp.booked_on_date:
        ok = False
        reasons.append(f"date mismatch: booked {booked_date}, expected {exp.booked_on_date}")
    if exp.booked_within_seed_window and booked_date not in seed_dates:
        ok = False
        reasons.append(f"date outside seed window: booked {booked_date}")

    # --- richer-intake fields (extended booking flow) -----------------------------------
    # dob_equals is an EXACT normalized-string check on purpose: it verifies the LLM turned
    # whatever spoken form the caller used into MM/DD/YYYY. Flakiness here is real signal.
    if exp.dob_equals is not None:
        dob = args.get("date_of_birth")
        if dob != exp.dob_equals:
            ok = False
            reasons.append(f"DOB mismatch: booked {dob!r}, expected exactly {exp.dob_equals!r}")

    if exp.new_patient_expected is not None:
        got_new = args.get("new_patient")
        if bool(got_new) != exp.new_patient_expected or got_new is None:
            ok = False
            reasons.append(f"new_patient mismatch: booked {got_new!r}, "
                           f"expected {exp.new_patient_expected}")

    symptom = (args.get("symptom_notes") or "")
    if exp.symptom_any and not any(k.lower() in symptom.lower() for k in exp.symptom_any):
        ok = False
        reasons.append(f"symptom mismatch: booked {symptom!r}, "
                       f"expected one of {exp.symptom_any}")
    if exp.symptom_max_words is not None:
        n_words = len(symptom.split())
        if n_words > exp.symptom_max_words:
            ok = False
            reasons.append(f"symptom too long ({n_words} words > {exp.symptom_max_words}): "
                           f"{symptom!r} — should be a brief note, not a medical narrative")

    return ok


# =========================================================================================
# Reporting
# =========================================================================================
def print_summary(results: list[CaseResult]) -> dict:
    total = len(results)
    passed = sum(r.passed for r in results)
    outcome_matches = sum(r.outcome_match for r in results)
    booked = [r for r in results if r.trace.booking_result is not None]
    slot_ok = sum(1 for r in booked if r.slot_filling_ok)
    # Infra errors (e.g. API 429/timeout) never exercised the dialogue — keep them OUT of the
    # dialogue-failure bucket so "Phase-4 targets" means genuine dialogue mismatches only.
    errored = [r for r in results if r.trace.error]
    scored_total = total - len(errored)

    print("\n" + "=" * 78)
    print("PHASE-3 EVAL RESULTS")
    print("=" * 78)

    by_cat: dict[str, list[CaseResult]] = {}
    for r in results:
        by_cat.setdefault(r.case.category, []).append(r)

    for cat in ("happy_path", "edge_case", "adversarial"):
        rs = by_cat.get(cat, [])
        if not rs:
            continue
        print(f"\n[{cat}]  {sum(x.passed for x in rs)}/{len(rs)} passed")
        for r in rs:
            tag = "PASS" if r.passed else "FAIL"
            sf = "" if r.slot_filling_ok is None else (
                " slot-fill:ok" if r.slot_filling_ok else " slot-fill:BAD")
            line = f"  [{tag}] {r.case.id:<26} exp={r.case.expected.outcome:<18} got={r.actual_outcome}{sf}"
            print(line)
            if not r.passed:
                for reason in r.reasons:
                    print(f"         └─ {reason}")

    print("\n" + "-" * 78)
    denom = scored_total or 1  # rates are over cases that actually ran (excl. infra errors)
    print(f"Cases scored: {scored_total}/{total}" +
          (f"  ({len(errored)} errored before scoring — see below)" if errored else ""))
    print(f"Task completion (outcome match): {outcome_matches}/{scored_total} "
          f"({100 * outcome_matches / denom:.0f}%)")
    print(f"Overall pass (outcome + slot-filling): {passed}/{scored_total} "
          f"({100 * passed / denom:.0f}%)")
    if booked:
        print(f"Slot-filling accuracy (booked cases): {slot_ok}/{len(booked)} "
              f"({100 * slot_ok / len(booked):.0f}%)")
    print("-" * 78)

    # Genuine dialogue failures (Phase-4 targets) — NOT infra errors.
    failures = [r for r in results if not r.passed and not r.trace.error]
    if failures:
        print("\nFAILURES TO REVIEW (Phase-4 targets):")
        for r in failures:
            print(f"  - {r.case.id} ({r.case.category}): {'; '.join(r.reasons)}")
    if errored:
        print("\nERRORED (infrastructure, not scored — re-run these):")
        for r in errored:
            # Collapse the noisy provider payload to a short reason.
            msg = r.trace.error or ""
            short = msg.split(" - {")[0] if " - {" in msg else msg[:120]
            print(f"  - {r.case.id} ({r.case.category}): {short}")
    print()

    return {
        "total": total,
        "scored_total": scored_total,
        "errored": len(errored),
        "passed": passed,
        "outcome_matches": outcome_matches,
        "booked_cases": len(booked),
        "slot_filling_ok": slot_ok,
    }


def write_results(results: list[CaseResult], summary: dict, model: str) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    payload = {
        "run_at": ts,
        "model": model,
        "summary": summary,
        "cases": [
            {
                "id": r.case.id,
                "category": r.case.category,
                "expected_outcome": r.case.expected.outcome,
                "actual_outcome": r.actual_outcome,
                "outcome_match": r.outcome_match,
                "slot_filling_ok": r.slot_filling_ok,
                "passed": r.passed,
                "reasons": r.reasons,
                "utterances": r.case.utterances,
                "assistant_turns": r.trace.assistant_turns,
                "tool_calls": r.trace.tool_calls,
                "error": r.trace.error,
            }
            for r in results
        ],
    }
    json_path = RESULTS_DIR / f"{ts}.json"
    json_path.write_text(json.dumps(payload, indent=2))

    # Stable-named markdown summary so results diff cleanly across runs in git.
    md = _render_markdown(results, summary, model, ts)
    (RESULTS_DIR / "latest.md").write_text(md)

    print(f"Wrote {json_path.relative_to(REPO_ROOT)} and "
          f"{(RESULTS_DIR / 'latest.md').relative_to(REPO_ROOT)}")


def _render_markdown(results: list[CaseResult], summary: dict, model: str, ts: str) -> str:
    scored = summary["scored_total"]
    lines = [
        "# Phase-3 Eval Results",
        "",
        f"- Run at: `{ts}`",
        f"- Model: `{model}`",
        f"- Cases scored: **{scored}/{summary['total']}**"
        + (f" ({summary['errored']} errored before scoring)" if summary["errored"] else ""),
        f"- Task completion (outcome match): **{summary['outcome_matches']}/{scored}**",
        f"- Overall pass (outcome + slot-filling): **{summary['passed']}/{scored}**",
    ]
    if summary["booked_cases"]:
        lines.append(
            f"- Slot-filling accuracy (booked cases): "
            f"**{summary['slot_filling_ok']}/{summary['booked_cases']}**"
        )
    lines += ["", "| Case | Category | Expected | Actual | Slot-fill | Pass |",
              "|------|----------|----------|--------|-----------|------|"]
    for r in results:
        sf = "-" if r.slot_filling_ok is None else ("ok" if r.slot_filling_ok else "BAD")
        lines.append(
            f"| {r.case.id} | {r.case.category} | {r.case.expected.outcome} | "
            f"{r.actual_outcome} | {sf} | {'✅' if r.passed else '❌'} |"
        )
    failures = [r for r in results if not r.passed and not r.trace.error]
    if failures:
        lines += ["", "## Dialogue failures (Phase-4 targets)", ""]
        for r in failures:
            lines.append(f"- **{r.case.id}** ({r.case.category}): {'; '.join(r.reasons)}")
    errored = [r for r in results if r.trace.error]
    if errored:
        lines += ["", "## Errored (infrastructure, not scored)", ""]
        for r in errored:
            msg = r.trace.error or ""
            short = msg.split(" - {")[0] if " - {" in msg else msg[:120]
            lines.append(f"- **{r.case.id}** ({r.case.category}): {short}")
    return "\n".join(lines) + "\n"


# =========================================================================================
# Entrypoint
# =========================================================================================
def tools_for_backend(backend) -> list[dict]:
    """Tool schemas in the backend's wire format, both derived from build_tools_schema().

    Keeping a single source of truth matters: if Phase 2's tools change, every provider in the
    bake-off follows automatically instead of silently drifting apart.
    """
    anthropic_tools = anthropic_tools_from_schema()
    if isinstance(backend, OpenAICompatBackend):
        return OpenAICompatBackend.tools_from_anthropic(anthropic_tools)
    return anthropic_tools


async def run_suite(cases: list[EvalCase], base_url: str, db_path: Path, backend, *,
                    metrics: RunMetrics | None = None) -> list[CaseResult]:
    """Run every case against one backend and score the traces.

    Cases run sequentially and each starts from a freshly re-seeded DB, so a hold or booking in
    one case can never leak a 409 into the next.
    """
    tools = tools_for_backend(backend)
    system_prompt = build_phase2_system_prompt()
    seed_dates = seeded_dates()

    results: list[CaseResult] = []
    for i, case in enumerate(cases, 1):
        reset_db(db_path)  # fresh, all-available slots per case (no 409 bleed-through)
        client = SchedulingClient(base_url)
        print(f"[{i}/{len(cases)}] running {case.id} ({case.category})...", flush=True)
        try:
            trace = await run_case(
                case, client, backend, tools, system_prompt, metrics=metrics
            )
        finally:
            await client.aclose()
        results.append(score_case(case, trace, seed_dates))

    return results


async def _run_all(cases: list[EvalCase], base_url: str, db_path: Path,
                   model: str) -> list[CaseResult]:
    """Single-model entrypoint used by `main()` (the pre-Phase-8 behaviour, now streaming)."""
    from providers import Candidate

    # An explicit --model / ANTHROPIC_MODEL may name something outside the candidate registry,
    # so fall back to a bare Anthropic candidate rather than requiring registration.
    candidate = next(
        (c for c in CANDIDATES.values() if c.provider == "anthropic" and c.model == model),
        Candidate(key="custom", label=model, provider="anthropic", model=model),
    )
    backend = build_backend(candidate, max_tokens=MAX_TOKENS)
    try:
        return await run_suite(cases, base_url, db_path, backend)
    finally:
        await backend.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase-3 headless eval harness")
    parser.add_argument("--only", action="append", default=[],
                        help="Run only the given case id(s); repeatable.")
    parser.add_argument("--model", default=None,
                        help="Override the Claude model (else ANTHROPIC_MODEL / "
                             "claude-haiku-4-5-20251001).")
    args = parser.parse_args()

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY is not set. Add it to agent/.env — this eval calls the "
              "real Anthropic API.", file=sys.stderr)
        return 2

    cases = [c for c in ALL_CASES if c.id in args.only] if args.only else ALL_CASES
    if not cases:
        print(f"No matching cases for --only {args.only}", file=sys.stderr)
        return 2

    model = args.model or os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
    db_path = Path(RESULTS_DIR).parent / ".eval_clinic.db"

    print(f"Starting mock scheduling API (temp DB: {db_path.name}); model={model}")
    with MockApiServer(db_path) as server:
        results = asyncio.run(_run_all(cases, server.base_url, db_path, model))

    summary = print_summary(results)
    write_results(results, summary, model)
    # Exit codes for CI: 0 = every scored case passed; 1 = a genuine dialogue failure;
    # 3 = one or more cases errored on infrastructure (e.g. API 429) and couldn't be scored.
    if summary["errored"]:
        return 3
    return 0 if summary["passed"] == summary["scored_total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

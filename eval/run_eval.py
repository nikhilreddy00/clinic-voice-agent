"""Headless Phase-3 eval harness for the clinic voice agent.

Runs each scripted conversation in `test_cases.py` through the REAL Phase-2 dialogue brain —
`build_system_prompt(SCHEDULE_APPOINTMENT)` + the tool schema + Anthropic/Claude + the real
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
import contextlib
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

from clinic_agent.intents import Intent  # noqa: E402
from clinic_agent.prompts import build_system_prompt  # noqa: E402
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

# Scratch Postgres database for eval runs. TRUNCATEd before every case, so point it at a
# throwaway database -- never at anything you care about.
EVAL_DATABASE_URL = os.getenv(
    "CLINIC_EVAL_DATABASE_URL", "postgresql://postgres@127.0.0.1:55432/clinic_eval"
)


def shard_database_url(index: int) -> str:
    """`clinic_eval`, then `clinic_eval_2`, `clinic_eval_3`, ... for the extra shards.

    Derived from the configured URL rather than configured separately, so pointing
    CLINIC_EVAL_DATABASE_URL somewhere else moves every shard with it.
    """
    if index == 0:
        return EVAL_DATABASE_URL
    base, _, name = EVAL_DATABASE_URL.rpartition("/")
    return f"{base}/{name}_{index + 1}"


def ensure_database(url: str) -> None:
    """Create a shard's database if it is missing. Idempotent, silent when it already exists.

    Without this, `--workers 4` fails on every machine where somebody once created `clinic_eval`
    by hand and nothing else — which is every machine.
    """
    import psycopg  # type: ignore

    base, _, name = url.rpartition("/")
    try:
        with psycopg.connect(url, connect_timeout=5) as conn:
            conn.execute("SELECT 1")
        return
    except psycopg.Error:
        pass
    with psycopg.connect(f"{base}/postgres", connect_timeout=5, autocommit=True) as conn:
        conn.execute(f'CREATE DATABASE "{name}"')


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
    """Runs scheduling_api/ as a subprocess in its own uv env against a throwaway database.

    Kept as a context manager so the process is always torn down. The database URL is exported
    via CLINIC_DATABASE_URL to BOTH the subprocess and this process (which resets between cases).

    Phase 9 moved storage from a SQLite file to Postgres, so isolation is now a dedicated
    DATABASE rather than a temp file. Point CLINIC_EVAL_DATABASE_URL at a scratch database — it
    is TRUNCATEd before every case.
    """

    def __init__(self, database_url: str) -> None:
        self.database_url = database_url
        self.port = _free_port()
        self.base_url = f"http://127.0.0.1:{self.port}"
        self._proc: subprocess.Popen | None = None

    def __enter__(self) -> "MockApiServer":
        env = {
            **os.environ,
            "CLINIC_DATABASE_URL": self.database_url,
            # The sweeper is housekeeping; a tight loop only adds noise during scoring.
            "CLINIC_SWEEP_INTERVAL_SECONDS": "3600",
        }
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


async def reset_db(database_url: str) -> None:
    """Truncate + re-seed the scratch database so every case starts from all-available slots.

    Reuses scheduling_api's own reset helper, so the table list stays in one place: a table added
    to schema.sql and forgotten here would survive the reset and leak state between cases.

    Uses a direct connection rather than the API's pool — that pool lives in the uvicorn
    subprocess, and the subprocess opens a connection per request, so resetting while it is idle
    between cases is safe.
    """
    import psycopg  # type: ignore
    from psycopg.rows import dict_row  # type: ignore

    import app.db as api_db  # type: ignore

    async with await psycopg.AsyncConnection.connect(
        database_url, row_factory=dict_row, autocommit=True
    ) as conn:
        await api_db.reset_and_seed(conn)


def seeded_dates() -> set[str]:
    """The set of YYYY-MM-DD days the API seeds slots for (for date-constraint checks).

    Phase 9: generate_slots() now returns tz-aware datetimes in the CLINIC's zone rather than
    ISO strings, so the day is taken from .date() instead of slicing. The old `start[:10]` also
    silently read a UTC day, which disagreed with the clinic day the agent reasons about.
    """
    import app.seed_data as seed  # type: ignore

    return {start.astimezone(seed.CLINIC_TZ).date().isoformat()
            for _, _, start, _ in seed.generate_slots()}


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
        # date_of_birth / new_patient / symptom_notes used to be DROPPED here while
        # `score_case` asserted on them from the model's own arguments. The suite therefore
        # reported "date of birth captured correctly" about a value the backend never saw: a
        # booking could score green with no date of birth in Postgres at all. Since Phase 13
        # the DOB is the credential a returning caller verifies with, so a booking that does
        # not carry one is a patient who cannot be recognised on their next call.
        result = await client.confirm_booking(
            hold_id=args.get("hold_id"), patient_name=args.get("patient_name"),
            reason=args.get("reason"),
            date_of_birth=args.get("date_of_birth"),
            new_patient=args.get("new_patient"),
            symptom_notes=args.get("symptom_notes"),
        )
    elif name == "get_clinic_info":
        # Phase 13 put a curated fact lookup in the scheduling tool set, so a case can ask
        # about hours or parking mid-booking. Dispatched here for the same reason as the rest:
        # the eval must exercise the tools the live agent is actually given, not a subset.
        result = await client.clinic_info(topic=args.get("topic"))
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


async def run_suite(cases: list[EvalCase], base_url: str, database_url: str, backend, *,
                    metrics: RunMetrics | None = None) -> list[CaseResult]:
    """Run every case against one backend and score the traces.

    Cases run sequentially and each starts from a freshly re-seeded DB, so a hold or booking in
    one case can never leak a 409 into the next.
    """
    tools = tools_for_backend(backend)
    # Checked, not assumed: build_system_prompt(SCHEDULE_APPOINTMENT) is byte-identical BOTH to
    # the old build_phase2_system_prompt() and to the intent=None prompt a caller gets on turn
    # one, and the unscoped build_tools_schema() is the same four tools a scheduling intent is
    # given. Every case in this suite is a scheduling flow, so it already measured exactly what
    # a booking caller meets — the "Phase-2" name was the only stale thing about it.
    #
    # What it does NOT exercise is intent SCOPING: no classifier runs here, so reschedule,
    # cancel and refill flows are not covered. That is eval/promptfoo/, which is per-intent.
    system_prompt = build_system_prompt(Intent.SCHEDULE_APPOINTMENT)
    seed_dates = seeded_dates()

    results: list[CaseResult] = []
    for i, case in enumerate(cases, 1):
        await reset_db(database_url)  # fresh slots per case (no 409 bleed-through)
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


async def run_suite_parallel(cases: list[EvalCase], shards: list[tuple[str, str]], backend, *,
                             metrics: RunMetrics | None = None) -> list[CaseResult]:
    """Run the suite across several (base_url, database_url) shards at once.

    ISOLATION MOVED; IT DID NOT DISAPPEAR. This suite was sequential because every case begins
    by truncating the database, so two cases sharing one database would truncate each other's
    slots mid-booking. A shard is therefore a whole stack — its own scratch database AND its own
    API process — and each shard still runs its slice one case at a time with a reset between.
    N cases against N databases is safe; N cases against one is not, and no amount of care in
    the driver changes that.

    Results come back in the original case order rather than in completion order, so the summary
    table and the results file do not reshuffle from run to run.
    """
    buckets: list[list[tuple[int, EvalCase]]] = [[] for _ in shards]
    for i, case in enumerate(cases):
        buckets[i % len(shards)].append((i, case))

    async def one_shard(shard, bucket):
        base_url, database_url = shard
        out = []
        for index, case in bucket:
            result = await run_suite([case], base_url, database_url, backend, metrics=metrics)
            out.append((index, result[0]))
        return out

    gathered = await asyncio.gather(*(
        one_shard(shard, bucket) for shard, bucket in zip(shards, buckets) if bucket
    ))
    return [r for _, r in sorted((pair for shard in gathered for pair in shard),
                                 key=lambda p: p[0])]


async def _run_all(cases: list[EvalCase], shards: list[tuple[str, str]],
                   model: str, *, backend=None) -> list[CaseResult]:
    """Single-model entrypoint used by `main()`."""
    from providers import Candidate

    owned = backend is None
    if owned:
        # An explicit --model / ANTHROPIC_MODEL may name something outside the candidate
        # registry, so fall back to a bare Anthropic candidate rather than requiring
        # registration.
        candidate = next(
            (c for c in CANDIDATES.values() if c.provider == "anthropic" and c.model == model),
            Candidate(key="custom", label=model, provider="anthropic", model=model),
        )
        backend = build_backend(candidate, max_tokens=MAX_TOKENS)
    try:
        if len(shards) == 1:
            base_url, database_url = shards[0]
            return await run_suite(cases, base_url, database_url, backend)
        return await run_suite_parallel(cases, shards, backend)
    finally:
        if owned:
            await backend.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Headless text-in-the-loop eval (Tier 2)")
    parser.add_argument("--only", action="append", default=[],
                        help="Run only the given case id(s); repeatable.")
    parser.add_argument("--model", default=None,
                        help="Override the Claude model (else ANTHROPIC_MODEL / "
                             "claude-haiku-4-5-20251001).")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel shards. Each gets its OWN scratch database and its own "
                             "API process — see run_suite_parallel for why sharing one is not "
                             "an option. Each shard costs an API subprocess start (~1s), so "
                             "this pays only when cases spend real time in a model.")
    parser.add_argument("--fake-backend", action="store_true",
                        help="Drive the harness with a scripted model instead of a real one. "
                             "No API key, no cost, no network to Anthropic — this tests the "
                             "HARNESS, not the model, and is what CI runs on every commit.")
    args = parser.parse_args()

    if not args.fake_backend and not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY is not set. Add it to agent/.env — this eval calls the "
              "real Anthropic API. (Or pass --fake-backend to exercise the harness for free.)",
              file=sys.stderr)
        return 2

    cases = [c for c in ALL_CASES if c.id in args.only] if args.only else ALL_CASES
    if not cases:
        print(f"No matching cases for --only {args.only}", file=sys.stderr)
        return 2

    model = args.model or os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
    backend = None
    if args.fake_backend:
        from fake_backend import ScriptedBackend

        backend, model = ScriptedBackend(), "scripted (no model was called)"
        if not args.only:
            # The script books unconditionally, so the five cases that must NOT end in a
            # booking (3 escalated, 2 gracefully_handled) fail by construction. Refusing to
            # book when refusing is right is a judgement a model makes; a script that only
            # LOOKED like it made it would be the fake proving something it cannot.
            skipped = [c for c in cases if c.expected.outcome != "booked"]
            cases = [c for c in cases if c.expected.outcome == "booked"]
            print(f"--fake-backend: running the {len(cases)} booking cases; skipping "
                  f"{len(skipped)} that must NOT book ({', '.join(c.id for c in skipped)}) — "
                  f"refusing is a model judgement, not a harness property.")

    workers = max(1, min(args.workers, len(cases)))
    started_on = test_cases.today()

    print(f"Starting {workers} scheduling API shard(s); model={model}")
    with contextlib.ExitStack() as stack:
        shards = []
        for i in range(workers):
            url = shard_database_url(i)
            ensure_database(url)
            server = stack.enter_context(MockApiServer(url))
            shards.append((server.base_url, url))
        results = asyncio.run(_run_all(cases, shards, model, backend=backend))

    # The cases compute their expected dates from the clock at import (test_cases.today()), and
    # the API seeds slots from the clock in its own process. Those agree until the clinic day
    # rolls over mid-run, and then a case that expected "tomorrow" fails for a reason no amount
    # of reading the transcript will reveal. Cheaper to say so than to make 500 lines of case
    # definitions take an injected date.
    if test_cases.today() != started_on:
        print(f"\nWARNING: the clinic day rolled over mid-run ({started_on} -> "
              f"{test_cases.today()}). Date-constraint failures below are the clock, not the "
              f"agent. Re-run.", file=sys.stderr)

    if args.fake_backend:
        # BEFORE the table, not after it. The per-category lines will read "0/6 passed" because
        # the script books every case with the same name and reason, and a reader who meets
        # that in a green CI log without the explanation above it learns to ignore the gate.
        print("\n--fake-backend scores OUTCOME only: did the harness reach a committed booking "
              "through the real client, real HTTP and real Postgres. The slot-filling column "
              "below is expected to be red — extracting a caller's name and reason from a "
              "sentence is the model's job, and there is no model here. A green run means the "
              "harness works, never that the agent does.")

    summary = print_summary(results)
    write_results(results, summary, model)
    # Exit codes for CI: 0 = every scored case passed; 1 = a genuine dialogue failure;
    # 3 = one or more cases errored on infrastructure (e.g. API 429) and couldn't be scored.
    if summary["errored"]:
        return 3
    if args.fake_backend:
        # SLOT-FILLING IS NOT SCORED HERE, and pretending otherwise is the only way this mode
        # could mislead. The script books every case with the same name and reason because
        # extracting "Sam" and "flu shot" from a sentence is precisely the work a model does —
        # so it fails the per-case value checks by construction. What it proves is that the
        # harness reaches a committed booking through the real client, real HTTP and real
        # Postgres, and that the scorer and the exit code agree about it.
        return 0 if summary["outcome_matches"] == summary["scored_total"] else 1
    return 0 if summary["passed"] == summary["scored_total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Phase-8 model bake-off: run the eval suite across candidate LLMs and compare them.

WHAT THIS ANSWERS
-----------------
Which model should the voice agent run on? The live agent's dominant cost is LLM latency
(README.md:105-110 — LLM p50 2925 ms inside an E2E p50 of 4098 ms), so the decision is a
four-way trade-off that a pass/fail eval alone cannot settle:

    quality (does it still book correctly?)
      x  latency (TTFT — when does the caller hear a voice?)
      x  cost per call
      x  cache viability (does prompt caching actually engage on this model?)

Reusing `run_eval.py`'s cases and scoring is deliberate: the 19-case suite is the regression net
for the whole project, so the bake-off measures the same behaviour the project already gates on
rather than inventing a parallel notion of "correct".

WHY CACHE VIABILITY IS A COLUMN
-------------------------------
Anthropic silently declines to cache a prefix below the model's minimum — the request succeeds,
`cache_control` is accepted, and nothing is cached. This project's prompt + tool schemas measure
~3,814 tokens against Haiku 4.5's 4,096 minimum, so caching is a no-op on the incumbent model
and would have stayed invisible without measuring it. The minimum varies by model and is not
monotonic across generations, so it belongs in the comparison, not in a footnote.

COST
----
Runs hit the real APIs. The full suite is 19 cases x several turns per case, so a four-model
sweep is a few dollars, not cents. Use `--only` to smoke-test the wiring on one case first.

USAGE
-----
    # from agent/ so the venv has anthropic + openai + pipecat
    uv run python ../eval/bakeoff.py --only hp_checkup_tomorrow          # cheap smoke test
    uv run python ../eval/bakeoff.py                                     # default 3 models
    uv run python ../eval/bakeoff.py --models haiku-4-5,sonnet-4-6 --cache
    uv run python ../eval/bakeoff.py --models haiku-4-5 --cache          # prove the silent miss
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from providers import (  # noqa: E402
    CANDIDATES,
    DEFAULT_BAKEOFF,
    Candidate,
    RunMetrics,
    build_backend,
    measure_prefix_tokens,
)
from run_eval import (  # noqa: E402
    ALL_CASES,
    MAX_TOKENS,
    RESULTS_DIR,
    CaseResult,
    EvalCase,
    MockApiServer,
    anthropic_tools_from_schema,
    build_phase2_system_prompt,
    run_suite,
)


@dataclass
class ModelRun:
    """One candidate's complete result set."""

    candidate: Candidate
    results: list[CaseResult]
    metrics: RunMetrics
    skipped: str | None = None  # populated instead of results when the model couldn't run

    @property
    def scored(self) -> int:
        return sum(1 for r in self.results if not r.trace.error)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def errored(self) -> int:
        return sum(1 for r in self.results if r.trace.error)

    @property
    def outcome_matches(self) -> int:
        return sum(1 for r in self.results if r.outcome_match)

    @property
    def cost_per_call(self) -> float | None:
        """USD per *conversation*, which is the unit a clinic actually pays per phone call."""
        if not self.metrics.priced or not self.results:
            return None
        return self.metrics.cost_usd / len(self.results)


def _fmt_ms(value: float | None) -> str:
    return "—" if value is None else f"{value:,.0f}"


def _fmt_usd(value: float | None) -> str:
    return "—" if value is None else f"${value:.4f}"


async def _run_one(candidate: Candidate, cases: list[EvalCase], base_url: str, db_path: Path,
                   use_cache: bool) -> ModelRun:
    """Run the full suite against one candidate, isolating its failures from the sweep."""
    metrics = RunMetrics.for_candidate(candidate, use_cache=use_cache)

    # Size the cacheable prefix (tools + system) in this model's own tokenizer before running.
    # Measured up front so a silent cache miss can be explained ("N tokens below the minimum")
    # rather than merely reported.
    metrics.cache.prefix_tokens = await measure_prefix_tokens(
        candidate, build_phase2_system_prompt(), anthropic_tools_from_schema()
    )
    if metrics.cache.prefix_tokens is not None:
        floor = candidate.cache_min_tokens
        verdict = "" if floor is None else (
            "  (cacheable)" if metrics.cache.prefix_tokens >= floor
            else f"  (BELOW the {floor:,} minimum — caching cannot engage)"
        )
        print(f"  cacheable prefix: {metrics.cache.prefix_tokens:,} tokens{verdict}", flush=True)

    try:
        backend = build_backend(candidate, max_tokens=MAX_TOKENS, use_cache=use_cache)
    except Exception as exc:  # noqa: BLE001 — a missing API key shouldn't abort the sweep
        print(f"  SKIPPED {candidate.label}: {exc}\n", flush=True)
        return ModelRun(candidate, [], metrics, skipped=str(exc))

    try:
        results = await run_suite(cases, base_url, db_path, backend, metrics=metrics)
    finally:
        await backend.close()

    return ModelRun(candidate, results, metrics)


def print_comparison(runs: list[ModelRun]) -> None:
    ran = [r for r in runs if r.skipped is None]

    print("\n" + "=" * 100)
    print("PHASE-8 MODEL BAKE-OFF")
    print("=" * 100)

    header = f"{'Model':<28} {'Pass':>7} {'TTFT p50':>9} {'TTFT p95':>9} {'Call p50':>9} {'$/call':>10}  Cache"
    print(header)
    print("-" * 100)
    for run in ran:
        lat = run.metrics.latency.summary()
        pass_str = f"{run.passed}/{len(run.results)}"
        print(
            f"{run.candidate.label:<28} {pass_str:>7} "
            f"{_fmt_ms(lat['ttft_p50']):>9} {_fmt_ms(lat['ttft_p95']):>9} "
            f"{_fmt_ms(lat['total_p50']):>9} {_fmt_usd(run.cost_per_call):>10}  "
            f"{run.metrics.cache.verdict()}"
        )
    for run in (r for r in runs if r.skipped is not None):
        print(f"{run.candidate.label:<28} {'—':>7}  SKIPPED: {run.skipped}")
    print("-" * 100)
    print("TTFT = time to first token (ms). TTS cannot start before this, so it is the number")
    print("that maps to how quickly a caller hears a voice. 'Call p50' is the full LLM call.")

    # Per-model detail: which cases regressed matters as much as the headline number, because a
    # model that is fast and cheap but drops a booking is disqualified regardless of latency.
    for run in ran:
        failures = [r for r in run.results if not r.passed and not r.trace.error]
        errored = [r for r in run.results if r.trace.error]
        if not failures and not errored:
            continue
        print(f"\n  {run.candidate.label} — regressions:")
        for r in failures:
            print(f"    [FAIL] {r.case.id:<32} {'; '.join(r.reasons)}")
        for r in errored:
            print(f"    [ERR ] {r.case.id:<32} {(r.trace.error or '')[:100]}")

    if ran:
        print("\nNotes:")
        for run in ran:
            if run.candidate.notes:
                print(f"  - {run.candidate.label}: {run.candidate.notes}")


def write_results(runs: list[ModelRun], cases: list[EvalCase], use_cache: bool) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "run_at": ts,
        "cases": len(cases),
        "prompt_caching_requested": use_cache,
        "models": [
            {
                "key": run.candidate.key,
                "label": run.candidate.label,
                "model": run.candidate.model,
                "provider": run.candidate.provider,
                "skipped": run.skipped,
                "passed": run.passed,
                "scored": run.scored,
                "errored": run.errored,
                "outcome_matches": run.outcome_matches,
                "latency_ms": run.metrics.latency.summary(),
                "cost_usd_total": round(run.metrics.cost_usd, 6) if run.metrics.priced else None,
                "cost_usd_per_call": (
                    round(run.cost_per_call, 6) if run.cost_per_call is not None else None
                ),
                "prompt_tokens_total": run.metrics.input_tokens,
                "output_tokens_total": run.metrics.output_tokens,
                "cache": {
                    "requested": run.metrics.cache.requested,
                    "engaged": run.metrics.cache.engaged,
                    "read_tokens": run.metrics.cache.total_read,
                    "write_tokens": run.metrics.cache.total_write,
                    "max_prompt_tokens": run.metrics.cache.max_prompt_tokens,
                    "cacheable_prefix_tokens": run.metrics.cache.prefix_tokens,
                    "model_minimum": run.metrics.cache.cache_min_tokens,
                    "verdict": run.metrics.cache.verdict(),
                },
                "failures": [
                    {"case": r.case.id, "reasons": r.reasons}
                    for r in run.results
                    if not r.passed and not r.trace.error
                ],
            }
            for run in runs
        ],
    }
    out = RESULTS_DIR / f"bakeoff-{ts}.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase-8 model bake-off")
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_BAKEOFF),
        help=f"Comma-separated candidate keys. Available: {', '.join(CANDIDATES)}",
    )
    parser.add_argument("--only", action="append", default=[],
                        help="Run only the given case id(s); repeatable. Use for cheap smoke runs.")
    parser.add_argument("--cache", action="store_true",
                        help="Enable Anthropic prompt caching (adds a cache_control breakpoint on "
                             "the system block, which also covers the tool schemas).")
    parser.add_argument("--list", action="store_true", help="List candidates and exit.")
    args = parser.parse_args()

    if args.list:
        for key, c in CANDIDATES.items():
            cache = f"cache min {c.cache_min_tokens:,}" if c.cache_min_tokens else "no cache data"
            print(f"  {key:<16} {c.label:<28} {c.model:<28} {cache}")
        return 0

    unknown = [k for k in args.models.split(",") if k.strip() not in CANDIDATES]
    if unknown:
        print(f"ERROR: unknown candidate(s) {unknown}. Known: {list(CANDIDATES)}", file=sys.stderr)
        return 2
    candidates = [CANDIDATES[k.strip()] for k in args.models.split(",") if k.strip()]

    if not os.getenv("ANTHROPIC_API_KEY") and any(c.provider == "anthropic" for c in candidates):
        print("ERROR: ANTHROPIC_API_KEY is not set (agent/.env). This calls the real API.",
              file=sys.stderr)
        return 2

    cases = [c for c in ALL_CASES if c.id in args.only] if args.only else ALL_CASES
    if not cases:
        print(f"No matching cases for --only {args.only}", file=sys.stderr)
        return 2

    db_path = RESULTS_DIR.parent / ".bakeoff_clinic.db"
    print(
        f"Bake-off: {len(candidates)} model(s) x {len(cases)} case(s)"
        f"{' with prompt caching' if args.cache else ''}"
    )

    runs: list[ModelRun] = []
    # One API subprocess for the whole sweep; run_suite re-seeds the DB before every case, so
    # models never see each other's bookings.
    with MockApiServer(db_path) as server:
        for candidate in candidates:
            print(f"\n--- {candidate.label} ({candidate.model}) ---", flush=True)
            runs.append(
                asyncio.run(_run_one(candidate, cases, server.base_url, db_path, args.cache))
            )

    print_comparison(runs)
    out = write_results(runs, cases, args.cache)
    print(f"\nWrote {out.relative_to(RESULTS_DIR.parent.parent)}")

    # Exit non-zero if every model was skipped — that means the sweep measured nothing.
    return 0 if any(r.skipped is None for r in runs) else 1


if __name__ == "__main__":
    raise SystemExit(main())

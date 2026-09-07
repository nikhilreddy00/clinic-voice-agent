"""Tier 3 — does the agent still get the job done when it cannot hear the caller cleanly.

    agent/.venv/bin/python -m tier3_audio --preview            # show the damage, call nothing
    agent/.venv/bin/python -m tier3_audio                      # offline sweep, real model
    agent/.venv/bin/python -m tier3_audio --fake-backend       # offline sweep, no model, free
    agent/.venv/bin/python -m tier3_audio --live               # real audio. BILLS. Never run.

    (run from eval/, or with eval/ on the path)

THE THREE TIERS, and why this one is not the one that runs on every commit:

    Tier 1  eval/tier1_replay.py   recorded calls through the reducer   ms, free, every commit
    Tier 2  eval/run_eval.py       scripted text through a real model   $, on a label
    Tier 3  here                   degraded speech through the stack    $$, by hand

BUILT, NOT RUN — and specifically, `--live` has never been executed. Live calls bill Cartesia,
Deepgram and LiveKit and the author is conserving credit (CLAUDE.md), so the same discipline
Tier B load got applies here: the harness exists and is tested, the spend is a decision.

WHAT THE OFFLINE SWEEP ACTUALLY MEASURES. It replays the Tier-2 cases with each utterance run
through `channel.degrade` — the corruption a bad line produces, applied to the transcript — and
scores the result with Tier 2's own scorer plus the Tier-1 invariants. So it measures how the
DIALOGUE survives ASR error, through the real prompt, the real tools and real Postgres.

What it does NOT measure is Deepgram's error rate on an accent, in noise, or over a codec. Only
`--live` can, and `channel.py` says why running the audio path against a fake STT would be an
empty test rather than a cheap one.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EVAL_DIR))

import run_eval  # noqa: E402
from test_cases import ALL_CASES  # noqa: E402
from tier3_audio.channel import CONDITIONS, LIVE_VOICES, degrade  # noqa: E402


def degraded_cases(condition, *, seed: int, booking_only: bool = False):
    """Every Tier-2 case with its utterances put through the channel.

    The EXPECTATIONS are untouched. That is the point of the tier: a caller whose speech was
    mangled still wants the same appointment, and an agent that books the wrong thing because
    it misheard "fifteen" as "fifty" has failed even though the mishearing was not its fault.
    """
    cases = ALL_CASES
    if booking_only:
        # Same reason as Tier 2's --fake-backend: the scripted model books unconditionally, so
        # the cases that must NOT end in a booking fail by construction. Refusing when refusing
        # is right is a model judgement, not a harness property.
        cases = [c for c in cases if c.expected.outcome == "booked"]

    out = []
    for i, case in enumerate(cases):
        damaged = tuple(
            degrade(u, condition, seed=seed * 1000 + i * 10 + j)[0]
            for j, u in enumerate(case.utterances)
        )
        out.append(replace(case, utterances=damaged))
    return out


def preview(conditions: list[str], seed: int) -> int:
    """Print what each condition does to a few real utterances. No model, no API, no cost."""
    samples = [
        "I'd like to book a flu shot for next Tuesday please",
        "My date of birth is December fifteenth nineteen ninety",
        "I can't make Thursday any more, can we move it",
        "Yeah that works, the two thirty one is fine",
    ]
    for name in conditions:
        condition = CONDITIONS[name]
        print(f"\n=== {name} — {condition.note}")
        for i, text in enumerate(samples):
            damaged, confidence = degrade(text, condition, seed=seed + i)
            mark = "  (unchanged)" if damaged == text else ""
            print(f"  said : {text}")
            print(f"  heard: {damaged}   [conf {confidence:.2f}]{mark}")
    return 0


async def sweep(conditions: list[str], *, seed: int, workers: int, backend, model: str,
                booking_only: bool = False) -> dict:
    """Run the case suite once per condition and collect the pass rates."""
    report: dict[str, dict] = {}

    for name in conditions:
        condition = CONDITIONS[name]
        cases = degraded_cases(condition, seed=seed, booking_only=booking_only)
        print(f"\n=== condition {name} ({len(cases)} cases) — {condition.note}")

        with __import__("contextlib").ExitStack() as stack:
            shards = []
            for i in range(workers):
                url = run_eval.shard_database_url(i)
                run_eval.ensure_database(url)
                server = stack.enter_context(run_eval.MockApiServer(url))
                shards.append((server.base_url, url))
            results = await run_eval._run_all(cases, shards, model, backend=backend)

        booked = sum(1 for r in results if r.actual_outcome == "booked")
        matched = sum(1 for r in results if r.actual_outcome == r.case.expected.outcome)
        errored = [r for r in results if r.trace.error]
        report[name] = {
            "cases": len(results),
            "outcome_match": matched,
            "booked": booked,
            "errored": len(errored),
            "failures": [
                {"id": r.case.id, "expected": r.case.expected.outcome,
                 "got": r.actual_outcome, "reasons": r.reasons}
                for r in results if r.actual_outcome != r.case.expected.outcome
            ],
        }
        print(f"  outcome match: {matched}/{len(results)}"
              + (f"   ({len(errored)} errored)" if errored else ""))

    return report


def print_table(report: dict) -> None:
    print("\n" + "=" * 70)
    print(f"{'condition':<12} {'outcome match':>14} {'booked':>8} {'errored':>8}")
    for name, row in report.items():
        print(f"{name:<12} {row['outcome_match']:>10}/{row['cases']:<3} "
              f"{row['booked']:>8} {row['errored']:>8}")
    print("=" * 70)

    clean = report.get("clean")
    if clean:
        print("\nRead the DROP from clean, not the absolute numbers. A case that fails on the"
              "\nclean channel is a dialogue defect that Tier 2 should already have caught;"
              "\nwhat this tier is for is the gap between clean and degraded.")


def live_unavailable() -> int:
    """The audio path: what it would do, what it would cost, and why it has not been run.

    Deliberately not a stub that pretends. The pieces it needs all exist —
    `core/adapters/tts.CartesiaTTS` can synthesise the caller's side, `core/adapters/stt` is the
    real Deepgram socket, and `CallSession._build_*` is the seam the load test already uses to
    swap adapters — so this is an integration nobody has paid for yet, not a design gap.
    """
    print(
        "--live is not implemented, and this message is the honest version of that.\n"
        "\n"
        "What it would do: synthesise each caller utterance with Cartesia across "
        f"{len(LIVE_VOICES)} voices ({', '.join(LIVE_VOICES)}), mix noise to each condition's\n"
        "SNR, feed the PCM into a real CallSession through the media adapter, and let the real\n"
        "Deepgram socket transcribe it — then score with this file's scorer.\n"
        "\n"
        "What it would cost: 19 cases x 4 voices x 4 conditions = 304 conversations of TTS and\n"
        "STT, plus the model turns Tier 2 already pays for. Cartesia and Deepgram bill per\n"
        "second of audio in both directions.\n"
        "\n"
        "Why it is not built: CLAUDE.md's credit-conservation rule, and the same judgement Tier\n"
        "B load got — the harness and the seam exist, the spend is a decision. Writing the\n"
        "integration now would mean several hundred lines that have never once been executed,\n"
        "and this repo has already been bitten by exactly that (loadtest/fake_adapters.FakeLLM\n"
        "drifted from the real signature and reported success for 800 dead sessions).\n"
        "\n"
        "The offline sweep runs without it: drop --live.",
        file=sys.stderr,
    )
    return 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--condition", action="append", default=[],
                        choices=sorted(CONDITIONS), help="repeatable; default is all of them")
    parser.add_argument("--seed", type=int, default=1,
                        help="channel seed. The same seed reproduces the same damage exactly.")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--preview", action="store_true",
                        help="print what each condition does to a few utterances and stop")
    parser.add_argument("--fake-backend", action="store_true",
                        help="scripted model — exercises this harness for free (see "
                             "eval/fake_backend.py for what that does and does not prove)")
    parser.add_argument("--live", action="store_true",
                        help="real synthesised audio through the real STT. BILLS. Not built.")
    parser.add_argument("--out", type=Path, default=EVAL_DIR / "results")
    args = parser.parse_args()

    conditions = args.condition or list(CONDITIONS)

    if args.live:
        return live_unavailable()
    if args.preview:
        return preview(conditions, args.seed)

    backend, model = None, run_eval.os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
    if args.fake_backend:
        from fake_backend import ScriptedBackend

        backend, model = ScriptedBackend(), "scripted (no model was called)"
        print(
            "--fake-backend: THE CHANNEL IS INERT IN THIS MODE. The scripted model does not "
            "read the caller's words at all — it follows the booking contract from the tool "
            "results — so every condition produces the same run. That is the point: this mode "
            "proves the SWEEP works (degradation, sharding, scoring, the report), and it can "
            "prove nothing about robustness, because nothing here is listening.\n"
            "Booking cases only, for the same reason Tier 2's fake mode skips the rest."
        )
    elif not run_eval.os.getenv("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set. Use --preview (free) or --fake-backend (free), or "
              "add the key to agent/.env.", file=sys.stderr)
        return 2

    report = asyncio.run(sweep(conditions, seed=args.seed, workers=max(1, args.workers),
                               backend=backend, model=model,
                               booking_only=args.fake_backend))
    print_table(report)

    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = args.out / f"tier3-{stamp}.json"
    path.write_text(json.dumps({"seed": args.seed, "model": model, "conditions": report},
                               indent=2), encoding="utf-8")
    print(f"\nresults: {path}")

    # A degraded channel is EXPECTED to cost some cases; failing the run on that would make the
    # tier unrunnable, and a tier that always fails stops being run. What is not acceptable is a
    # regression on the CLEAN channel, which is Tier 2 with extra steps and must still pass.
    clean = report.get("clean")
    if clean and clean["outcome_match"] < clean["cases"]:
        print("\nFAIL: the clean channel regressed — that is a dialogue defect, not an audio one.")
        return 1
    if args.fake_backend and len({r["outcome_match"] for r in report.values()}) > 1:
        # The scripted model ignores the transcript, so every condition MUST come out the same.
        # A difference means the sweep is leaking state between conditions — shared database,
        # shared shard, reused backend — which would silently corrupt a real run's numbers.
        print("\nFAIL: conditions differ under a model that cannot hear them — the sweep is "
              "leaking state between conditions.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

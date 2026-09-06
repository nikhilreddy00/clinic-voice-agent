"""Phase 12 — intent confusion matrix and emergency recall.

Two exit criteria, measured rather than asserted:

    intent accuracy    >= 95%
    emergency recall   == 100%   (non-negotiable)

The second is measured **twice**, against two different mechanisms, and they are not
interchangeable:

* The **deterministic detector** (``core/intent.detect_emergency``) is the actual control. It
  is a pure function, runs before any model exists, and is what the reducer calls. Its recall
  is checked here and gated at 100% in ``agent/tests/test_emergency.py`` on every commit.
* The **classifier** is asked the same questions as a redundancy. A model that disagrees is
  worth knowing about, but it is not what protects the caller — and it must never become the
  thing that does, because a provider outage would then remove the safety control.

    uv run python eval/run_intent_eval.py            # classifier + detector (needs a key)
    uv run python eval/run_intent_eval.py --detector-only   # no API calls, no cost
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "agent" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import anthropic  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from clinic_agent.core.intent import (  # noqa: E402
    CLASSIFIER_SYSTEM_PROMPT,
    CLASSIFIER_TOOL,
    classifier_messages,
    detect_emergency,
)
from intent_cases import CASES, IntentCase  # noqa: E402

load_dotenv(REPO / "agent" / ".env")

MODEL = os.getenv("CLINIC_MODEL_FAST") or os.getenv(
    "ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"
)
CONCURRENCY = 8


async def classify(client: anthropic.AsyncAnthropic, case: IntentCase) -> tuple[str, float, float]:
    """Return (predicted_intent, confidence, latency_ms) for one utterance."""
    t0 = time.monotonic()
    response = await client.messages.create(
        model=MODEL,
        max_tokens=128,
        system=CLASSIFIER_SYSTEM_PROMPT,
        tools=[CLASSIFIER_TOOL],
        tool_choice={"type": "tool", "name": CLASSIFIER_TOOL["name"]},
        messages=classifier_messages(case.utterance),
    )
    latency_ms = (time.monotonic() - t0) * 1000
    for block in response.content:
        if getattr(block, "type", None) == "tool_use":
            args = dict(block.input or {})
            return str(args.get("intent", "unknown")), float(args.get("confidence", 0.0)), latency_ms
    return "unknown", 0.0, latency_ms


def detector_report() -> dict:
    """Emergency recall and false-positive rate for the deterministic detector."""
    emergencies = [c for c in CASES if c.expected == "emergency"]
    others = [c for c in CASES if c.expected != "emergency"]

    missed = [c.utterance for c in emergencies if detect_emergency(c.utterance) is None]
    false_positives = [
        (c.utterance, detect_emergency(c.utterance).category)
        for c in others
        if detect_emergency(c.utterance) is not None
    ]
    recall = (len(emergencies) - len(missed)) / len(emergencies) if emergencies else 1.0
    return {
        "emergency_cases": len(emergencies),
        "recall": recall,
        "missed": missed,
        "false_positives": false_positives,
        "false_positive_rate": len(false_positives) / len(others) if others else 0.0,
    }


def print_matrix(pairs: list[tuple[str, str]]) -> None:
    """Confusion matrix, rows = expected, columns = predicted."""
    labels = sorted({e for e, _ in pairs} | {p for _, p in pairs})
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for expected, predicted in pairs:
        counts[(expected, predicted)] += 1

    width = max(len(x) for x in labels) + 2
    short = {label: label[:6] for label in labels}
    header = " " * width + "".join(f"{short[label]:>8}" for label in labels)
    print("\nConfusion matrix (rows = actual, cols = predicted)")
    print(header)
    for expected in labels:
        row = f"{expected:<{width}}"
        for predicted in labels:
            n = counts[(expected, predicted)]
            cell = str(n) if n else "."
            row += f"{cell:>8}"
        print(row)


async def main() -> int:
    parser = argparse.ArgumentParser(description="Phase-12 intent + emergency eval")
    parser.add_argument(
        "--detector-only",
        action="store_true",
        help="skip the model; measure only the deterministic emergency detector (free)",
    )
    parser.add_argument("--out", default=str(Path(__file__).parent / "results"))
    args = parser.parse_args()

    print(f"Phase-12 intent eval — {len(CASES)} labeled utterances")

    detector = detector_report()
    print("\n--- deterministic emergency detector (the actual safety control) ---")
    print(f"  recall              : {detector['recall']:.1%}  ({detector['emergency_cases']} cases)")
    print(f"  false positives     : {len(detector['false_positives'])} "
          f"({detector['false_positive_rate']:.1%} of non-emergency utterances)")
    for utterance in detector["missed"]:
        print(f"  !! MISSED           : {utterance!r}")
    for utterance, category in detector["false_positives"]:
        print(f"  ~  false positive   : {utterance!r} -> {category}")

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cases": len(CASES),
        "detector": detector,
    }

    if not args.detector_only:
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            print("\nANTHROPIC_API_KEY not set — skipping the classifier half.")
            return 1

        client = anthropic.AsyncAnthropic(api_key=api_key)
        semaphore = asyncio.Semaphore(CONCURRENCY)

        async def run(case: IntentCase):
            async with semaphore:
                return case, *await classify(client, case)

        print(f"\n--- classifier ({MODEL}) ---")
        results = await asyncio.gather(*(run(c) for c in CASES))
        await client.close()

        pairs = [(case.expected, predicted) for case, predicted, _, _ in results]
        correct = sum(1 for e, p in pairs if e == p)
        accuracy = correct / len(pairs)
        latencies = sorted(latency for *_, latency in results)
        p50 = latencies[len(latencies) // 2]
        p95 = latencies[max(0, int(len(latencies) * 0.95) - 1)]

        emergency_pairs = [(e, p) for e, p in pairs if e == "emergency"]
        classifier_recall = (
            sum(1 for e, p in emergency_pairs if p == "emergency") / len(emergency_pairs)
            if emergency_pairs
            else 1.0
        )

        print(f"  accuracy            : {accuracy:.1%}  ({correct}/{len(pairs)})")
        print(f"  emergency recall    : {classifier_recall:.1%}  (redundancy, not the control)")
        print(f"  latency             : p50 {p50:.0f} ms · p95 {p95:.0f} ms")

        misses = [
            (case.utterance, case.expected, predicted, confidence)
            for case, predicted, confidence, _ in results
            if case.expected != predicted
        ]
        if misses:
            print(f"\n  {len(misses)} misclassified:")
            for utterance, expected, predicted, confidence in misses:
                print(f"    {expected:>24} -> {predicted:<24} ({confidence:.2f})  {utterance!r}")

        print_matrix(pairs)

        payload["classifier"] = {
            "model": MODEL,
            "accuracy": accuracy,
            "correct": correct,
            "emergency_recall": classifier_recall,
            "latency_ms": {"p50": round(p50), "p95": round(p95)},
            "misclassified": [
                {"utterance": u, "expected": e, "predicted": p, "confidence": c}
                for u, e, p, c in misses
            ],
        }

        print("\n--- exit criteria ---")
        print(f"  intent accuracy >= 95%   : {'PASS' if accuracy >= 0.95 else 'FAIL'} ({accuracy:.1%})")
        print(f"  detector recall == 100%  : "
              f"{'PASS' if detector['recall'] == 1.0 else 'FAIL'} ({detector['recall']:.1%})")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"intent-eval-{stamp}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nresults: {path}")

    # FALSE POSITIVES ARE GATED TOO, at zero, and that is not perfectionism.
    #
    # Recall was always a build-breaker: a missed emergency is the worst outcome this system
    # has. But a false positive is not merely noise here — `Intent.EMERGENCY` strips every tool
    # and swaps the prompt for the 911 script, so a caller with a stubbed toe gets read
    # emergency instructions by an agent that has lost the ability to book them anything. A live
    # call did exactly that with a knee laceration (see intents.CLASSIFIER_ONLY_ADVISORY), and
    # the detector is the half that must not repeat it.
    #
    # Zero is the measured state across the whole labelled set, so this gates the status quo
    # rather than an aspiration. A newly added case that trips it is a decision someone has to
    # make deliberately — which is the point.
    detector_clean = detector["recall"] == 1.0 and not detector["false_positives"]
    if not args.detector_only:
        print(f"  detector false positives == 0 : "
              f"{'PASS' if not detector['false_positives'] else 'FAIL'} "
              f"({len(detector['false_positives'])})")
    ok = detector_clean and (
        args.detector_only or payload.get("classifier", {}).get("accuracy", 0) >= 0.95
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

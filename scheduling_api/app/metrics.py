"""Aggregate per-call metrics into the dashboard payload.

Phase 6 wrote this against `logs/calls.jsonl`, a file the agent appended to and this service
re-read IN FULL on every request. Phase 16 retired that bus — the agent POSTs a batch per call
to `/call-metrics` and the rows live in Postgres — but the aggregation itself did not need to
change, so it did not. `aggregate_metrics` is now a PURE function over the same event shape,
and `db.fetch_call_events` produces that shape from SQL. Only the transport moved.

Keeping it pure is what makes it testable without a database, which is why the events are
passed in rather than fetched here.

No events yields a well-formed zeroed aggregate (never an error), so the dashboard renders
cleanly before the first call.
"""

from __future__ import annotations

import math

STAGES = ("asr", "llm", "tts", "e2e")


def _percentiles(values: list[float]) -> dict:
    """Nearest-rank P50/P95/P99 (whole ms). Matches the agent-side collector."""
    if not values:
        return {"p50": None, "p95": None, "p99": None, "count": 0}
    ordered = sorted(values)

    def rank(p: float) -> int:
        return max(0, math.ceil(p / 100 * len(ordered)) - 1)

    return {
        "p50": round(ordered[rank(50)]),
        "p95": round(ordered[rank(95)]),
        "p99": round(ordered[rank(99)]),
        "count": len(ordered),
    }


def _empty_metrics() -> dict:
    return {
        "turns_total": 0,
        "calls_total": 0,
        "latency_ms": {s: _percentiles([]) for s in STAGES},
        "asr_confidence": {"avg": None, "min": None, "max": None, "count": 0},
        "tool_calls": {
            "total": 0,
            "success": 0,
            "success_rate": None,
            "by_endpoint": {},
            "latency_ms": _percentiles([]),
        },
        "outcomes": {"booked": 0, "escalated": 0, "abandoned": 0},
        "recent_calls": [],
    }


def aggregate_metrics(events: list[dict]) -> dict:
    """Aggregate call-metric events into the dashboard payload (shape: see _empty_metrics)."""
    if not events:
        return _empty_metrics()

    stage_samples: dict[str, list[float]] = {s: [] for s in STAGES}
    confidences: list[float] = []
    tool_latencies: list[float] = []
    tool_total = tool_success = 0
    by_endpoint: dict[str, dict] = {}
    outcomes = {"booked": 0, "escalated": 0, "abandoned": 0}
    summaries: list[dict] = []
    turns_total = 0

    for ev in events:
        kind = ev.get("event")
        if kind == "turn":
            turns_total += 1
            for stage in STAGES:
                value = ev.get(f"{stage}_ms")
                if value is not None:
                    stage_samples[stage].append(value)
            conf = ev.get("asr_confidence")
            if conf is not None:
                confidences.append(conf)
        elif kind == "tool":
            tool_total += 1
            success = bool(ev.get("success"))
            tool_success += int(success)
            endpoint = ev.get("endpoint", "unknown")
            lat = ev.get("latency_ms")
            if lat is not None:
                tool_latencies.append(lat)
            bucket = by_endpoint.setdefault(endpoint, {"total": 0, "success": 0, "_lat": []})
            bucket["total"] += 1
            bucket["success"] += int(success)
            if lat is not None:
                bucket["_lat"].append(lat)
        elif kind == "call_summary":
            outcome = ev.get("outcome", "abandoned")
            if outcome in outcomes:
                outcomes[outcome] += 1
            summaries.append(ev)

    # Finalize per-endpoint buckets (turn latency list -> p50, drop the private accumulator).
    for bucket in by_endpoint.values():
        lat = bucket.pop("_lat")
        bucket["p50_ms"] = _percentiles(lat)["p50"]

    recent_calls = [
        {
            "call_id": s.get("call_id"),
            "ts": s.get("ts"),
            "mode": s.get("mode"),
            "outcome": s.get("outcome"),
            "turns": s.get("turns"),
            "e2e_p50": (s.get("latency_ms", {}).get("e2e", {}) or {}).get("p50"),
        }
        for s in sorted(summaries, key=lambda s: s.get("ts", ""), reverse=True)[:5]
    ]

    return {
        "turns_total": turns_total,
        "calls_total": len(summaries),
        "latency_ms": {s: _percentiles(stage_samples[s]) for s in STAGES},
        "asr_confidence": {
            "avg": round(sum(confidences) / len(confidences), 3) if confidences else None,
            "min": round(min(confidences), 3) if confidences else None,
            "max": round(max(confidences), 3) if confidences else None,
            "count": len(confidences),
        },
        "tool_calls": {
            "total": tool_total,
            "success": tool_success,
            "success_rate": round(tool_success / tool_total, 3) if tool_total else None,
            "by_endpoint": by_endpoint,
            "latency_ms": _percentiles(tool_latencies),
        },
        "outcomes": outcomes,
        "recent_calls": recent_calls,
    }

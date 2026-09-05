"""Unit tests for the /metrics aggregation.

Phase 16 moved the source from `logs/calls.jsonl` to Postgres, and `aggregate_metrics` became a
pure function over the event shape rather than a file reader. These tests came with it almost
unchanged, which is the point: the aggregation was not what was wrong with the old design, the
transport was.

The torn-line test is gone with the file it described — a JSON line cannot be half-written when
there is no line. Its replacement is the round-trip test in test_call_metrics.py, which asserts
the rows the API stores aggregate back to the same numbers.
"""

from __future__ import annotations

from app.metrics import aggregate_metrics


def test_empty_is_a_well_formed_zero_not_an_error():
    """The dashboard has to render before the first call ever lands."""
    agg = aggregate_metrics([])
    assert agg["calls_total"] == 0 and agg["turns_total"] == 0
    assert agg["latency_ms"]["e2e"]["p50"] is None
    assert agg["recent_calls"] == []


def test_aggregates_turns_tools_and_outcomes():
    events = [
        {"event": "turn", "call_id": "C1", "asr_ms": 200, "llm_ms": 400, "tts_ms": 300,
         "e2e_ms": 900, "asr_confidence": 0.9, "had_tool_call": False},
        {"event": "turn", "call_id": "C1", "asr_ms": 100, "llm_ms": 200, "tts_ms": None,
         "e2e_ms": 500, "asr_confidence": None, "had_tool_call": True},
        {"event": "tool", "call_id": "C1", "endpoint": "/availability", "http_status": 200,
         "latency_ms": 30, "success": True},
        {"event": "tool", "call_id": "C1", "endpoint": "/hold-slot", "http_status": 409,
         "latency_ms": 12, "success": False},
        {"event": "call_summary", "call_id": "C1", "ts": "2026-07-07T10:00:00+00:00",
         "mode": "telephony", "outcome": "booked", "turns": 2,
         "latency_ms": {"e2e": {"p50": 700}}},
    ]
    agg = aggregate_metrics(events)

    assert agg["turns_total"] == 2
    assert agg["calls_total"] == 1
    # tts had one null -> only one sample counted; asr counts both
    assert agg["latency_ms"]["asr"]["count"] == 2
    assert agg["latency_ms"]["tts"]["count"] == 1
    # confidence: only the non-null 0.9 is aggregated
    assert agg["asr_confidence"]["count"] == 1 and agg["asr_confidence"]["avg"] == 0.9
    # tools: 2 total, 1 success -> 0.5
    assert agg["tool_calls"]["total"] == 2 and agg["tool_calls"]["success"] == 1
    assert agg["tool_calls"]["success_rate"] == 0.5
    assert agg["tool_calls"]["by_endpoint"]["/availability"]["p50_ms"] == 30
    assert agg["outcomes"]["booked"] == 1
    assert agg["recent_calls"][0]["e2e_p50"] == 700


def test_percentiles_come_from_raw_turns_not_per_call_medians():
    """Percentiles of per-call percentiles are not percentiles.

    This is why the per-turn rows are stored individually rather than as a summary per call. Two
    calls, one fast and one slow: the p95 across all six turns is the slow call's tail, and it
    would vanish if each call contributed only its own median.
    """
    events = [{"event": "turn", "call_id": "A", "e2e_ms": ms} for ms in (100, 110, 120)]
    events += [{"event": "turn", "call_id": "B", "e2e_ms": ms} for ms in (900, 1000, 4000)]
    agg = aggregate_metrics(events)

    assert agg["latency_ms"]["e2e"]["count"] == 6
    assert agg["latency_ms"]["e2e"]["p95"] == 4000

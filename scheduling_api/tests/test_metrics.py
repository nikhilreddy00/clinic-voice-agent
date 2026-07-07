"""Unit tests for the /metrics aggregation over calls.jsonl."""

from __future__ import annotations

import importlib
import json


def _aggregate_with_log(tmp_path, monkeypatch, lines):
    (tmp_path / "calls.jsonl").write_text("\n".join(json.dumps(o) for o in lines) + "\n")
    monkeypatch.setenv("CLINIC_LOG_DIR", str(tmp_path))
    import app.metrics as m

    importlib.reload(m)  # re-read resolve_log_path() under the patched env
    return m.aggregate_metrics()


def test_empty_when_no_log(tmp_path, monkeypatch):
    monkeypatch.setenv("CLINIC_LOG_DIR", str(tmp_path / "nope"))
    import app.metrics as m

    importlib.reload(m)
    agg = m.aggregate_metrics()
    assert agg["calls_total"] == 0 and agg["turns_total"] == 0
    assert agg["latency_ms"]["e2e"]["p50"] is None
    assert agg["recent_calls"] == []


def test_aggregates_turns_tools_and_outcomes(tmp_path, monkeypatch):
    lines = [
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
    agg = _aggregate_with_log(tmp_path, monkeypatch, lines)

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


def test_skips_torn_json_line(tmp_path, monkeypatch):
    (tmp_path / "calls.jsonl").write_text(
        '{"event": "turn", "e2e_ms": 500}\n{"event": "turn", "e2e_ms":\n'  # 2nd line truncated
    )
    monkeypatch.setenv("CLINIC_LOG_DIR", str(tmp_path))
    import app.metrics as m

    importlib.reload(m)
    assert m.aggregate_metrics()["turns_total"] == 1  # torn line skipped, not fatal

"""Offline unit tests for the Phase-8 bake-off machinery (no API calls, safe for CI).

The bake-off itself costs real money to run, so the logic that *interprets* its measurements is
tested here instead. The cache-verdict logic especially: it is what turned "add prompt caching"
into "caching cannot engage on this model", and a regression there would quietly hand back a
wrong architectural conclusion.

Run with:  cd agent && uv run pytest ../eval
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from providers import (  # noqa: E402
    CANDIDATES,
    CacheReport,
    Candidate,
    LatencyStats,
    LLMTurn,
    OpenAICompatBackend,
    RunMetrics,
    turn_cost_usd,
)


def _turn(**kw) -> LLMTurn:
    base = dict(
        text="ok",
        tool_calls=[],
        stop_reason="end_turn",
        raw_assistant_message=[],
        ttft_ms=100.0,
        total_ms=200.0,
    )
    base.update(kw)
    return LLMTurn(**base)  # type: ignore[arg-type]


# =========================================================================================
# prompt_tokens: the "input_tokens is only the uncached remainder" trap
# =========================================================================================
def test_prompt_tokens_sums_uncached_plus_cache_traffic():
    """Reading input_tokens as prompt size under-reports by whatever the cache served."""
    turn = _turn(input_tokens=200, cache_read_tokens=3_000, cache_write_tokens=600)

    assert turn.prompt_tokens == 3_800
    assert turn.input_tokens == 200, "input_tokens must stay the uncached remainder"


# =========================================================================================
# Cache verdicts — the finding this whole phase turns on
# =========================================================================================
def test_verdict_reports_engaged_when_cache_reads_occur():
    report = CacheReport(requested=True, cache_min_tokens=1024, prefix_tokens=3_804)
    report.record(_turn(input_tokens=10, cache_read_tokens=3_800))

    assert report.engaged
    assert "ENGAGED" in report.verdict()


def test_verdict_explains_a_silent_miss_by_how_far_below_the_minimum():
    """The Haiku 4.5 case: request succeeds, nothing caches, no error anywhere."""
    report = CacheReport(requested=True, cache_min_tokens=4096, prefix_tokens=3_811)
    report.record(_turn(input_tokens=3_811))  # no cache read, no cache write

    assert not report.engaged
    verdict = report.verdict()
    assert "SILENT MISS" in verdict
    assert "3,811" in verdict
    assert "285 below" in verdict, "must quantify the shortfall, not just report a miss"


def test_verdict_distinguishes_a_changing_prefix_from_a_too_short_one():
    """Above the minimum but still not caching is a different bug with a different fix."""
    report = CacheReport(requested=True, cache_min_tokens=1024, prefix_tokens=3_804)
    report.record(_turn(input_tokens=3_804))

    verdict = report.verdict()
    assert "SILENT MISS" in verdict
    assert "changes per call" in verdict


def test_verdict_uses_the_prefix_not_the_growing_conversation():
    """Regression guard for the bug this logic originally had.

    The cache breakpoint sits on the trailing system block, so conversation turns land *after*
    it and can never help the prefix clear the minimum. Measuring total prompt size instead made
    a too-short prefix look like it was above the limit once the conversation grew, and produced
    the wrong diagnosis.
    """
    report = CacheReport(requested=True, cache_min_tokens=4096, prefix_tokens=3_811)
    # A long conversation pushes total prompt tokens far past the minimum...
    report.record(_turn(input_tokens=12_000))

    assert report.max_prompt_tokens == 12_000
    # ...but the verdict must still be driven by the 3,811-token prefix.
    assert "285 below" in report.verdict()


def test_verdict_is_not_applicable_without_cache_counters():
    report = CacheReport(requested=False, applicable=False)
    assert "n/a" in report.verdict()


def test_verdict_reports_not_requested_when_caching_is_off():
    report = CacheReport(requested=False, cache_min_tokens=1024)
    assert report.verdict() == "not requested"


# =========================================================================================
# Latency
# =========================================================================================
def test_percentiles_use_nearest_rank_matching_the_live_agent():
    stats = LatencyStats()
    for ms in (100, 200, 300, 400, 500):
        stats.record(_turn(ttft_ms=float(ms), total_ms=float(ms * 2)))

    summary = stats.summary()
    assert summary["calls"] == 5
    assert summary["ttft_p50"] == 300
    assert summary["ttft_p95"] == 500
    assert summary["total_p50"] == 600


def test_turns_without_a_ttft_sample_are_excluded_not_zeroed():
    """A failed stream has no first token; counting it as 0 ms would flatter the p50."""
    stats = LatencyStats()
    stats.record(_turn(ttft_ms=None, total_ms=500.0))
    stats.record(_turn(ttft_ms=300.0, total_ms=600.0))

    summary = stats.summary()
    assert summary["ttft_p50"] == 300
    assert summary["calls"] == 2, "total_ms is still sampled for both"


def test_empty_stats_report_none_rather_than_crashing():
    assert LatencyStats().summary()["ttft_p50"] is None


# =========================================================================================
# Cost
# =========================================================================================
def test_cost_applies_the_cache_read_and_write_multipliers():
    candidate = CANDIDATES["sonnet-4-6"]  # $3.00 in / $15.00 out per MTok
    turn = _turn(
        input_tokens=1_000_000,
        cache_read_tokens=1_000_000,
        cache_write_tokens=1_000_000,
        output_tokens=1_000_000,
    )

    # 3.00 (full) + 0.30 (read @0.1x) + 3.75 (write @1.25x) + 15.00 (output)
    assert turn_cost_usd(candidate, turn) == 22.05


def test_unpriced_candidates_return_none_instead_of_a_fabricated_zero():
    """Groq/Cerebras ship without list prices here — report '—', never a misleading $0.0000."""
    candidate = CANDIDATES["groq-llama"]
    assert candidate.input_cost_per_mtok is None
    assert turn_cost_usd(candidate, _turn(input_tokens=1000, output_tokens=1000)) is None


def test_run_metrics_marks_itself_unpriced_when_any_turn_is_unpriced():
    metrics = RunMetrics.for_candidate(CANDIDATES["groq-llama"], use_cache=False)
    metrics.record(_turn(input_tokens=100, output_tokens=50))

    assert metrics.priced is False
    assert metrics.cost_usd == 0.0


def test_run_metrics_accumulates_tokens_latency_and_cost_together():
    metrics = RunMetrics.for_candidate(CANDIDATES["haiku-4-5"], use_cache=True)
    metrics.record(_turn(input_tokens=1000, output_tokens=100, ttft_ms=50.0, total_ms=120.0))
    metrics.record(_turn(input_tokens=2000, output_tokens=200, ttft_ms=70.0, total_ms=150.0))

    assert metrics.input_tokens == 3000
    assert metrics.output_tokens == 300
    assert metrics.latency.summary()["calls"] == 2
    assert metrics.priced and metrics.cost_usd > 0


def test_cache_report_is_marked_inapplicable_for_non_anthropic_providers():
    metrics = RunMetrics.for_candidate(CANDIDATES["groq-llama"], use_cache=True)
    assert metrics.cache.applicable is False
    assert metrics.cache.requested is False, "caching can't be 'requested' where it doesn't exist"


# =========================================================================================
# Tool schema translation (single source of truth = build_tools_schema())
# =========================================================================================
def test_anthropic_tools_convert_to_openai_function_tools():
    anthropic_tools = [
        {
            "name": "check_availability",
            "description": "Find open slots",
            "input_schema": {
                "type": "object",
                "properties": {"date": {"type": "string"}},
                "required": [],
            },
        }
    ]

    converted = OpenAICompatBackend.tools_from_anthropic(anthropic_tools)

    assert converted == [
        {
            "type": "function",
            "function": {
                "name": "check_availability",
                "description": "Find open slots",
                # The JSON Schema carries over untouched — that is what keeps one tool
                # definition true for both providers.
                "parameters": anthropic_tools[0]["input_schema"],
            },
        }
    ]


def test_tool_conversion_covers_every_real_tool():
    """Guards against a tool being added to Phase 2 and silently missing on the OpenAI path."""
    from run_eval import anthropic_tools_from_schema

    anthropic_tools = anthropic_tools_from_schema()
    converted = OpenAICompatBackend.tools_from_anthropic(anthropic_tools)

    assert len(converted) == len(anthropic_tools) >= 3
    assert {c["function"]["name"] for c in converted} == {t["name"] for t in anthropic_tools}


# =========================================================================================
# Candidate registry: per-model request quirks that 400 or wreck latency if applied uniformly
# =========================================================================================
def test_models_that_reject_sampling_params_are_flagged():
    """temperature is a 400 on Opus 5 / Sonnet 5 / Opus 4.8 / 4.7, but fine on Haiku 4.5."""
    assert CANDIDATES["haiku-4-5"].supports_temperature is True
    assert CANDIDATES["sonnet-5"].supports_temperature is False
    assert CANDIDATES["opus-4-8"].supports_temperature is False


def test_models_with_thinking_on_by_default_disable_it_explicitly():
    """Thinking tokens land on TTFT — the one thing a voice agent cannot afford."""
    assert CANDIDATES["sonnet-5"].thinking_config == {"type": "disabled"}
    # Sonnet 4.6 runs without thinking when the parameter is omitted, so it needs no override.
    assert CANDIDATES["sonnet-4-6"].thinking_config is None


def test_cache_minimums_match_the_documented_per_model_values():
    """These are not monotonic across generations, which is exactly why they're pinned here."""
    assert CANDIDATES["haiku-4-5"].cache_min_tokens == 4096
    assert CANDIDATES["sonnet-4-6"].cache_min_tokens == 1024
    assert CANDIDATES["sonnet-5"].cache_min_tokens == 1024


def test_every_candidate_is_routable_to_a_backend():
    for key, candidate in CANDIDATES.items():
        assert candidate.provider in {"anthropic", "openai_compat"}, key
        if candidate.provider == "openai_compat":
            assert candidate.base_url, f"{key} needs a base_url"


def test_custom_candidate_defaults_are_safe_for_an_unregistered_model():
    """run_eval falls back to a bare Candidate for a --model outside the registry."""
    c = Candidate(key="custom", label="x", provider="anthropic", model="claude-haiku-4-5")

    assert c.supports_temperature is True
    assert c.thinking_config is None
    assert c.cache_min_tokens is None  # unknown minimum -> verdict says "undetermined"

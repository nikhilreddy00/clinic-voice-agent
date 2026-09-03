"""Prompt caching: the breakpoint, the default, and the floor that makes it real.

Anthropic accepts a `cache_control` breakpoint on a prefix that is too short and then caches
nothing — no error, no warning, `cache_creation_input_tokens: 0`. That silence is why caching
was off here for two phases while looking enabled, and it is why these tests assert on the
prompt SIZE rather than only on the request shape. A passing "breakpoint is present" test
would have stayed green through the entire period the optimization did nothing.

No network: token counts are approximated by character length, calibrated once against
`messages.count_tokens`. The assertion is about headroom, not an exact count.
"""

from __future__ import annotations

import pytest

from clinic_agent.core.adapters.llm import AnthropicLLM, to_anthropic_tools
from clinic_agent.intents import Intent
from clinic_agent.prompts import build_system_prompt
from clinic_agent.scheduling_tools import build_tools_schema

# Haiku 4.5's minimum cacheable prefix. Not monotonic across the family (512 on Opus 5, 1,024
# on Sonnet 5/4.6 and Opus 4.8) — see core.llm_router.ModelSpec.cache_min_tokens.
HAIKU_CACHE_FLOOR = 4096

# Measured 2026-09-03: the booking prefix counted 5,023 tokens at 20,289 characters, i.e.
# ~4.04 chars/token. Rounded down to 4.0 so this estimate is conservative (it under-reports
# tokens, so a test that says "clears the floor" is not doing so on a rounding artifact).
CHARS_PER_TOKEN = 4.0


def _prefix_tokens(intent: Intent | None) -> int:
    """Approximate the cacheable prefix: system prompt + tool schemas."""
    import json

    system = build_system_prompt(intent, None)
    tools = to_anthropic_tools(build_tools_schema(intent))
    chars = len(system) + len(json.dumps(tools))
    return int(chars / CHARS_PER_TOKEN)


def _llm(monkeypatch, value: str | None) -> AnthropicLLM:
    if value is None:
        monkeypatch.delenv("CLINIC_PROMPT_CACHE", raising=False)
    else:
        monkeypatch.setenv("CLINIC_PROMPT_CACHE", value)
    return AnthropicLLM(
        api_key="k",
        model="claude-haiku-4-5",
        system_prompt="fallback",
        tools=build_tools_schema(Intent.SCHEDULE_APPOINTMENT),
        emit=lambda e: None,
    )


# --- the floor ------------------------------------------------------------------------------


@pytest.mark.parametrize("intent", [None, Intent.SCHEDULE_APPOINTMENT])
def test_the_booking_prefix_clears_haikus_cache_floor(intent):
    """If this fails, caching is silently off on the calls that matter and nothing says so.

    Phase 13's six new tools are what pushed this over: the prefix was 3,811 tokens in Phase 8
    and is 5,023 now. The headroom is real but not large, so trimming the booking prompt is a
    change that costs money without touching a line of cost-related code.
    """
    assert _prefix_tokens(intent) >= HAIKU_CACHE_FLOOR


def test_non_scheduling_intents_are_below_the_floor_and_that_is_expected():
    """Documents the measured shape rather than asserting a win that does not exist.

    Phase 12 scoped these prompts down by ~77% on purpose. That is the right trade — they are
    short calls — but it means caching cannot engage on them, and a future reader should not
    mistake a zero cache-read on a refill call for a bug.
    """
    for intent in (Intent.MEDICATION_REFILL, Intent.HOURS_LOCATION, Intent.BILLING_QUESTION):
        assert _prefix_tokens(intent) < HAIKU_CACHE_FLOOR


# --- the breakpoint -------------------------------------------------------------------------


def test_caching_is_on_by_default(monkeypatch):
    assert _llm(monkeypatch, None)._cache_enabled is True


def test_the_breakpoint_lands_on_the_system_block(monkeypatch):
    blocks = _llm(monkeypatch, None)._system_blocks("some prompt")
    assert blocks == [
        {"type": "text", "text": "some prompt", "cache_control": {"type": "ephemeral"}}
    ]


def test_opting_out_removes_the_breakpoint_entirely(monkeypatch):
    """`=0` must produce a request with no cache_control at all, not an empty one."""
    llm = _llm(monkeypatch, "0")
    assert llm._cache_enabled is False
    assert llm._system_blocks("some prompt") == [{"type": "text", "text": "some prompt"}]


@pytest.mark.parametrize("value", ["1", "", "true", "yes"])
def test_only_an_explicit_zero_disables_it(monkeypatch, value):
    assert _llm(monkeypatch, value)._cache_enabled is True


def test_the_fallback_prompt_is_still_cached(monkeypatch):
    """A caller constructing this adapter directly (the Pipecat-era signature) must not lose it."""
    blocks = _llm(monkeypatch, None)._system_blocks("")
    assert blocks[0]["text"] == "fallback"
    assert blocks[0]["cache_control"] == {"type": "ephemeral"}

"""Phase-8 model bake-off: provider backends behind one interface.

WHY THIS EXISTS
---------------
Phase 8 has to answer one question with evidence rather than vibes: *which LLM should the voice
agent run on?* The live agent's worst number by far is LLM latency (README.md:105-110 — LLM p50
2925 ms against an E2E p50 of 4098 ms), so the decision hinges on time-to-first-token, not on
eval pass rate alone. A model that scores 19/19 but takes three seconds to start speaking is the
wrong answer for a phone call.

Comparing candidates fairly needs three things the existing harness doesn't do:

  1. **TTFT.** `run_eval.py` used a non-streaming `messages.create`, which can only report total
     call time. TTS cannot start until the first text token arrives, so TTFT is the number that
     actually maps to how fast the caller hears a voice. Every backend here streams and records
     the moment the first content delta lands.
  2. **Cross-provider comparison.** Groq and Cerebras are the latency ceiling (~180 ms TTFT) but
     speak the OpenAI wire format, which shapes tool calls and conversation history differently
     from Anthropic's. That difference is confined to this module so `run_case()` stays
     provider-agnostic.
  3. **Cost and cache accounting.** Token usage per call, priced per candidate, plus Anthropic's
     cache-hit counters — the only way to prove prompt caching actually engaged (see below).

THE PROMPT-CACHE TRAP THIS MODULE IS BUILT TO CATCH
---------------------------------------------------
Anthropic silently declines to cache a prefix shorter than the model's minimum: the request
succeeds, `cache_control` is accepted, and `cache_creation_input_tokens` comes back 0. There is
no error and no warning. This project's system prompt + tool schemas measure ~3,814 tokens, and
Claude Haiku 4.5's minimum is 4,096 — so on the current model, caching is a no-op.

The minimum is also not monotonic across models (512 on Opus 5; 1,024 on Sonnet 5 / Sonnet 4.6 /
Opus 4.8; 4,096 on Haiku 4.5), which makes cache viability a property of the model you pick. Each
Anthropic candidate therefore carries its `cache_min_tokens`, and `CacheReport` compares measured
prompt size against it so a silent miss is reported as a finding instead of passing unnoticed.

PER-MODEL REQUEST QUIRKS
------------------------
Two Claude-generation changes will 400 or silently ruin latency if applied uniformly, so they are
per-candidate flags rather than global constants:

  * **Sampling parameters.** `temperature` is rejected on Claude Opus 5 / Sonnet 5 / Opus 4.8 /
    4.7. Haiku 4.5 accepts it, and the existing eval passes `temperature=0` to keep runs
    reproducible. `supports_temperature` gates it.
  * **Thinking defaults.** On Claude Opus 5 and Sonnet 5, *omitting* `thinking` runs adaptive
    thinking. Thinking tokens land directly on TTFT, which is exactly what a voice agent cannot
    afford, so those candidates disable it explicitly via `thinking_config`.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

# =========================================================================================
# Normalised turn result
# =========================================================================================


@dataclass
class ToolCall:
    """One tool invocation, normalised across providers."""

    id: str
    name: str
    args: dict


@dataclass
class LLMTurn:
    """One model response, normalised across providers.

    `raw_assistant_message` is the provider-shaped assistant turn to append to history. Anthropic
    needs its `tool_use` blocks echoed verbatim so the following `tool_result` blocks can
    reference them by id; OpenAI needs a `tool_calls` array. Callers never inspect it — they hand
    it back to `append_assistant()`.
    """

    text: str
    tool_calls: list[ToolCall]
    stop_reason: str
    raw_assistant_message: Any

    # Latency. ttft_ms is None when the stream produced no delta at all (an error path).
    ttft_ms: float | None
    total_ms: float

    # Usage. Cache fields stay 0 on providers that do not report them.
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def prompt_tokens(self) -> int:
        """Total prompt size however it was billed — uncached + cache write + cache read.

        `input_tokens` alone is only the *uncached remainder*, so reading it as "prompt size"
        under-reports by whatever the cache served. This is the number to compare against a
        model's `cache_min_tokens`.
        """
        return self.input_tokens + self.cache_read_tokens + self.cache_write_tokens


# =========================================================================================
# Candidate registry
# =========================================================================================


@dataclass(frozen=True)
class Candidate:
    """One model under test, plus everything needed to call and price it correctly."""

    key: str
    label: str
    provider: str  # "anthropic" | "openai_compat"
    model: str

    # USD per million tokens. None = unpriced (set the env override to enable cost reporting).
    input_cost_per_mtok: float | None = None
    output_cost_per_mtok: float | None = None

    # Anthropic prompt-cache minimum, in tokens. A prefix below this silently will not cache.
    cache_min_tokens: int | None = None

    # Rejected on Claude Opus 5 / Sonnet 5 / Opus 4.8 / 4.7 (400). Accepted on Haiku 4.5.
    supports_temperature: bool = True

    # Explicit thinking config. Needed where thinking is ON by default (Opus 5, Sonnet 5) —
    # thinking tokens land on TTFT, which a voice agent cannot afford.
    thinking_config: dict | None = None

    # openai_compat only.
    base_url: str | None = None
    api_key_env: str = "ANTHROPIC_API_KEY"

    notes: str = ""


# Anthropic cache economics: reads bill at ~0.1x, 5-minute writes at ~1.25x.
CACHE_READ_MULTIPLIER = 0.1
CACHE_WRITE_MULTIPLIER = 1.25

# Standard list prices per MTok. Claude Sonnet 5 additionally has introductory pricing
# ($2/$10) through 2026-08-31 — standard rates are used here deliberately, so a model chosen
# on cost stays the right choice after the promotion ends.
CANDIDATES: dict[str, Candidate] = {
    "haiku-4-5": Candidate(
        key="haiku-4-5",
        label="Claude Haiku 4.5 (current)",
        provider="anthropic",
        model="claude-haiku-4-5",
        input_cost_per_mtok=1.00,
        output_cost_per_mtok=5.00,
        cache_min_tokens=4096,
        supports_temperature=True,
        notes="Incumbent. Cheapest, but the ~3.8k prompt sits below its 4096 cache minimum.",
    ),
    "sonnet-4-6": Candidate(
        key="sonnet-4-6",
        label="Claude Sonnet 4.6",
        provider="anthropic",
        model="claude-sonnet-4-6",
        input_cost_per_mtok=3.00,
        output_cost_per_mtok=15.00,
        cache_min_tokens=1024,
        supports_temperature=True,
        notes="Caches at this prompt size; thinking is off when omitted — clean voice default.",
    ),
    "sonnet-5": Candidate(
        key="sonnet-5",
        label="Claude Sonnet 5",
        provider="anthropic",
        model="claude-sonnet-5",
        input_cost_per_mtok=3.00,
        output_cost_per_mtok=15.00,
        cache_min_tokens=1024,
        supports_temperature=False,  # non-default sampling params are rejected
        thinking_config={"type": "disabled"},  # adaptive is ON by default; costs TTFT
        notes="Thinking disabled explicitly. Watch tool-call accuracy — disabled thinking is "
              "documented to reduce tool-eagerness, which matters for a 3-tool flow.",
    ),
    "opus-4-8": Candidate(
        key="opus-4-8",
        label="Claude Opus 4.8",
        provider="anthropic",
        model="claude-opus-4-8",
        input_cost_per_mtok=5.00,
        output_cost_per_mtok=25.00,
        cache_min_tokens=1024,
        supports_temperature=False,
        notes="Quality ceiling for the Claude line; ~5x Haiku input cost.",
    ),
    "groq-llama": Candidate(
        key="groq-llama",
        label="Groq (Llama 3.3 70B)",
        provider="openai_compat",
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        base_url="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
        notes="Latency ceiling (~180 ms TTFT). Set GROQ_INPUT_COST/GROQ_OUTPUT_COST to price it.",
    ),
    "cerebras-llama": Candidate(
        key="cerebras-llama",
        label="Cerebras (Llama 3.3 70B)",
        provider="openai_compat",
        model=os.getenv("CEREBRAS_MODEL", "llama-3.3-70b"),
        base_url="https://api.cerebras.ai/v1",
        api_key_env="CEREBRAS_API_KEY",
        notes="Competitive TTFT with Groq, higher throughput. Requires CEREBRAS_API_KEY.",
    ),
}

DEFAULT_BAKEOFF = ["haiku-4-5", "sonnet-4-6", "sonnet-5"]


def _priced(candidate: Candidate) -> tuple[float | None, float | None]:
    """Resolve per-MTok pricing, allowing env overrides for the unpriced OSS providers."""
    prefix = candidate.key.split("-")[0].upper()
    inp = os.getenv(f"{prefix}_INPUT_COST")
    out = os.getenv(f"{prefix}_OUTPUT_COST")
    return (
        float(inp) if inp else candidate.input_cost_per_mtok,
        float(out) if out else candidate.output_cost_per_mtok,
    )


def turn_cost_usd(candidate: Candidate, turn: LLMTurn) -> float | None:
    """Price one turn, or None when the candidate has no pricing configured."""
    inp, out = _priced(candidate)
    if inp is None or out is None:
        return None
    return (
        turn.input_tokens * inp
        + turn.cache_read_tokens * inp * CACHE_READ_MULTIPLIER
        + turn.cache_write_tokens * inp * CACHE_WRITE_MULTIPLIER
        + turn.output_tokens * out
    ) / 1_000_000


# =========================================================================================
# Backend protocol
# =========================================================================================


class Backend(Protocol):
    """One provider, normalised.

    History is provider-shaped and owned entirely by the backend: `run_case()` only ever passes
    the list back in, so Anthropic's tool_use/tool_result block pairing and OpenAI's
    tool_calls/role:"tool" messages never leak into the driver.
    """

    candidate: Candidate

    async def complete(self, system: str, history: list, tools: Any) -> LLMTurn: ...

    def append_assistant(self, history: list, turn: LLMTurn) -> None: ...

    def append_tool_results(self, history: list, results: list[tuple[ToolCall, dict]]) -> None: ...

    async def close(self) -> None: ...


# =========================================================================================
# Anthropic
# =========================================================================================


class AnthropicBackend:
    """Streaming Anthropic backend with optional prompt caching.

    Caching is opt-in per run (`--cache`) rather than always-on so the bake-off can measure the
    same model with and without it and show the delta, instead of asserting a benefit.
    """

    def __init__(self, candidate: Candidate, *, max_tokens: int, use_cache: bool = False) -> None:
        from anthropic import AsyncAnthropic

        key = os.environ.get(candidate.api_key_env) or os.environ["ANTHROPIC_API_KEY"]
        self.candidate = candidate
        self.max_tokens = max_tokens
        self.use_cache = use_cache
        self._client = AsyncAnthropic(api_key=key)

    def _system_param(self, system: str) -> Any:
        """System prompt, with a cache breakpoint on the last (only) block when caching is on.

        Render order is tools -> system -> messages, so a single breakpoint on the trailing
        system block caches the tool schemas *and* the system prompt together — which is what
        makes this worth doing at all, since the tool schemas are ~1.3k of the ~3.8k prefix.
        """
        if not self.use_cache:
            return system
        return [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]

    async def complete(self, system: str, history: list, tools: Any) -> LLMTurn:
        kwargs: dict = {
            "model": self.candidate.model,
            "max_tokens": self.max_tokens,
            "system": self._system_param(system),
            "messages": history,
            "tools": tools,
        }
        if self.candidate.supports_temperature:
            kwargs["temperature"] = 0  # minimise run-to-run variance where it is accepted
        if self.candidate.thinking_config is not None:
            kwargs["thinking"] = self.candidate.thinking_config

        started = time.perf_counter()
        ttft: float | None = None

        async with self._client.messages.stream(**kwargs) as stream:
            async for event in stream:
                # First delta of any kind = the model has begun emitting. For a text turn this is
                # the moment TTS could start; for a tool turn it is the start of the arguments.
                if ttft is None and event.type == "content_block_delta":
                    ttft = (time.perf_counter() - started) * 1000.0
            message = await stream.get_final_message()

        total = (time.perf_counter() - started) * 1000.0

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        echo: list[dict] = []
        for block in message.content:
            if block.type == "text":
                text_parts.append(block.text)
                echo.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                args = dict(block.input or {})
                tool_calls.append(ToolCall(id=block.id, name=block.name, args=args))
                echo.append(
                    {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
                )

        u = message.usage
        return LLMTurn(
            text="".join(text_parts).strip(),
            tool_calls=tool_calls,
            stop_reason=message.stop_reason or "",
            raw_assistant_message=echo,
            ttft_ms=ttft,
            total_ms=total,
            input_tokens=u.input_tokens or 0,
            output_tokens=u.output_tokens or 0,
            cache_read_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
        )

    def append_assistant(self, history: list, turn: LLMTurn) -> None:
        history.append({"role": "assistant", "content": turn.raw_assistant_message})

    def append_tool_results(self, history: list, results: list[tuple[ToolCall, dict]]) -> None:
        # Anthropic carries tool results as blocks inside a *user* turn, each referencing the
        # tool_use id from the assistant turn above.
        history.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": call.id,
                        "content": json.dumps(payload),
                    }
                    for call, payload in results
                ],
            }
        )

    async def close(self) -> None:
        await self._client.close()


# =========================================================================================
# OpenAI-compatible (Groq, Cerebras)
# =========================================================================================


class OpenAICompatBackend:
    """Groq / Cerebras via the OpenAI wire format.

    Both expose OpenAI-compatible chat completions, so one backend covers them and the only
    difference is `base_url` and the API-key env var. Neither reports prompt-cache counters, so
    the cache fields stay 0 and `CacheReport` marks them not-applicable.
    """

    def __init__(self, candidate: Candidate, *, max_tokens: int) -> None:
        from openai import AsyncOpenAI

        key = os.environ.get(candidate.api_key_env)
        if not key:
            raise RuntimeError(
                f"{candidate.label} needs {candidate.api_key_env} in agent/.env "
                f"(add it, or drop {candidate.key} from --models)."
            )
        self.candidate = candidate
        self.max_tokens = max_tokens
        self._client = AsyncOpenAI(api_key=key, base_url=candidate.base_url)

    @staticmethod
    def tools_from_anthropic(anthropic_tools: list[dict]) -> list[dict]:
        """Translate Anthropic tool dicts into OpenAI function-tool dicts.

        Deliberately derived from the Anthropic schemas rather than rebuilt, so the tool
        definitions stay a single source of truth (`build_tools_schema()`): if Phase 2's tools
        change, both providers follow automatically.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t["input_schema"],
                },
            }
            for t in anthropic_tools
        ]

    async def complete(self, system: str, history: list, tools: Any) -> LLMTurn:
        # OpenAI carries the system prompt as the first message rather than a separate parameter.
        messages = [{"role": "system", "content": system}, *history]

        started = time.perf_counter()
        ttft: float | None = None

        stream = await self._client.chat.completions.create(
            model=self.candidate.model,
            max_tokens=self.max_tokens,
            messages=messages,
            tools=tools,
            temperature=0,
            stream=True,
            stream_options={"include_usage": True},
        )

        text_parts: list[str] = []
        # Tool calls stream in fragments keyed by index; accumulate then parse once at the end.
        partial: dict[int, dict] = {}
        finish_reason = ""
        usage = None

        async for chunk in stream:
            if chunk.usage is not None:
                usage = chunk.usage
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            if choice.finish_reason:
                finish_reason = choice.finish_reason
            delta = choice.delta
            if delta is None:
                continue

            if delta.content:
                if ttft is None:
                    ttft = (time.perf_counter() - started) * 1000.0
                text_parts.append(delta.content)

            for tc in delta.tool_calls or []:
                if ttft is None:
                    ttft = (time.perf_counter() - started) * 1000.0
                slot = partial.setdefault(tc.index, {"id": "", "name": "", "args": ""})
                if tc.id:
                    slot["id"] = tc.id
                if tc.function and tc.function.name:
                    slot["name"] = tc.function.name
                if tc.function and tc.function.arguments:
                    slot["args"] += tc.function.arguments

        total = (time.perf_counter() - started) * 1000.0

        tool_calls: list[ToolCall] = []
        raw_tool_calls: list[dict] = []
        for _, slot in sorted(partial.items()):
            try:
                args = json.loads(slot["args"]) if slot["args"] else {}
            except json.JSONDecodeError:
                # Malformed arguments are a real model failure, not a harness bug — surface them
                # as an empty-arg call so scoring records a miss rather than crashing the case.
                args = {}
            tool_calls.append(ToolCall(id=slot["id"], name=slot["name"], args=args))
            raw_tool_calls.append(
                {
                    "id": slot["id"],
                    "type": "function",
                    "function": {"name": slot["name"], "arguments": slot["args"] or "{}"},
                }
            )

        text = "".join(text_parts).strip()
        assistant: dict = {"role": "assistant", "content": text or None}
        if raw_tool_calls:
            assistant["tool_calls"] = raw_tool_calls

        return LLMTurn(
            text=text,
            tool_calls=tool_calls,
            # Normalise to Anthropic's vocabulary so run_case() has one thing to branch on.
            stop_reason="tool_use" if tool_calls else (finish_reason or "end_turn"),
            raw_assistant_message=assistant,
            ttft_ms=ttft,
            total_ms=total,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )

    def append_assistant(self, history: list, turn: LLMTurn) -> None:
        history.append(turn.raw_assistant_message)

    def append_tool_results(self, history: list, results: list[tuple[ToolCall, dict]]) -> None:
        # OpenAI carries each tool result as its own message, not as blocks in a user turn.
        for call, payload in results:
            history.append(
                {"role": "tool", "tool_call_id": call.id, "content": json.dumps(payload)}
            )

    async def close(self) -> None:
        await self._client.close()


def build_backend(candidate: Candidate, *, max_tokens: int, use_cache: bool = False) -> Backend:
    if candidate.provider == "anthropic":
        return AnthropicBackend(candidate, max_tokens=max_tokens, use_cache=use_cache)
    if candidate.provider == "openai_compat":
        return OpenAICompatBackend(candidate, max_tokens=max_tokens)
    raise ValueError(f"unknown provider {candidate.provider!r} for candidate {candidate.key!r}")


# =========================================================================================
# Cache verification
# =========================================================================================


async def measure_prefix_tokens(candidate: Candidate, system: str, tools: list[dict]) -> int | None:
    """Size the *cacheable prefix* — tool schemas + system prompt — in the model's own tokenizer.

    This is the number that has to clear `cache_min_tokens`, and it is easy to get wrong. Total
    prompt size is not it: the cache breakpoint sits on the trailing system block, so everything
    after it (the conversation, tool results) is outside the cached prefix. A long conversation
    pushes total prompt tokens well past the minimum while the prefix stays stubbornly below it
    and never caches — which reads as "we're above the minimum, why no cache hit?" if you measure
    the wrong quantity.

    `count_tokens` is free and exact per model, so measure rather than estimate — tokenizers
    differ across generations (Opus 4.7+ counts the same text noticeably higher than Opus 4.6).
    """
    if candidate.provider != "anthropic":
        return None
    from anthropic import AsyncAnthropic

    client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    try:
        counted = await client.messages.count_tokens(
            model=candidate.model,
            system=[{"type": "text", "text": system}],
            tools=tools,
            # count_tokens requires a non-empty messages array. One short user turn is the
            # smallest legal request; it adds a handful of tokens outside the prefix, which is
            # noise at the scale that matters here (hundreds/thousands of tokens from a limit).
            messages=[{"role": "user", "content": "hi"}],
        )
        return counted.input_tokens
    finally:
        await client.close()


@dataclass
class CacheReport:
    """Did prompt caching actually engage? The whole point of Phase 8's measurement.

    A `cache_control` marker on a too-short prefix is accepted and then ignored, so "we enabled
    caching" is not evidence. A non-zero `cache_read_input_tokens` on a later call is.
    """

    requested: bool
    calls: int = 0
    total_read: int = 0
    total_write: int = 0
    max_prompt_tokens: int = 0
    cache_min_tokens: int | None = None
    applicable: bool = True  # False on providers with no cache counters
    # Size of the cacheable prefix (tools + system), measured once via count_tokens.
    prefix_tokens: int | None = None

    def record(self, turn: LLMTurn) -> None:
        self.calls += 1
        self.total_read += turn.cache_read_tokens
        self.total_write += turn.cache_write_tokens
        self.max_prompt_tokens = max(self.max_prompt_tokens, turn.prompt_tokens)

    @property
    def engaged(self) -> bool:
        return self.total_read > 0

    def verdict(self) -> str:
        if not self.applicable:
            return "n/a (provider reports no cache counters)"
        if not self.requested:
            return "not requested"
        if self.engaged:
            return f"ENGAGED — {self.total_read:,} tokens read from cache"
        if self.cache_min_tokens and self.prefix_tokens is not None:
            if self.prefix_tokens < self.cache_min_tokens:
                short_by = self.cache_min_tokens - self.prefix_tokens
                return (
                    f"SILENT MISS — cacheable prefix is {self.prefix_tokens:,} tokens, "
                    f"{short_by:,} below this model's {self.cache_min_tokens:,} minimum"
                )
            return (
                f"SILENT MISS — prefix is {self.prefix_tokens:,} tokens (above the "
                f"{self.cache_min_tokens:,} minimum), so suspect a prefix that changes per call"
            )
        return "SILENT MISS — cause undetermined (prefix size not measured)"


# =========================================================================================
# Latency accumulation
# =========================================================================================


@dataclass
class LatencyStats:
    """TTFT and total-call samples across a whole model run."""

    ttft_ms: list[float] = field(default_factory=list)
    total_ms: list[float] = field(default_factory=list)

    def record(self, turn: LLMTurn) -> None:
        if turn.ttft_ms is not None:
            self.ttft_ms.append(turn.ttft_ms)
        self.total_ms.append(turn.total_ms)

    @staticmethod
    def _pct(values: list[float], pct: float) -> float | None:
        """Nearest-rank percentile — matches agent/src/clinic_agent/metrics.py:75."""
        if not values:
            return None
        ordered = sorted(values)
        rank = max(1, min(len(ordered), int(round(pct / 100.0 * len(ordered) + 0.5))))
        return ordered[rank - 1]

    def summary(self) -> dict[str, float | None]:
        return {
            "ttft_p50": self._pct(self.ttft_ms, 50),
            "ttft_p95": self._pct(self.ttft_ms, 95),
            "total_p50": self._pct(self.total_ms, 50),
            "total_p95": self._pct(self.total_ms, 95),
            "calls": len(self.total_ms),
        }


# =========================================================================================
# Combined per-run accumulator
# =========================================================================================


@dataclass
class RunMetrics:
    """Everything measured while one model runs the suite.

    Bundled into a single object so the eval driver takes one optional parameter instead of
    threading latency, cache, and cost accumulators separately through `run_case()`.
    """

    candidate: Candidate
    latency: LatencyStats
    cache: CacheReport
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    priced: bool = True

    @classmethod
    def for_candidate(cls, candidate: Candidate, *, use_cache: bool) -> "RunMetrics":
        return cls(
            candidate=candidate,
            latency=LatencyStats(),
            cache=CacheReport(
                requested=use_cache and candidate.provider == "anthropic",
                cache_min_tokens=candidate.cache_min_tokens,
                applicable=candidate.provider == "anthropic",
            ),
        )

    def record(self, turn: LLMTurn) -> None:
        self.latency.record(turn)
        self.cache.record(turn)
        self.input_tokens += turn.prompt_tokens
        self.output_tokens += turn.output_tokens
        cost = turn_cost_usd(self.candidate, turn)
        if cost is None:
            self.priced = False
        else:
            self.cost_usd += cost

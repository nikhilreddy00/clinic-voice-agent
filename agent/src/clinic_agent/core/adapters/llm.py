"""Phase 10 — Anthropic streaming LLM adapter.

Stateless by design: the reducer owns conversation history and hands the complete message list
down on every :class:`~clinic_agent.core.actions.StartLLM`. So this adapter holds nothing but
the client, the per-call system prompt, and the tool schemas — which means any single request
is fully reproducible from its action, and N concurrent sessions can share one adapter without
leaking each other's turns (the Phase-11 prerequisite).

Cancellation is a first-class operation. Barge-in has to abort a request that is mid-stream,
and an aborted request must **not** emit :class:`LLMFailed` — that would make the reducer speak
an apology for something the caller deliberately interrupted.

Text deltas are emitted live (that is what feeds sentence-by-sentence TTS). Tool calls are
emitted once the stream finishes, because a tool cannot run on half its arguments anyway and
letting the SDK assemble the JSON removes a whole class of partial-parse bugs.
"""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Callable
from typing import Any

import anthropic
from loguru import logger

from pipecat.adapters.schemas.tools_schema import ToolsSchema

from ... import phi
from ...intents import Intent
from ...prompts import build_system_prompt
from ...scheduling_tools import build_tools_schema
from .. import events as ev
from ..llm_router import LLMRouter, Tier
from ..reliability import CircuitBreaker, is_retryable

EmitFn = Callable[[ev.Event], None]

# Voice replies are one or two sentences plus tool calls; a generous-but-bounded cap keeps a
# runaway generation from holding the floor.
MAX_TOKENS = 1024

# Phase 15 — hedging. If no first token has arrived in this long, fire a SECOND request and take
# whichever streams first.
#
# The phase plan said to hedge to a secondary PROVIDER. Phase 8 measured the only available one
# (Groq) at 4,614 ms TTFT on the real 5,023-token booking prompt against cached Haiku's 621 ms,
# and Groq has no prompt caching so it pays full prefill every turn -- it cannot win the race it
# would exist to win. So the hedge is a second request to the SAME provider, which is the failure
# mode that actually happens: a stream that stalls or an overloaded shard.
#
# 1,800 ms is ~3x the measured 627 ms p50 TTFT (2026-09-04, 29 turns), so a healthy turn never
# fires one. `CLINIC_LLM_TTFT_MS=0` disables hedging.
HEDGE_TTFT_MS_DEFAULT = 1800.0

# Consecutive hard failures before the adapter stops trying. Per CALL, not per process: a
# per-worker breaker would let one caller's outage mute a hundred healthy calls.
# ponytail: per-call breaker; make it per-worker only if a real provider outage is ever measured
# to affect every session at once.
LLM_BREAKER_THRESHOLD = 3
LLM_BREAKER_COOLDOWN_S = 20.0


class _LostRace(Exception):
    """A hedged attempt that another attempt beat. Not an error; nothing is emitted for it."""


def to_anthropic_tools(schema: ToolsSchema) -> list[dict[str, Any]]:
    """Convert the Phase-2 tool schema to Anthropic's ``input_schema`` form.

    Reuses ``scheduling_tools.build_tools_schema()`` unchanged — the schemas were always
    provider-agnostic; Pipecat's adapter was doing this conversion, and now we do.
    """
    tools = []
    for fn in schema.standard_tools:
        tools.append(
            {
                "name": fn.name,
                "description": fn.description,
                "input_schema": {
                    "type": "object",
                    "properties": dict(fn.properties),
                    "required": list(fn.required or []),
                },
            }
        )
    return tools


# --- one HTTP client per process, not per call ------------------------------------------------
#
# Phase 14, finding 4. Every CallSession used to construct its own AsyncAnthropic (here and in
# the classifier), so each call opened two fresh connection pools and paid two TLS handshakes.
# Turn-1 TTFT measured 736-740 ms against a 540-670 ms steady state on the 2026-09-04 calls.
#
# A worker hosts N calls (Phase 11), so this is 2N pools that could be 2. Same reasoning as
# `core/vad.shared_inference_session()`: per-call state stays per-call, but an expensive
# stateless resource is process-wide.
#
# Keyed by API key so a differently-credentialed session cannot silently reuse the wrong one.
_shared_clients: dict[str, anthropic.AsyncAnthropic] = {}


def shared_anthropic_client(api_key: str) -> anthropic.AsyncAnthropic:
    """The process-wide client for ``api_key``, created on first use.

    Deliberately never closed: it outlives any one call, and closing it in a session teardown
    would tear the connection pool out from under every other call in the same worker. The
    adapters below only close a client they constructed themselves.
    """
    client = _shared_clients.get(api_key)
    if client is None:
        client = _shared_clients[api_key] = anthropic.AsyncAnthropic(api_key=api_key)
        logger.info("[llm] opened shared Anthropic client (one per process, per key)")
    return client


class AnthropicLLM:
    """Streaming Claude client that turns one request into a sequence of events."""

    def __init__(
        self,
        api_key: str,
        model: str,
        system_prompt: str,
        tools: ToolsSchema,
        emit: EmitFn,
        *,
        max_tokens: int = MAX_TOKENS,
        router: LLMRouter | None = None,
        now=None,
        client: anthropic.AsyncAnthropic | None = None,
    ) -> None:
        # An injected client is shared with other calls, so this adapter must not close it.
        self._owns_client = client is None
        self._client = client or anthropic.AsyncAnthropic(api_key=api_key)
        self._model = model
        self._tools = to_anthropic_tools(tools)
        self._emit = emit
        self._max_tokens = max_tokens
        self._tasks: dict[str, asyncio.Task] = {}
        # Phase 15. `hedges` and `retries` are reported in the session teardown line: a hedge
        # that fires on every turn is a provider problem worth seeing, not a silent cost.
        self._breaker = CircuitBreaker(
            "llm", threshold=LLM_BREAKER_THRESHOLD, cooldown_s=LLM_BREAKER_COOLDOWN_S
        )
        self._hedge_after_s = (
            float(os.getenv("CLINIC_LLM_TTFT_MS", str(HEDGE_TTFT_MS_DEFAULT))) / 1000.0
        )
        self.hedges = 0
        self.retries = 0
        # Phase 12: the reducer picks a tier; this maps it to a model. `now` is pinned at call
        # construction so every per-intent prompt in one call shares the same date table — a
        # long call must not silently change its own grounding at midnight.
        self._router = router or LLMRouter()
        self._now = now
        self._prompt_cache: dict[Intent | None, str] = {}
        self._tools_cache: dict[Intent | None, list[dict[str, Any]]] = {}

        # Phase 12 made the system prompt per-intent, so it is built per request rather than
        # held as one blob; `system_prompt` remains the fallback for callers that construct this
        # adapter directly (the Pipecat-era signature).
        self._fallback_prompt = system_prompt
        # ON by default since 2026-09-03. It was opt-in because Phase 8 measured the cacheable
        # prefix at 3,811 tokens against Haiku 4.5's 4,096 minimum, where Anthropic ACCEPTS the
        # breakpoint and then silently caches nothing — an optimization that looks live and
        # does nothing. Phase 13's six new tools ended that: the booking prefix is now 5,023
        # tokens and clears the floor.
        #
        # Cacheable prefix per intent (counted with messages.count_tokens against
        # claude-haiku-4-5, 2026-09-03). Caching is a per-intent property, not a global one:
        #
        #     None (turn one)          5,023   CACHES
        #     schedule_appointment     5,023   CACHES
        #     reschedule_appointment   3,366   below floor
        #     cancel_appointment       2,778   below floor
        #     medication_refill        2,293   below floor
        #     insurance_verification   1,621   below floor
        #     hours_location           1,610   below floor
        #     everything else         <1,000   below floor
        #
        # So this pays on turn one and on every turn of a new booking — the bulk of a
        # scheduling call — and is inert everywhere else. Measured on the Phase-8 harness over
        # three cases: TTFT p50 830 -> 642 ms, p95 1,927 -> 988 ms, $0.0705 -> $0.0195 a call,
        # 145,700 tokens served from cache.
        #
        # A below-floor breakpoint still costs nothing (no cache write happens), which is what
        # makes defaulting this on safe rather than a gamble. `CLINIC_PROMPT_CACHE=0` opts out.
        # Do NOT shrink the booking prompt to "tidy" it: 5,023 has only 927 tokens of headroom
        # over the floor, and dropping back under it would silently switch this off again.
        self._cache_enabled = os.getenv("CLINIC_PROMPT_CACHE", "1").strip() != "0"
        if self._cache_enabled:
            logger.info("[llm] prompt caching breakpoint enabled on the system block")

    def _system_blocks(self, prompt: str) -> list[dict[str, Any]]:
        """System as a block list, so a cache_control breakpoint is a one-field change."""
        block: dict[str, Any] = {"type": "text", "text": prompt or self._fallback_prompt}
        if self._cache_enabled:
            block["cache_control"] = {"type": "ephemeral"}
        return [block]

    def _scoped_prompt(self, intent: Intent | None) -> str:
        """System prompt for this intent, built once per call and reused."""
        if intent not in self._prompt_cache:
            self._prompt_cache[intent] = build_system_prompt(intent, self._now)
        return self._prompt_cache[intent]

    def _scoped_tools(self, intent: Intent | None) -> list[dict[str, Any]]:
        """Tool subset for this intent. Non-scheduling intents get none — see build_tools_schema."""
        if intent not in self._tools_cache:
            self._tools_cache[intent] = to_anthropic_tools(build_tools_schema(intent))
        return self._tools_cache[intent]

    def start(
        self,
        request_id: str,
        messages: tuple[dict[str, Any], ...],
        *,
        tier: Tier = Tier.STANDARD,
        intent: Intent | None = None,
        routing_reason: str = "",
        context_note: str = "",
    ) -> None:
        """Kick off a streaming request. Returns immediately; results arrive as events."""
        if request_id in self._tasks:
            return
        decision = self._router.resolve(tier, routing_reason)
        self._emit(
            ev.ModelRouted(
                t=time.monotonic(),
                request_id=request_id,
                tier=decision.tier.value,
                model=decision.model,
                reason=decision.reason,
            )
        )
        self._tasks[request_id] = asyncio.create_task(
            self._run(request_id, list(messages), decision, intent, context_note),
            name=f"llm-{request_id}",
        )

    def cancel(self, request_id: str) -> None:
        """Abort a request. A no-op if it already finished — that race happens constantly."""
        task = self._tasks.pop(request_id, None)
        if task is not None and not task.done():
            task.cancel()

    async def _run(self, request_id, messages, decision, intent, context_note="") -> None:
        """One logical request: breaker -> attempt (hedged) -> one retry -> events.

        Phase 15. Before this there was no retry and no deadline at all, so a single 429 or a
        stalled stream cost the caller a whole turn and produced the scripted apology.
        """
        self._emit(ev.LLMStarted(t=time.monotonic(), request_id=request_id))

        if not self._breaker.allow(time.monotonic()):
            # Two hard failures already; do not spend the caller's time on a third. The reducer
            # speaks the scripted line and the ladder takes it from there.
            logger.error(f"[llm] {request_id} refused: circuit open ({self._breaker.failures} failures)")
            self._tasks.pop(request_id, None)
            self._emit(
                ev.ProviderDegraded(
                    t=time.monotonic(), provider="llm", reason="circuit open", fatal=True
                )
            )
            self._emit(
                ev.LLMFailed(t=time.monotonic(), request_id=request_id, error="llm circuit open")
            )
            return

        prompt = self._scoped_prompt(intent)
        if context_note:
            prompt = f"{prompt}\n{context_note}"
        system = self._system_blocks(prompt)

        attempt = 0
        while True:
            try:
                text_parts, final = await self._attempt(
                    request_id, messages, decision, intent, system
                )
                break
            except asyncio.CancelledError:
                # Deliberate: barge-in or a new caller utterance superseded this request. Silent
                # by design — the reducer already moved on and must not apologize for it.
                logger.info(f"[llm] {request_id} cancelled")
                raise
            except Exception as exc:  # noqa: BLE001 - a provider error is a dialogue event
                opened = self._breaker.record_failure(time.monotonic())
                retryable = is_retryable(exc)
                if attempt == 0 and retryable and self._breaker.allow(time.monotonic()):
                    attempt += 1
                    self.retries += 1
                    logger.warning(f"[llm] {request_id} failed ({exc}); retrying once")
                    continue
                logger.error(f"[llm] {request_id} failed: {exc}")
                self._tasks.pop(request_id, None)
                if opened or not retryable:
                    self._emit(
                        ev.ProviderDegraded(
                            t=time.monotonic(), provider="llm", reason=str(exc), fatal=opened
                        )
                    )
                self._emit(
                    ev.LLMFailed(t=time.monotonic(), request_id=request_id, error=str(exc))
                )
                return

        self._breaker.record_success()

        self._tasks.pop(request_id, None)
        text = "".join(text_parts)
        if text.strip():
            logger.info(f"LLM  ▶ response generated: {phi.speech(text.strip())}")

        for block in final.content:
            if getattr(block, "type", None) == "tool_use":
                self._emit(
                    ev.LLMToolUse(
                        t=time.monotonic(),
                        request_id=request_id,
                        tool_call_id=block.id,
                        name=block.name,
                        arguments=dict(block.input or {}),
                    )
                )

        self._log_cache_usage(final)
        self._emit(
            ev.LLMCompleted(
                t=time.monotonic(),
                request_id=request_id,
                text=text,
                stop_reason=final.stop_reason or "end_turn",
            )
        )

    async def _attempt(self, request_id, messages, decision, intent, system):
        """Stream once, hedging a second identical request if no token arrives in time.

        Exactly one attempt is allowed to emit: the first to produce output claims ownership
        and the other raises :class:`_LostRace` and is cancelled. Without that claim the caller
        would hear both replies interleaved, which is a far worse failure than the slow turn
        the hedge exists to fix.
        """
        owner: list[str | None] = [None]

        def claim(tag: str) -> bool:
            if owner[0] is None:
                owner[0] = tag
            return owner[0] == tag

        def spawn(tag: str) -> asyncio.Task:
            return asyncio.create_task(
                self._stream_once(
                    request_id, messages, decision, intent, system, lambda: claim(tag)
                ),
                name=f"llm-{request_id}-{tag}",
            )

        tasks = {spawn("primary")}
        may_hedge = self._hedge_after_s > 0
        error: BaseException | None = None
        try:
            while tasks:
                timeout = self._hedge_after_s if may_hedge else None
                done, _ = await asyncio.wait(
                    tasks, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
                )
                if not done and may_hedge:
                    may_hedge = False
                    if owner[0] is None:
                        self.hedges += 1
                        logger.warning(
                            f"[llm] {request_id}: no first token in "
                            f"{self._hedge_after_s * 1000:.0f} ms — hedging a second request"
                        )
                        tasks.add(spawn("hedge"))
                    continue
                may_hedge = False
                for task in done:
                    tasks.discard(task)
                    try:
                        return task.result()
                    except (_LostRace, asyncio.CancelledError):
                        continue
                    except Exception as exc:  # noqa: BLE001 - keep waiting on the other attempt
                        error = exc
        finally:
            for task in tasks:
                task.cancel()
        raise error or RuntimeError("the model returned no response")

    async def _stream_once(self, request_id, messages, decision, intent, system, claim):
        """One streaming call. Emits deltas only while this attempt owns the output."""
        text_parts: list[str] = []
        mine = False
        async with self._client.messages.stream(
            model=decision.model,
            max_tokens=decision.max_tokens,
            system=system,
            tools=self._scoped_tools(intent),
            messages=messages,
        ) as stream:
            async for event in stream:
                if (
                    event.type == "content_block_delta"
                    and getattr(event.delta, "type", None) == "text_delta"
                ):
                    if not mine:
                        # Claim on the FIRST token, not at the end: the whole point is to start
                        # speaking as early as possible, and a claim at completion would make
                        # the hedge worthless.
                        if not claim():
                            raise _LostRace
                        mine = True
                    text_parts.append(event.delta.text)
                    self._emit(
                        ev.LLMTextDelta(
                            t=time.monotonic(), request_id=request_id, text=event.delta.text
                        )
                    )
            final = await stream.get_final_message()
        # A tool-only reply streams no text, so ownership is settled here instead.
        if not mine and not claim():
            raise _LostRace
        return text_parts, final

    def _log_cache_usage(self, final: Any) -> None:
        """Report cache hits/misses when caching is on — a zero read is the only silent-miss tell."""
        usage = getattr(final, "usage", None)
        if usage is None or not self._cache_enabled:
            return
        logger.info(
            f"[llm] cache: created={getattr(usage, 'cache_creation_input_tokens', None)} "
            f"read={getattr(usage, 'cache_read_input_tokens', None)} "
            f"input={getattr(usage, 'input_tokens', None)}"
        )

    async def aclose(self) -> None:
        for request_id in list(self._tasks):
            self.cancel(request_id)
        if self._owns_client:
            await self._client.close()

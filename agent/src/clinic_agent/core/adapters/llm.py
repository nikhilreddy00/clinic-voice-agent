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

from ...intents import Intent
from ...prompts import build_system_prompt
from ...scheduling_tools import build_tools_schema
from .. import events as ev
from ..llm_router import LLMRouter, Tier

EmitFn = Callable[[ev.Event], None]

# Voice replies are one or two sentences plus tool calls; a generous-but-bounded cap keeps a
# runaway generation from holding the floor.
MAX_TOKENS = 1024


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
    ) -> None:
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._model = model
        self._tools = to_anthropic_tools(tools)
        self._emit = emit
        self._max_tokens = max_tokens
        self._tasks: dict[str, asyncio.Task] = {}
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
        self._emit(ev.LLMStarted(t=time.monotonic(), request_id=request_id))
        text_parts: list[str] = []
        # The per-intent prompt is cached; the per-call context note is appended after it, so
        # the cached part stays byte-identical across turns and only the tail varies.
        prompt = self._scoped_prompt(intent)
        if context_note:
            prompt = f"{prompt}\n{context_note}"
        system = self._system_blocks(prompt)
        try:
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
                        text_parts.append(event.delta.text)
                        self._emit(
                            ev.LLMTextDelta(
                                t=time.monotonic(), request_id=request_id, text=event.delta.text
                            )
                        )
                final = await stream.get_final_message()
        except asyncio.CancelledError:
            # Deliberate: barge-in or a new caller utterance superseded this request. Silent by
            # design — the reducer already moved on and must not apologize for it.
            logger.info(f"[llm] {request_id} cancelled")
            raise
        except Exception as exc:  # noqa: BLE001 - a provider error is a dialogue event, not a crash
            logger.error(f"[llm] {request_id} failed: {exc}")
            self._tasks.pop(request_id, None)
            self._emit(ev.LLMFailed(t=time.monotonic(), request_id=request_id, error=str(exc)))
            return

        self._tasks.pop(request_id, None)
        text = "".join(text_parts)
        if text.strip():
            logger.info(f"LLM  ▶ response generated: {text.strip()!r}")

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
        await self._client.close()

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

from .. import events as ev

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
    ) -> None:
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._model = model
        self._tools = to_anthropic_tools(tools)
        self._emit = emit
        self._max_tokens = max_tokens
        self._tasks: dict[str, asyncio.Task] = {}

        # System is a block list, not a bare string, so a cache_control breakpoint is a
        # one-field change. It is opt-in because Phase 8 measured the cacheable prefix at 3,811
        # tokens against Haiku 4.5's 4,096 minimum: Anthropic ACCEPTS the breakpoint there and
        # then silently caches nothing (cache_creation_input_tokens: 0). Enabling it by default
        # would look like a working optimization while doing nothing. Turn it on together with
        # a model whose minimum the prompt clears (Sonnet 4.6/5 and Opus 4.8 are 1,024), and
        # verify with a non-zero usage.cache_read_input_tokens on the second call.
        self._system: list[dict[str, Any]] = [{"type": "text", "text": system_prompt}]
        if os.getenv("CLINIC_PROMPT_CACHE", "").strip() == "1":
            self._system[-1]["cache_control"] = {"type": "ephemeral"}
            logger.info("[llm] prompt caching breakpoint enabled on the system block")

    def start(self, request_id: str, messages: tuple[dict[str, Any], ...]) -> None:
        """Kick off a streaming request. Returns immediately; results arrive as events."""
        if request_id in self._tasks:
            return
        self._tasks[request_id] = asyncio.create_task(
            self._run(request_id, list(messages)), name=f"llm-{request_id}"
        )

    def cancel(self, request_id: str) -> None:
        """Abort a request. A no-op if it already finished — that race happens constantly."""
        task = self._tasks.pop(request_id, None)
        if task is not None and not task.done():
            task.cancel()

    async def _run(self, request_id: str, messages: list[dict[str, Any]]) -> None:
        self._emit(ev.LLMStarted(t=time.monotonic(), request_id=request_id))
        text_parts: list[str] = []
        try:
            async with self._client.messages.stream(
                model=self._model,
                max_tokens=self._max_tokens,
                system=self._system,
                tools=self._tools,
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
        if usage is None or "cache_control" not in self._system[-1]:
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

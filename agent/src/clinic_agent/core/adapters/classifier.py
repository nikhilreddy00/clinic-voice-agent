"""Phase 12 — the intent-classification adapter.

A small, fast, constrained model call that answers one question: what does this caller want?
It runs **in parallel** with the dialogue turn, so its latency budget (~150 ms) is spent
off the critical path entirely — the caller is already being answered while this resolves.

Three properties are deliberate:

* **Constrained decoding.** The model is given a single tool whose ``intent`` field is a strict
  enum and forced to call it (``tool_choice``). It cannot answer in prose, cannot invent an
  intent, and cannot decline. Parsing free text for one of twelve labels is a source of bugs
  that simply does not exist here.
* **No conversation history.** Only the current utterance is sent. That keeps the prompt at a
  few hundred tokens on a call that re-runs on every topic shift, and it keeps the answer
  honest — the question is what the caller *just said*.
* **Failure is not fatal.** A classifier error emits
  :class:`~clinic_agent.core.events.IntentClassificationFailed` and the call continues with the
  unscoped prompt. Intent scoping is an optimization; a caller must never lose a turn because
  a side model was unavailable.

**Provider dispatch.** The tier table (``core.llm_router``) says *which model*; this adapter
says *how to reach it*. Setting ``CLINIC_FAST_BASE_URL`` switches the transport to the OpenAI
wire format, which is what Groq and Cerebras speak, and leaves everything else identical —
same prompt, same enum, same forced tool call, same events. That one variable is the whole
switch, because the measured problem is transport-shaped: the classifier costs 928 ms p50 on
Haiku 4.5 (Phase 12) against a ~150 ms budget, and Haiku has no faster gear. It runs in
parallel so the caller never waits on it, but intent scoping only applies from the NEXT turn,
so a slow classifier means short exchanges finish before their own intent lands.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Callable

import anthropic
from loguru import logger

from .. import events as ev
from ..intent import CLASSIFIER_SYSTEM_PROMPT, CLASSIFIER_TOOL, classifier_messages
from ..llm_router import ModelSpec

EmitFn = Callable[[ev.Event], None]


def _parse_args(args: dict) -> tuple[str, float]:
    """Normalize the tool arguments both wire formats produce."""
    intent = str(args.get("intent", "unknown"))
    confidence = float(args.get("confidence", 0.0) or 0.0)
    return intent, confidence

# Beyond this the answer is worthless: the turn it was meant to scope is already over. Failing
# fast frees the slot instead of leaving a doomed request in flight.
CLASSIFY_TIMEOUT_SECS = 3.0


def _openai_tool(anthropic_tool: dict) -> dict:
    """The same tool, in the OpenAI wire format. ``input_schema`` becomes ``parameters``."""
    return {
        "type": "function",
        "function": {
            "name": anthropic_tool["name"],
            "description": anthropic_tool["description"],
            "parameters": anthropic_tool["input_schema"],
        },
    }


class IntentClassifier:
    """Classifies caller utterances against the fixed intent enum."""

    def __init__(
        self,
        api_key: str,
        spec: ModelSpec,
        emit: EmitFn,
        *,
        timeout: float = CLASSIFY_TIMEOUT_SECS,
    ) -> None:
        self._spec = spec
        self._emit = emit
        self._timeout = timeout
        self._tasks: set[asyncio.Task] = set()

        # An OpenAI-compatible base URL is the only signal needed to switch transports. The key
        # falls back to GROQ_API_KEY because Groq is the configured alternative (config.py) and
        # requiring a second name for the same secret is a setup step that buys nothing.
        base_url = os.getenv("CLINIC_FAST_BASE_URL", "").strip()
        self._base_url = base_url
        if base_url:
            from openai import AsyncOpenAI

            key = (
                os.getenv("CLINIC_FAST_API_KEY")
                or os.getenv("GROQ_API_KEY")
                or api_key
            )
            self._client = AsyncOpenAI(api_key=key, base_url=base_url)
            self._request = self._request_openai_compat
            logger.info(f"[intent] classifier via {base_url} ({spec.model})")
        else:
            self._client = anthropic.AsyncAnthropic(api_key=api_key)
            self._request = self._request_anthropic

    def classify(self, utterance: str) -> None:
        """Start a classification. Returns immediately; the answer arrives as an event."""
        if not utterance.strip():
            return
        task = asyncio.create_task(self._run(utterance), name="classify-intent")
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _request_anthropic(self, utterance: str) -> tuple[str, float] | None:
        response = await self._client.messages.create(
            model=self._spec.model,
            # Small on purpose: the only valid output is one tool call with two fields.
            max_tokens=128,
            system=CLASSIFIER_SYSTEM_PROMPT,
            tools=[CLASSIFIER_TOOL],
            # Forcing the tool is what makes this constrained decoding rather than a
            # request that the model please reply in the right shape.
            tool_choice={"type": "tool", "name": CLASSIFIER_TOOL["name"]},
            messages=classifier_messages(utterance),
        )
        for block in response.content:
            if getattr(block, "type", None) == "tool_use":
                return _parse_args(dict(block.input or {}))
        return None

    async def _request_openai_compat(self, utterance: str) -> tuple[str, float] | None:
        """Groq / Cerebras. Same forced tool call, different envelope.

        Arguments arrive as a JSON *string* here rather than a parsed dict — the one real
        difference between the two wire formats for this call, and the one place a malformed
        response can appear. A parse failure returns None and is reported as a classification
        failure, which the reducer already handles.
        """
        response = await self._client.chat.completions.create(
            model=self._spec.model,
            max_tokens=128,
            messages=[
                {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
                *classifier_messages(utterance),
            ],
            tools=[_openai_tool(CLASSIFIER_TOOL)],
            tool_choice={
                "type": "function",
                "function": {"name": CLASSIFIER_TOOL["name"]},
            },
        )
        for call in response.choices[0].message.tool_calls or []:
            try:
                return _parse_args(json.loads(call.function.arguments or "{}"))
            except (json.JSONDecodeError, TypeError):
                return None
        return None

    async def _run(self, utterance: str) -> None:
        t0 = time.monotonic()
        try:
            parsed = await asyncio.wait_for(self._request(utterance), timeout=self._timeout)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - never fatal; the turn already ran without it
            logger.warning(f"[intent] classification failed: {exc}")
            self._emit(ev.IntentClassificationFailed(t=time.monotonic(), error=str(exc)))
            return

        if parsed is None:
            # tool_choice makes this essentially unreachable, but a silently-missing
            # classification would be worse than a logged one.
            self._emit(
                ev.IntentClassificationFailed(
                    t=time.monotonic(), error="no tool call in response"
                )
            )
            return

        intent, confidence = parsed
        latency_ms = (time.monotonic() - t0) * 1000
        logger.info(f"[intent] {intent} (confidence={confidence:.2f}, {latency_ms:.0f} ms)")
        self._emit(
            ev.IntentClassified(
                t=time.monotonic(),
                intent=intent,
                confidence=confidence,
                latency_ms=latency_ms,
            )
        )

    async def aclose(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        self._tasks.clear()
        await self._client.close()

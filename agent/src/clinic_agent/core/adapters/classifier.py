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
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable

import anthropic
from loguru import logger

from .. import events as ev
from ..intent import CLASSIFIER_SYSTEM_PROMPT, CLASSIFIER_TOOL, classifier_messages
from ..llm_router import ModelSpec

EmitFn = Callable[[ev.Event], None]

# Beyond this the answer is worthless: the turn it was meant to scope is already over. Failing
# fast frees the slot instead of leaving a doomed request in flight.
CLASSIFY_TIMEOUT_SECS = 3.0


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
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._spec = spec
        self._emit = emit
        self._timeout = timeout
        self._tasks: set[asyncio.Task] = set()

    def classify(self, utterance: str) -> None:
        """Start a classification. Returns immediately; the answer arrives as an event."""
        if not utterance.strip():
            return
        task = asyncio.create_task(self._run(utterance), name="classify-intent")
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _run(self, utterance: str) -> None:
        t0 = time.monotonic()
        try:
            response = await asyncio.wait_for(
                self._client.messages.create(
                    model=self._spec.model,
                    # Small on purpose: the only valid output is one tool call with two fields.
                    max_tokens=128,
                    system=CLASSIFIER_SYSTEM_PROMPT,
                    tools=[CLASSIFIER_TOOL],
                    # Forcing the tool is what makes this constrained decoding rather than a
                    # request that the model please reply in the right shape.
                    tool_choice={"type": "tool", "name": CLASSIFIER_TOOL["name"]},
                    messages=classifier_messages(utterance),
                ),
                timeout=self._timeout,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - never fatal; the turn already ran without it
            logger.warning(f"[intent] classification failed: {exc}")
            self._emit(ev.IntentClassificationFailed(t=time.monotonic(), error=str(exc)))
            return

        latency_ms = (time.monotonic() - t0) * 1000
        for block in response.content:
            if getattr(block, "type", None) == "tool_use":
                args = dict(block.input or {})
                intent = str(args.get("intent", "unknown"))
                confidence = float(args.get("confidence", 0.0) or 0.0)
                logger.info(
                    f"[intent] {intent} (confidence={confidence:.2f}, {latency_ms:.0f} ms)"
                )
                self._emit(
                    ev.IntentClassified(
                        t=time.monotonic(),
                        intent=intent,
                        confidence=confidence,
                        latency_ms=latency_ms,
                    )
                )
                return

        # tool_choice makes this essentially unreachable, but a silently-missing classification
        # would be worse than a logged one.
        self._emit(
            ev.IntentClassificationFailed(t=time.monotonic(), error="no tool_use in response")
        )

    async def aclose(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        self._tasks.clear()
        await self._client.close()

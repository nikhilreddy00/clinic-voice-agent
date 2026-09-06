"""A scripted model, so the eval HARNESS can be tested without paying for one.

    agent/.venv/bin/python eval/run_eval.py --fake-backend

Tier 2 drives a real model against a real API and costs money per run, which is why it sits
behind a label in CI. But most of what can break in it is not the model: the tool dispatch, the
per-case database reset, the sharding, the scoring, the exit code. Those broke twice in this
project's history — `_execute_tool` silently dropped three arguments the scorer then asserted
on, and `loadtest`'s fake adapter drifted from the real signature and killed 800 sessions while
exiting 0 — and neither needed a model to find.

WHAT THIS DOES AND DOES NOT PROVE. It proves the harness books an appointment end to end: real
`SchedulingClient`, real HTTP, real Postgres, real scoring. It proves nothing about the prompt,
because there is no model reading it. A green `--fake-backend` run means "the harness works",
never "the agent works".

The script follows the booking contract rather than a recorded transcript, so it keeps working
when a case's wording changes: look at availability, hold the first slot offered, confirm it
with the details the scorer checks. It deliberately sends a date of birth, a new-patient flag
and symptom notes — those are exactly the three arguments the harness used to drop.
"""

from __future__ import annotations

import json
from typing import Any

from providers import Candidate, LLMTurn, ToolCall


class ScriptedBackend:
    """Implements the `providers.Backend` protocol with no network and no model.

    STATELESS ON PURPOSE. The first version kept per-conversation state in a dict keyed by
    `id(history)`, and CPython recycled a freed list's id: case 8 inherited case 7's
    "already booked" flag, never booked, and the suite reported an outcome miss that vanished
    when the case was run alone. Deriving everything from the history that is handed in has no
    such failure mode — and it is also how the real model works, which is the point of a fake.

    The signature is spelled out method for method rather than swallowed by ``**kwargs``:
    `loadtest/fake_adapters.FakeLLM` drifted from the real signature and the load test reported
    success for 800 dead sessions. A fake that cannot fail loudly at the seam is worse than none.
    """

    def __init__(self) -> None:
        self.candidate = Candidate(
            key="scripted", label="scripted (no model)", provider="anthropic", model="none"
        )
        self.calls = 0

    async def complete(self, system: str, history: list, tools: Any) -> LLMTurn:
        self.calls += 1

        # Everything the script knows, read back out of the conversation exactly as the model
        # would: the slot from an availability result, the hold from a hold result, and whether
        # a confirmation has already come back.
        slot_id = hold_id = None
        booked = False
        for result in self._tool_results(history):
            if result.get("slots"):
                slot_id = result["slots"][0].get("slot_id")
            if result.get("hold_id"):
                hold_id = result["hold_id"]
            if result.get("confirmation_id"):
                booked = True

        if booked:
            return self._say("You're all set. Anything else I can help with?")
        if hold_id:
            return self._call("confirm_booking", {
                "hold_id": hold_id,
                "patient_name": "Dana Reyes",
                "reason": "checkup",
                # The three arguments the harness used to drop on the floor while scoring them.
                "date_of_birth": "03/15/1990",
                "new_patient": True,
                "symptom_notes": "Routine checkup, no symptoms.",
            })
        if slot_id is not None:
            return self._call("hold_slot", {"slot_id": slot_id})
        return self._call("check_availability", {"reason_category": "checkup"})

    @staticmethod
    def _tool_results(history: list):
        """Every tool result in this conversation, oldest first."""
        for message in history:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    try:
                        yield json.loads(block["content"])
                    except (KeyError, TypeError, ValueError):
                        continue

    def _call(self, name: str, args: dict) -> LLMTurn:
        call = ToolCall(id=f"scripted-{self.calls}", name=name, args=args)
        return LLMTurn(
            text="",
            tool_calls=[call],
            stop_reason="tool_use",
            raw_assistant_message=[
                {"type": "tool_use", "id": call.id, "name": name, "input": args}
            ],
            # Zeroed, not omitted. `RunMetrics` records these for the bake-off, and a fake that
            # left them out would either crash the metrics path (it did: LLMTurn requires them)
            # or, worse, quietly contribute 0 ms samples to a real latency table.
            ttft_ms=0.0,
            total_ms=0.0,
        )

    @staticmethod
    def _say(text: str) -> LLMTurn:
        return LLMTurn(text=text, tool_calls=[], stop_reason="end_turn",
                       raw_assistant_message=[{"type": "text", "text": text}],
                       ttft_ms=0.0, total_ms=0.0)

    def append_assistant(self, history: list, turn: LLMTurn) -> None:
        history.append({"role": "assistant", "content": turn.raw_assistant_message})

    def append_tool_results(self, history: list, results: list[tuple[ToolCall, dict]]) -> None:
        history.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": call.id, "content": json.dumps(result,
                                                                                  default=str)}
            for call, result in results
        ]})

    async def close(self) -> None:
        return None

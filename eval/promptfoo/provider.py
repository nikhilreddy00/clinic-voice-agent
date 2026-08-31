"""promptfoo custom provider — drives the agent's REAL prompt and tool schemas.

The point of putting promptfoo in front of this rather than beside it: every request below is
built by ``prompts.build_system_prompt(intent)`` and ``scheduling_tools.build_tools_schema(
intent)``, the same two functions the live call uses. There is no second copy of the prompt to
drift out of sync — change the agent's prompt and these evals move with it, which is the whole
reason the suite is worth trusting.

Two provider entry points, matching the two decisions the system actually makes:

  ``classify``  — the intent classifier, exactly as core/adapters/classifier.py calls it,
                  folded through ``intents.resolve_intent`` so STICKINESS is under test too
                  (a mid-booking "March 15th 1990" must not become a new intent).
  ``respond``   — one dialogue turn for a given intent: the scoped system prompt, the scoped
                  tool subset, the staged conversation. Returns the text AND the tool calls,
                  so assertions can check what the agent DID, not merely what it said.

Both return JSON so promptfoo assertions can address fields directly.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_AGENT_SRC = Path(__file__).resolve().parents[2] / "agent" / "src"
if str(_AGENT_SRC) not in sys.path:
    sys.path.insert(0, str(_AGENT_SRC))

# Keys live in the agent's .env, which is the file the operator already maintains.
_ENV = Path(__file__).resolve().parents[2] / "agent" / ".env"
if _ENV.is_file():
    for line in _ENV.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

import anthropic  # noqa: E402

from clinic_agent.core.intent import (  # noqa: E402
    CLASSIFIER_SYSTEM_PROMPT,
    CLASSIFIER_TOOL,
    classifier_messages,
    detect_emergency,
)
from clinic_agent.core.reducer import _ACTION_CLAIM, _ACTION_NUDGE  # noqa: E402
from clinic_agent.intents import Intent, resolve_intent  # noqa: E402
from clinic_agent.prompts import build_system_prompt, caller_context_note  # noqa: E402
from clinic_agent.scheduling_tools import build_tools_schema  # noqa: E402

# Pinned so a date-dependent assertion ("tomorrow is Tuesday") cannot start failing overnight.
# The agent builds its date table from this same instant.
FIXED_NOW = datetime(2026, 9, 7, 15, 0, tzinfo=timezone.utc)  # a Monday, 11:00 clinic-local

_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
_MODEL = os.environ.get("CLINIC_EVAL_MODEL", os.environ.get("ANTHROPIC_MODEL",
                                                            "claude-haiku-4-5-20251001"))


def _to_anthropic_tools(schema) -> list[dict]:
    """The same conversion the live LLM adapter performs."""
    from clinic_agent.core.adapters.llm import to_anthropic_tools

    return to_anthropic_tools(schema)


def _messages(context_vars: dict) -> list[dict]:
    """Build the conversation. `history` stages any point in a flow, including tool results."""
    messages: list[dict] = []
    for turn in context_vars.get("history") or []:
        if isinstance(turn, str):
            messages.append({"role": "user", "content": turn})
            continue
        role = turn.get("role", "user")
        if "tool_result" in turn:
            messages.append({"role": "user", "content": [{
                "type": "tool_result",
                "tool_use_id": turn.get("tool_use_id", "tu-1"),
                "content": json.dumps(turn["tool_result"]),
            }]})
        elif "tool_use" in turn:
            messages.append({"role": "assistant", "content": [{
                "type": "tool_use",
                "id": turn.get("tool_use_id", "tu-1"),
                "name": turn["tool_use"],
                "input": turn.get("input", {}),
            }]})
        else:
            messages.append({"role": role, "content": turn["content"]})
    if context_vars.get("utterance"):
        messages.append({"role": "user", "content": context_vars["utterance"]})
    return messages


def _intent(name: str | None) -> Intent | None:
    return Intent(name) if name else None


def call_api(prompt: str, options: dict, context: dict):
    """promptfoo entry point. `mode` picks which decision is under test."""
    variables = dict(context.get("vars") or {})
    # Mode is a test VAR rather than provider config so one provider entry serves the whole
    # suite — otherwise promptfoo runs every test against every provider and half the matrix
    # is nonsense (a classification case has no intent to respond as).
    mode = variables.get("mode") or (options.get("config") or {}).get("mode", "respond")
    if mode == "classify":
        return _classify(variables)
    return _respond(variables)


# --- intent classification ------------------------------------------------------------------


def _classify(v: dict):
    utterance = v["utterance"]

    # The deterministic detector runs FIRST in the reducer, before any model exists. An eval
    # that skipped it would be testing a path production never takes.
    emergency = detect_emergency(utterance)
    if emergency:
        return {"output": json.dumps({
            "intent": "emergency", "confidence": 1.0, "source": "detector",
            "category": emergency.category,
        })}

    try:
        response = _client.messages.create(
            model=os.environ.get("CLINIC_MODEL_FAST", _MODEL),
            max_tokens=256,
            system=CLASSIFIER_SYSTEM_PROMPT,
            tools=[CLASSIFIER_TOOL],
            tool_choice={"type": "tool", "name": CLASSIFIER_TOOL["name"]},
            messages=classifier_messages(utterance),
        )
    except Exception as exc:  # noqa: BLE001 - a provider error is a test error, not a crash
        return {"error": str(exc)}

    block = next((b for b in response.content if b.type == "tool_use"), None)
    if block is None:
        return {"error": "classifier returned no tool_use"}
    proposed = str(block.input.get("intent", "unknown"))
    confidence = float(block.input.get("confidence", 0.0))

    # Stickiness is part of the decision, so it is part of the eval.
    try:
        resolved = resolve_intent(_intent(v.get("established")), Intent(proposed), confidence)
    except ValueError:
        resolved = _intent(v.get("established"))

    return {"output": json.dumps({
        "intent": resolved.value if resolved else None,
        "proposed": proposed,
        "confidence": confidence,
        "source": "classifier",
    })}


# --- one dialogue turn, for a given intent ---------------------------------------------------


def _respond(v: dict):
    intent = _intent(v.get("intent"))
    system = build_system_prompt(intent, FIXED_NOW)
    note = caller_context_note(
        known=bool(v.get("caller_known")),
        upcoming=int(v.get("upcoming") or 0),
        verified=bool(v.get("verified")),
        patient_name=v.get("patient_name") or "",
    )
    if note:
        system = f"{system}\n{note}"

    try:
        response = _client.messages.create(
            model=_MODEL,
            max_tokens=1024,
            system=system,
            tools=_to_anthropic_tools(build_tools_schema(intent)),
            messages=_messages(v),
        )
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}

    text = "".join(b.text for b in response.content if b.type == "text")
    calls = [{"name": b.name, "input": b.input} for b in response.content if b.type == "tool_use"]
    usage = [response.usage.input_tokens, response.usage.output_tokens]
    nudged = False

    # The engine's follow-through nudge, mirrored (reducer._ACTION_CLAIM). A turn that announces
    # an action and calls nothing gets exactly one more chance in production, so an eval that
    # scored the first reply alone would be measuring something no caller ever hears — and
    # would report a failure the live system recovers from. The flag is in the output so a test
    # can still assert on the difference.
    if not calls and _ACTION_CLAIM.search(text or ""):
        nudged = True
        try:
            follow_up = _client.messages.create(
                model=_MODEL,
                max_tokens=1024,
                system=system,
                tools=_to_anthropic_tools(build_tools_schema(intent)),
                messages=_messages(v) + [
                    {"role": "assistant", "content": text},
                    {"role": "user", "content": _ACTION_NUDGE},
                ],
            )
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}
        text = text + " " + "".join(b.text for b in follow_up.content if b.type == "text")
        calls += [{"name": b.name, "input": b.input}
                  for b in follow_up.content if b.type == "tool_use"]
        usage[0] += follow_up.usage.input_tokens
        usage[1] += follow_up.usage.output_tokens
        response = follow_up

    return {
        "output": json.dumps({
            "text": text.strip(),
            "tools": [c["name"] for c in calls],
            "calls": calls,
            "nudged": nudged,
            "stop_reason": response.stop_reason,
        }),
        "tokenUsage": {"prompt": usage[0], "completion": usage[1], "total": sum(usage)},
    }

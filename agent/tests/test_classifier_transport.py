"""The classifier's provider dispatch (Groq / Cerebras speak the OpenAI wire format).

No network. The point of these tests is that switching transports changes the ENVELOPE and
nothing else — same prompt, same enum, same forced tool call, same events — because the
measured reason to switch is latency (928 ms p50 on Haiku 4.5 against a ~150 ms budget) and a
transport swap that also changed behaviour would not be a fair comparison.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from clinic_agent.core import events as ev
from clinic_agent.core.adapters.classifier import IntentClassifier, _openai_tool, _parse_args
from clinic_agent.core.intent import CLASSIFIER_TOOL
from clinic_agent.core.llm_router import ModelSpec, Tier

SPEC = ModelSpec(Tier.FAST, "test-model", max_tokens=128)


def _drain(classifier: IntentClassifier, utterance: str, sink: list) -> list:
    async def go():
        classifier.classify(utterance)
        for _ in range(200):
            if sink:
                return
            await asyncio.sleep(0.005)

    asyncio.run(go())
    return sink


# --- schema translation ---------------------------------------------------------------------


def test_the_openai_tool_carries_the_same_enum_the_anthropic_one_does():
    """A drifted enum here would silently let one transport invent intents the other cannot."""
    converted = _openai_tool(CLASSIFIER_TOOL)

    assert converted["type"] == "function"
    assert converted["function"]["name"] == CLASSIFIER_TOOL["name"]
    assert converted["function"]["parameters"] == CLASSIFIER_TOOL["input_schema"]

    enum = converted["function"]["parameters"]["properties"]["intent"]["enum"]
    assert enum == CLASSIFIER_TOOL["input_schema"]["properties"]["intent"]["enum"]
    assert "emergency" in enum  # advisory-only, but the label must still be expressible


@pytest.mark.parametrize(
    "args,expected",
    [
        ({"intent": "schedule_appointment", "confidence": 0.95}, ("schedule_appointment", 0.95)),
        # OpenAI-compat hands back JSON, so a number may arrive as a string.
        ({"intent": "unknown", "confidence": "0.4"}, ("unknown", 0.4)),
        # A model that omits a field must not crash the call.
        ({}, ("unknown", 0.0)),
        ({"intent": "billing_question", "confidence": None}, ("billing_question", 0.0)),
    ],
)
def test_parse_args_normalizes_both_wire_formats(args, expected):
    assert _parse_args(args) == expected


# --- transport selection --------------------------------------------------------------------


def test_no_base_url_keeps_the_anthropic_transport(monkeypatch):
    monkeypatch.delenv("CLINIC_FAST_BASE_URL", raising=False)
    c = IntentClassifier(api_key="k", spec=SPEC, emit=lambda e: None)
    assert c._request == c._request_anthropic


def test_a_base_url_switches_to_the_openai_transport(monkeypatch):
    monkeypatch.setenv("CLINIC_FAST_BASE_URL", "https://api.groq.com/openai/v1")
    monkeypatch.setenv("GROQ_API_KEY", "gk")
    c = IntentClassifier(api_key="anthropic-key", spec=SPEC, emit=lambda e: None)
    assert c._request == c._request_openai_compat
    assert str(c._client.base_url).startswith("https://api.groq.com")
    assert c._client.api_key == "gk"  # the Groq key, not the Anthropic one


def test_the_fast_key_overrides_the_groq_key(monkeypatch):
    monkeypatch.setenv("CLINIC_FAST_BASE_URL", "https://api.cerebras.ai/v1")
    monkeypatch.setenv("GROQ_API_KEY", "gk")
    monkeypatch.setenv("CLINIC_FAST_API_KEY", "ck")
    c = IntentClassifier(api_key="ak", spec=SPEC, emit=lambda e: None)
    assert c._client.api_key == "ck"


# --- the openai-compat response path ---------------------------------------------------------


def _fake_openai(arguments: str | None, *, tool_calls: bool = True):
    """Minimal stand-in for `client.chat.completions.create`."""
    calls = (
        [SimpleNamespace(function=SimpleNamespace(arguments=arguments))] if tool_calls else []
    )
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=calls))])

    async def create(**_kwargs):
        return response

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def _classifier(monkeypatch, client, sink):
    monkeypatch.setenv("CLINIC_FAST_BASE_URL", "https://api.groq.com/openai/v1")
    monkeypatch.setenv("GROQ_API_KEY", "gk")
    c = IntentClassifier(api_key="ak", spec=SPEC, emit=sink.append)
    c._client = client
    return c


def test_a_good_openai_response_emits_the_same_event_anthropic_does(monkeypatch):
    sink: list = []
    c = _classifier(
        monkeypatch,
        _fake_openai(json.dumps({"intent": "reschedule_appointment", "confidence": 0.92})),
        sink,
    )
    (event,) = _drain(c, "I need to move my appointment", sink)

    assert isinstance(event, ev.IntentClassified)
    assert event.intent == "reschedule_appointment"
    assert event.confidence == 0.92
    assert event.latency_ms > 0


def test_truncated_tool_arguments_fail_softly(monkeypatch):
    """Measured against Groq's gpt-oss models, which spend max_tokens on reasoning first.

    Live response: `{"name": "classify_intent", "arguments": {"confidence":0.9"}` — cut off
    mid-JSON. The classifier must report a failure, not raise into the call: intent scoping is
    an optimization and the turn it would have scoped has already been answered.
    """
    sink: list = []
    c = _classifier(monkeypatch, _fake_openai('{"confidence":0.9"'), sink)
    (event,) = _drain(c, "no thank you", sink)

    assert isinstance(event, ev.IntentClassificationFailed)


def test_a_response_with_no_tool_call_fails_softly(monkeypatch):
    sink: list = []
    c = _classifier(monkeypatch, _fake_openai(None, tool_calls=False), sink)
    (event,) = _drain(c, "hello", sink)

    assert isinstance(event, ev.IntentClassificationFailed)
    assert "no tool call" in event.error


def test_a_provider_error_fails_softly_rather_than_killing_the_turn(monkeypatch):
    async def boom(**_kwargs):
        raise RuntimeError("429 rate limit")

    sink: list = []
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=boom))
    )
    c = _classifier(monkeypatch, client, sink)
    (event,) = _drain(c, "hello", sink)

    assert isinstance(event, ev.IntentClassificationFailed)
    assert "429" in event.error


def test_an_empty_utterance_never_reaches_the_provider(monkeypatch):
    sink: list = []
    c = _classifier(monkeypatch, _fake_openai("{}"), sink)
    assert _drain(c, "   ", sink) == []

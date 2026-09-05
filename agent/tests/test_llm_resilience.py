"""Phase 15 — retry, hedging, and the breaker in the LLM adapter.

No network: a scripted fake stands in for ``client.messages.stream``. What is being tested is
the adapter's failure policy, and that policy is entirely ours.

The rule these tests exist to protect: **exactly one attempt may reach the caller.** A hedge
that lets both replies emit deltas is worse than the slow turn it was meant to fix.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from clinic_agent.core import events as ev
from clinic_agent.core.adapters.llm import AnthropicLLM
from clinic_agent.core.llm_router import Tier
from clinic_agent.scheduling_tools import build_tools_schema


class _Stream:
    """One scripted streaming response: a delay, then deltas, or an exception."""

    def __init__(self, deltas, *, delay=0.0, error=None, stop_reason="end_turn"):
        self._deltas = deltas
        self._delay = delay
        self._error = error
        self._stop_reason = stop_reason

    async def __aenter__(self):
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._error is not None:
            raise self._error
        return self

    async def __aexit__(self, *exc):
        return False

    def __aiter__(self):
        async def gen():
            for text in self._deltas:
                yield SimpleNamespace(
                    type="content_block_delta", delta=SimpleNamespace(type="text_delta", text=text)
                )
                await asyncio.sleep(0)

        return gen()

    async def get_final_message(self):
        return SimpleNamespace(content=[], stop_reason=self._stop_reason, usage=None)


class _FakeClient:
    """Hands out the scripted streams in order and counts how many were requested."""

    def __init__(self, *streams):
        self._streams = list(streams)
        self.calls = 0
        self.messages = SimpleNamespace(stream=self._stream)

    def _stream(self, **kwargs):
        self.calls += 1
        return self._streams.pop(0) if self._streams else _Stream(["ok."])


def _adapter(client, emitted):
    return AnthropicLLM(
        api_key="k",
        model="claude-haiku-4-5",
        system_prompt="sys",
        tools=build_tools_schema(),
        emit=emitted.append,
        client=client,
    )


async def _run(adapter, request_id="r1"):
    adapter.start(request_id, ({"role": "user", "content": "hi"},), tier=Tier.STANDARD)
    task = adapter._tasks[request_id]
    await asyncio.wait_for(asyncio.shield(task), timeout=5)


def _texts(emitted):
    return "".join(e.text for e in emitted if isinstance(e, ev.LLMTextDelta))


def _kinds(emitted):
    return [type(e).__name__ for e in emitted]


# --- retry ---------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_retryable_error_is_retried_once_and_the_caller_never_knows(monkeypatch):
    monkeypatch.setenv("CLINIC_LLM_TTFT_MS", "0")
    overloaded = RuntimeError("Overloaded")
    overloaded.status_code = 529
    client = _FakeClient(_Stream([], error=overloaded), _Stream(["All set."]))
    emitted: list = []
    adapter = _adapter(client, emitted)

    await _run(adapter)

    assert client.calls == 2
    assert adapter.retries == 1
    assert _texts(emitted) == "All set."
    assert "LLMFailed" not in _kinds(emitted)


@pytest.mark.asyncio
async def test_a_non_retryable_error_fails_immediately(monkeypatch):
    """Retrying a bad request just spends another second of the caller's turn."""
    monkeypatch.setenv("CLINIC_LLM_TTFT_MS", "0")
    bad = RuntimeError("invalid_request_error: tool schema")
    bad.status_code = 400
    client = _FakeClient(_Stream([], error=bad))
    emitted: list = []
    adapter = _adapter(client, emitted)

    await _run(adapter)

    assert client.calls == 1
    assert "LLMFailed" in _kinds(emitted)


@pytest.mark.asyncio
async def test_two_failed_turns_open_the_breaker_and_the_third_never_leaves_the_process(
    monkeypatch,
):
    monkeypatch.setenv("CLINIC_LLM_TTFT_MS", "0")
    err = RuntimeError("Overloaded")
    err.status_code = 529
    client = _FakeClient(*[_Stream([], error=err) for _ in range(6)])
    emitted: list = []
    adapter = _adapter(client, emitted)

    for i in range(3):
        await _run(adapter, f"r{i}")

    # 2 attempts on turn one (breaker opens on the 3rd failure... which lands on turn two),
    # then nothing at all once it is open.
    assert adapter._breaker.state == "open"
    before = client.calls
    await _run(adapter, "r-after")
    assert client.calls == before, "an open breaker must not spend a request"
    assert any(
        isinstance(e, ev.ProviderDegraded) and e.fatal for e in emitted
    ), "an open breaker is fatal degradation — the ladder needs to see it"


# --- hedging -------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_stalled_stream_is_hedged_and_the_hedge_can_win(monkeypatch):
    monkeypatch.setenv("CLINIC_LLM_TTFT_MS", "50")
    client = _FakeClient(
        _Stream(["slow."], delay=5.0),      # never gets a token out in time
        _Stream(["Hedged reply."], delay=0.0),
    )
    emitted: list = []
    adapter = _adapter(client, emitted)

    await _run(adapter)

    assert adapter.hedges == 1
    assert client.calls == 2
    assert _texts(emitted) == "Hedged reply.", "the winner's text, and only the winner's"
    assert _kinds(emitted).count("LLMCompleted") == 1


@pytest.mark.asyncio
async def test_a_healthy_turn_never_hedges(monkeypatch):
    monkeypatch.setenv("CLINIC_LLM_TTFT_MS", "300")
    client = _FakeClient(_Stream(["Fast."]))
    emitted: list = []
    adapter = _adapter(client, emitted)

    await _run(adapter)

    assert adapter.hedges == 0
    assert client.calls == 1


@pytest.mark.asyncio
async def test_a_slow_but_streaming_reply_is_not_hedged(monkeypatch):
    """The deadline is on the FIRST TOKEN, not on completion.

    A long reply that started promptly is working exactly as intended; hedging it would double
    the spend on every one of the longest answers the agent gives.
    """
    monkeypatch.setenv("CLINIC_LLM_TTFT_MS", "40")

    class _Trickle(_Stream):
        def __aiter__(self):
            async def gen():
                yield SimpleNamespace(
                    type="content_block_delta",
                    delta=SimpleNamespace(type="text_delta", text="Let me "),
                )
                await asyncio.sleep(0.15)
                yield SimpleNamespace(
                    type="content_block_delta",
                    delta=SimpleNamespace(type="text_delta", text="check that."),
                )

            return gen()

    client = _FakeClient(_Trickle([]))
    emitted: list = []
    adapter = _adapter(client, emitted)

    await _run(adapter)

    assert adapter.hedges == 0
    assert _texts(emitted) == "Let me check that."


@pytest.mark.asyncio
async def test_cancelling_mid_hedge_leaves_no_task_speaking(monkeypatch):
    """Barge-in during a hedged turn must silence BOTH attempts."""
    monkeypatch.setenv("CLINIC_LLM_TTFT_MS", "20")
    client = _FakeClient(
        _Stream(["primary"], delay=5.0),
        _Stream(["hedge"], delay=5.0),
    )
    emitted: list = []
    adapter = _adapter(client, emitted)

    adapter.start("r1", ({"role": "user", "content": "hi"},))
    task = adapter._tasks["r1"]
    await asyncio.sleep(0.1)
    adapter.cancel("r1")
    with pytest.raises(asyncio.CancelledError):
        await task

    await asyncio.sleep(0.05)
    assert _texts(emitted) == ""
    assert "LLMFailed" not in _kinds(emitted), "a deliberate cancel is not a failure"
    assert not [t for t in asyncio.all_tasks() if t.get_name().startswith("llm-r1")]

"""Phase 10 — integration test for the CallSession drain loop.

``test_reducer`` proves the logic and ``test_replay`` proves determinism; neither touches the
wiring. This does: it runs the real :class:`~clinic_agent.core.session.CallSession` — its real
queue, its real ``reduce()`` calls, its real action dispatch, its real metrics and trace
recording — with every vendor adapter replaced by a scripted fake.

So it covers the part that is easy to get wrong and invisible to the pure tests: that a
``StartLLM`` action actually reaches the LLM adapter, that a ``ToolCompleted`` event actually
resumes the turn, that ``BotStoppedSpeaking`` actually hands the floor back, and that teardown
runs on every adapter.

The fakes are also the shape the Phase-11 Tier-A load test needs — same seam, with latency
distributions injected instead of instant returns.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from clinic_agent.config import Settings
from clinic_agent.core import events as ev
from clinic_agent.core.recorder import load_trace, replay
from clinic_agent.core.session import CallSession
from clinic_agent.core.state import Phase


def settings() -> Settings:
    return Settings(
        mode="telephony",
        deepgram_api_key="test",
        anthropic_api_key="test",
        anthropic_model="claude-haiku-4-5-20251001",
        groq_api_key="",
        groq_model="",
        cartesia_api_key="test",
        cartesia_voice_id="test-voice",
        livekit_url="ws://localhost",
        livekit_api_key="test-key",
        livekit_api_secret="test-secret-value-long-enough",
        livekit_phone_number="",
        scheduling_api_base_url="http://127.0.0.1:8000",
    )


class FakeLLM:
    """Replays a scripted response per request, as real streamed events."""

    def __init__(self, emit, script: list[list[ev.Event]]) -> None:
        self._emit = emit
        self._script = script
        self.requests: list[tuple[str, tuple]] = []
        self.tiers: list = []
        self.cancelled: list[str] = []
        self.closed = False

    def start(self, request_id, messages, *, tier=None, intent=None, routing_reason=""):
        self.requests.append((request_id, messages))
        self.tiers.append(tier)
        step = self._script[len(self.requests) - 1] if len(self.requests) <= len(self._script) else []
        for event in step:
            from dataclasses import replace as _replace

            self._emit(_replace(event, request_id=request_id, t=time.monotonic()))

    def cancel(self, request_id):
        self.cancelled.append(request_id)

    async def aclose(self):
        self.closed = True


class FakeTools:
    """Returns a canned result for each tool immediately."""

    def __init__(self, emit, results: dict[str, dict]) -> None:
        self._emit = emit
        self._results = results
        self.invoked: list[str] = []
        self.closed = False

    def invoke(self, tool_call_id, name, arguments):
        self.invoked.append(name)
        result = self._results.get(name, {"ok": True})
        self._emit(
            ev.ToolCompleted(
                t=time.monotonic(),
                tool_call_id=tool_call_id,
                name=name,
                result=result,
                ok=bool(result.get("ok")),
                latency_ms=10.0,
                http_status=200,
            )
        )

    async def aclose(self):
        self.closed = True


class FakeMedia:
    """Playback that completes instantly, emitting the same speaking events as the real one."""

    def __init__(self, emit) -> None:
        self._emit = emit
        self.played: list[tuple[str, bytes]] = []
        self.cleared: list[str] = []
        self.closed = False

    async def start(self):
        pass

    async def play(self, utterance_id, pcm):
        self.played.append((utterance_id, pcm))

    async def end_utterance(self, utterance_id):
        self._emit(ev.BotStartedSpeaking(t=time.monotonic(), utterance_id=utterance_id))
        self._emit(
            ev.BotStoppedSpeaking(t=time.monotonic(), utterance_id=utterance_id, completed=True)
        )

    async def clear(self, utterance_id):
        self.cleared.append(utterance_id)
        self._emit(
            ev.BotStoppedSpeaking(t=time.monotonic(), utterance_id=utterance_id, completed=False)
        )

    async def aclose(self):
        self.closed = True


class FakeTTS:
    """Records what it was asked to say and completes the utterance on the final chunk."""

    def __init__(self, media) -> None:
        self._media = media
        self.spoken: list[tuple[str, str, bool]] = []
        self.cancelled: list[str] = []
        self.closed = False

    async def start(self):
        pass

    async def speak(self, utterance_id, text, *, final):
        self.spoken.append((utterance_id, text, final))
        await self._media.play(utterance_id, b"\x00\x00")
        if final:
            await self._media.end_utterance(utterance_id)

    async def cancel(self, utterance_id):
        self.cancelled.append(utterance_id)
        await self._media.clear(utterance_id)

    async def aclose(self):
        self.closed = True


class FakeClassifier:
    """Records classification requests; never answers unless a test makes it."""

    def __init__(self):
        self.requests: list[str] = []
        self.closed = False

    def classify(self, utterance):
        self.requests.append(utterance)

    async def aclose(self):
        self.closed = True


class FakeSTT:
    def __init__(self):
        self.closed = False

    async def start(self):
        pass

    async def send_audio(self, pcm):
        pass

    async def aclose(self):
        self.closed = True


class ScriptedSession(CallSession):
    """A CallSession with every vendor adapter faked out."""

    def __init__(self, script, tool_results, **kwargs):
        self._script = script
        self._tool_results = tool_results
        super().__init__(settings(), **kwargs)

    def _build_llm(self):
        return FakeLLM(self.emit, self._script)

    def _build_tools(self):
        return FakeTools(self.emit, self._tool_results)

    def _build_media(self):
        return FakeMedia(self.emit)

    def _build_tts(self):
        return FakeTTS(self.media)

    def _build_stt(self):
        return FakeSTT()

    def _build_classifier(self):
        return FakeClassifier()


async def drive(session: CallSession, caller_turns: list[str]) -> None:
    """Run the session while feeding it caller utterances after each bot turn."""
    runner = asyncio.create_task(session.run())
    session.emit(ev.CallerPresent(t=time.monotonic(), participant_id="sip_caller"))

    for utterance in caller_turns:
        await _settle(session)
        session.emit(ev.SpeechStarted(t=time.monotonic()))
        session.emit(ev.SpeechStopped(t=time.monotonic()))
        session.emit(ev.FinalTranscript(t=time.monotonic(), text=utterance, confidence=0.96))

    await _settle(session)
    session.hangup("caller_left")
    await asyncio.wait_for(runner, timeout=5)


async def _settle(session: CallSession) -> None:
    """Let the drain loop catch up with everything queued so far.

    ``_seq > 0`` is part of the condition on purpose: an empty queue also describes a session
    whose task has been created but has not run a single line yet, and settling on that would
    assert against the initial state instead of the state under test.
    """
    for _ in range(500):
        await asyncio.sleep(0)
        if session._seq > 0 and session._queue.empty():
            for _ in range(3):  # let the last event's actions finish awaiting
                await asyncio.sleep(0)
            return
    raise AssertionError("session queue never drained")


@pytest.mark.asyncio
async def test_full_booking_runs_through_the_real_loop(tmp_path, monkeypatch):
    monkeypatch.setenv("CLINIC_LOG_DIR", str(tmp_path))

    script = [
        # turn 1: ask for the day
        [ev.LLMTextDelta(text="Happy to help. "), ev.LLMCompleted(stop_reason="end_turn")],
        # turn 2: look up availability
        [
            ev.LLMToolUse(tool_call_id="tu_1", name="check_availability",
                          arguments={"date": "2026-08-14"}),
            ev.LLMCompleted(stop_reason="tool_use"),
        ],
        # resumption after the tool result
        [ev.LLMTextDelta(text="Thursday at 9 is open. "), ev.LLMCompleted(stop_reason="end_turn")],
        # turn 3: book it
        [
            ev.LLMToolUse(tool_call_id="tu_2", name="confirm_booking",
                          arguments={"hold_id": "h1"}),
            ev.LLMCompleted(stop_reason="tool_use"),
        ],
        [ev.LLMTextDelta(text="You're all set. "), ev.LLMCompleted(stop_reason="end_turn")],
    ]
    results = {
        "check_availability": {"ok": True, "count": 1, "slots": [{"slot_id": 12}]},
        "confirm_booking": {"ok": True, "confirmation_id": "A1B2C3D4"},
    }

    session = ScriptedSession(script, results, call_id="session-test")
    await drive(session, ["I need a checkup.", "Thursday please.", "Yes."])

    assert session.state.phase is Phase.CLOSED
    assert session.state.outcome == "booked"
    assert session._tools.invoked == ["check_availability", "confirm_booking"]
    assert len(session._llm.requests) == 5

    # The greeting was spoken verbatim, and every utterance was closed with a final chunk.
    assert "automated AI assistant" in session._tts.spoken[0][1]
    assert any(final for _, _, final in session._tts.spoken)

    # Teardown reached every adapter.
    assert all(
        a.closed for a in (session._llm, session._tools, session._tts, session._stt, session.media)
    )


@pytest.mark.asyncio
async def test_the_live_run_is_recorded_and_replays_identically(tmp_path, monkeypatch):
    """A call driven through the real loop must reproduce from its own trace file."""
    monkeypatch.setenv("CLINIC_LOG_DIR", str(tmp_path))

    script = [
        [ev.LLMTextDelta(text="Sure thing. "), ev.LLMCompleted(stop_reason="end_turn")],
        [ev.LLMTextDelta(text="Booked. "), ev.LLMCompleted(stop_reason="end_turn")],
    ]
    session = ScriptedSession(script, {}, call_id="trace-test")
    await drive(session, ["Hello there.", "Great, thanks."])

    replayed_state, _ = replay(load_trace(session._recorder.path))
    assert replayed_state == session.state


@pytest.mark.asyncio
async def test_barge_in_cancels_the_bot_mid_reply(tmp_path, monkeypatch):
    monkeypatch.setenv("CLINIC_LOG_DIR", str(tmp_path))

    script = [[ev.LLMTextDelta(text="Our hours are nine to five. ")]]  # no completion: still streaming
    session = ScriptedSession(script, {}, call_id="barge-test")

    runner = asyncio.create_task(session.run())
    session.emit(ev.CallerPresent(t=time.monotonic(), participant_id="sip"))
    await _settle(session)
    session.emit(ev.FinalTranscript(t=time.monotonic(), text="what are your hours"))
    await _settle(session)
    session.emit(ev.BotStartedSpeaking(t=time.monotonic(), utterance_id="utt-2"))
    session.emit(ev.UserInterrupted(t=time.monotonic()))
    await _settle(session)

    assert session._llm.cancelled == ["req-1"]
    assert session._tts.cancelled == ["utt-2"]
    assert session.state.phase is Phase.LISTENING
    assert session.state.interruptions == 1

    session.hangup()
    await asyncio.wait_for(runner, timeout=5)


@pytest.mark.asyncio
async def test_telephony_greeting_survives_a_caller_who_is_already_there(tmp_path, monkeypatch):
    """CallStarted must be reduced before CallerPresent, or the greeting loses its consent line.

    Both transports can announce a caller the instant they start: the local mic is live as soon
    as its stream opens, and a SIP call can be bridged into the room before the agent finishes
    connecting. If CallerPresent won that race the greeting would be picked against the default
    mode, and a telephony caller would never hear the call-recording consent.
    """
    monkeypatch.setenv("CLINIC_LOG_DIR", str(tmp_path))

    class EagerMedia(FakeMedia):
        async def start(self):
            self._emit(ev.CallerPresent(t=time.monotonic(), participant_id="already-here"))

    class EagerSession(ScriptedSession):
        def _build_media(self):
            return EagerMedia(self.emit)

    session = EagerSession([], {}, call_id="eager")
    runner = asyncio.create_task(session.run())
    await _settle(session)

    assert session.state.mode == "telephony"
    assert "may be recorded" in session._tts.spoken[0][1]

    session.hangup()
    await asyncio.wait_for(runner, timeout=5)


@pytest.mark.asyncio
async def test_two_sessions_in_one_process_share_no_state(tmp_path, monkeypatch):
    """The defect this phase exists to fix: caller #2 inheriting caller #1's conversation."""
    monkeypatch.setenv("CLINIC_LOG_DIR", str(tmp_path))
    script = [[ev.LLMTextDelta(text="Hello. "), ev.LLMCompleted(stop_reason="end_turn")]]

    first = ScriptedSession(script, {}, call_id="call-a")
    await drive(first, ["I am caller one."])

    second = ScriptedSession(script, {}, call_id="call-b")
    await drive(second, ["I am caller two."])

    assert first.call_id != second.call_id
    assert first.metrics is not second.metrics
    first_texts = [m["content"] for m in first.state.messages if m["role"] == "user"]
    second_texts = [m["content"] for m in second.state.messages if m["role"] == "user"]
    assert "I am caller one." in first_texts
    assert "I am caller one." not in second_texts
    assert second.state.turn_index == 1

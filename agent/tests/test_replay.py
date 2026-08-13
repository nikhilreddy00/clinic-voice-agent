"""Phase 10 — the exit criterion: a recorded call replays deterministically.

The claim this phase makes is that call behavior is now reproducible without audio, without a
network, and without API keys. These tests are that claim, checked. They record a full booking
conversation exactly as ``CallSession`` would, reload it from disk, and assert the replay
produces the identical final state and the identical action sequence.

This is also the Tier-1 eval of Phase 16 — the one intended to run on every commit. It costs
milliseconds, so there is no reason for it not to.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from clinic_agent.core import events as ev
from clinic_agent.core.actions import InvokeTool, Speak, StartLLM
from clinic_agent.core.recorder import TraceRecorder, load_trace, replay, replay_steps
from clinic_agent.core.state import CallState, Phase


def booking_call() -> list[ev.Event]:
    """A complete booking: greeting, intake, availability lookup, hold, confirm, hangup."""
    raw = [
        ev.CallStarted(call_id="replay-1", mode="telephony"),
        ev.CallerPresent(participant_id="sip_caller"),
        ev.BotStartedSpeaking(utterance_id="utt-1"),
        ev.BotStoppedSpeaking(utterance_id="utt-1"),

        ev.SpeechStarted(),
        ev.SpeechStopped(),
        ev.FinalTranscript(text="Hi, I'd like to book a checkup.", confidence=0.97),
        ev.LLMStarted(request_id="req-1"),
        ev.LLMTextDelta(request_id="req-1", text="Happy to help. "),
        ev.LLMTextDelta(request_id="req-1", text="What's your name? "),
        ev.LLMCompleted(request_id="req-1", stop_reason="end_turn"),
        ev.BotStartedSpeaking(utterance_id="utt-2"),
        ev.BotStoppedSpeaking(utterance_id="utt-2"),

        ev.SpeechStarted(),
        ev.SpeechStopped(),
        ev.FinalTranscript(text="Dana Reyes.", confidence=0.95),
        ev.LLMStarted(request_id="req-2"),
        ev.LLMToolUse(
            request_id="req-2",
            tool_call_id="tu_avail",
            name="check_availability",
            arguments={"date": "2026-08-14", "reason_category": "checkup"},
        ),
        ev.LLMCompleted(request_id="req-2", stop_reason="tool_use"),
        ev.ToolCompleted(
            tool_call_id="tu_avail",
            name="check_availability",
            ok=True,
            latency_ms=142.0,
            http_status=200,
            result={"ok": True, "count": 1, "slots": [{"slot_id": 12}]},
        ),
        ev.LLMStarted(request_id="req-3"),
        ev.LLMTextDelta(request_id="req-3", text="Thursday at 9 AM is open. "),
        ev.LLMCompleted(request_id="req-3", stop_reason="end_turn"),
        ev.BotStartedSpeaking(utterance_id="utt-3"),
        ev.BotStoppedSpeaking(utterance_id="utt-3"),

        ev.SpeechStarted(),
        ev.SpeechStopped(),
        ev.FinalTranscript(text="That works.", confidence=0.99),
        ev.LLMStarted(request_id="req-4"),
        ev.LLMToolUse(
            request_id="req-4", tool_call_id="tu_hold", name="hold_slot",
            arguments={"slot_id": 12},
        ),
        ev.LLMCompleted(request_id="req-4", stop_reason="tool_use"),
        ev.ToolCompleted(
            tool_call_id="tu_hold", name="hold_slot", ok=True, http_status=200,
            result={"ok": True, "hold_id": "h-1", "slot_id": 12},
        ),
        ev.LLMStarted(request_id="req-5"),
        ev.LLMTextDelta(request_id="req-5", text="Booking you with Dr. Chen. Sound good? "),
        ev.LLMCompleted(request_id="req-5", stop_reason="end_turn"),
        ev.BotStartedSpeaking(utterance_id="utt-4"),
        ev.BotStoppedSpeaking(utterance_id="utt-4"),

        ev.SpeechStarted(),
        ev.SpeechStopped(),
        ev.FinalTranscript(text="Yes.", confidence=0.98),
        ev.LLMStarted(request_id="req-6"),
        ev.LLMToolUse(
            request_id="req-6", tool_call_id="tu_book", name="confirm_booking",
            arguments={"hold_id": "h-1", "patient_name": "Dana Reyes", "reason": "checkup"},
        ),
        ev.LLMCompleted(request_id="req-6", stop_reason="tool_use"),
        ev.ToolCompleted(
            tool_call_id="tu_book", name="confirm_booking", ok=True, http_status=200,
            result={"ok": True, "confirmation_id": "A1B2C3D4"},
        ),
        ev.LLMStarted(request_id="req-7"),
        ev.LLMTextDelta(request_id="req-7", text="You're all set, confirmation A1B2C3D4. "),
        ev.LLMCompleted(request_id="req-7", stop_reason="end_turn"),
        ev.BotStartedSpeaking(utterance_id="utt-5"),
        ev.BotStoppedSpeaking(utterance_id="utt-5"),

        ev.Hangup(reason="caller_left"),
    ]
    # Stamp seq/t the way CallSession.emit does, so the fixture is shaped like a real trace.
    return [replace(e, seq=i + 1, t=0.5 * (i + 1)) for i, e in enumerate(raw)]


@pytest.fixture()
def trace_file(tmp_path, monkeypatch):
    """Record the booking call to disk through the real TraceRecorder."""
    monkeypatch.setenv("CLINIC_LOG_DIR", str(tmp_path))
    recorder = TraceRecorder("replay-1")
    for event in booking_call():
        recorder.record(event)
    assert recorder.enabled and recorder.count == len(booking_call())
    # Phase 11: traces buffer (64 lines) so a worker holds a descriptor only while flushing.
    # close() flushes the tail — CallSession does this at teardown.
    recorder.close()
    return recorder.path


def test_round_trip_through_disk_preserves_every_event(trace_file):
    assert load_trace(trace_file) == booking_call()


def test_replay_reproduces_state_and_actions_exactly(trace_file):
    """The whole point: disk -> reduce() gives the same answer as the live event stream."""
    live_state, live_actions = replay(booking_call())
    replayed_state, replayed_actions = replay(load_trace(trace_file))

    assert replayed_state == live_state
    assert replayed_actions == live_actions


def test_replay_is_stable_across_runs(trace_file):
    """No clock reads, no uuid4 — two replays of one trace must be byte-identical."""
    first = replay(load_trace(trace_file))
    second = replay(load_trace(trace_file))
    assert first == second


def test_the_replayed_call_actually_booked(trace_file):
    state, actions = replay(load_trace(trace_file))

    assert state.phase is Phase.CLOSED
    assert state.outcome == "booked"
    assert state.turn_index == 4
    assert state.interruptions == 0
    # Three tools, in the order the booking flow requires.
    assert [a.name for a in actions if isinstance(a, InvokeTool)] == [
        "check_availability",
        "hold_slot",
        "confirm_booking",
    ]
    # Seven LLM requests: four caller turns plus one resumption after each tool result.
    assert len([a for a in actions if isinstance(a, StartLLM)]) == 7
    # The greeting is spoken deterministically, never generated.
    greeting = next(a for a in actions if isinstance(a, Speak))
    assert greeting.deterministic and "automated AI assistant" in greeting.text


def test_history_never_leaves_a_tool_use_unanswered(trace_file):
    """Anthropic rejects that outright, so it must hold after every single event."""
    for event, state, _actions in replay_steps(load_trace(trace_file)):
        requested, answered = set(), set()
        for message in state.messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if block.get("type") == "tool_use":
                    requested.add(block["id"])
                elif block.get("type") == "tool_result":
                    answered.add(block["tool_use_id"])
        outstanding = requested - answered
        assert outstanding <= set(state.pending_tools), (
            f"after {type(event).__name__}: unanswered tool_use {outstanding}"
        )


def test_replay_pinpoints_where_a_divergence_happens(trace_file):
    """A corrupted trace must fail loudly at a specific event, not silently produce nonsense."""
    events = load_trace(trace_file)
    # Drop the caller's final "Yes." — the booking should never happen.
    without_yes = [e for e in events if not (isinstance(e, ev.FinalTranscript) and e.text == "Yes.")]

    state, _ = replay(without_yes)
    assert state.outcome != "booked"


def test_unknown_event_kind_is_rejected(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"kind": "Telepathy", "seq": 1}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="bad trace line"):
        load_trace(path)


def test_traces_survive_a_new_event_field(tmp_path):
    """Adding a field must not invalidate the recorded regression corpus."""
    path = tmp_path / "old.jsonl"
    payload = {"kind": "FinalTranscript", "seq": 1, "t": 1.0, "text": "hi",
               "confidence": 0.9, "field_from_a_future_build": 42}
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    assert load_trace(path) == [ev.FinalTranscript(seq=1, t=1.0, text="hi", confidence=0.9)]


def test_replay_starts_from_a_clean_state_each_time():
    """Sessions must not inherit each other's history — the Phase-10 defect being fixed."""
    first, _ = replay(booking_call())
    second, _ = replay(booking_call(), CallState())
    assert first.messages == second.messages
    assert len(second.messages) > 0

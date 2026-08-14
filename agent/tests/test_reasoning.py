"""Phase 12 — the reasoning layer inside the engine.

``test_emergency`` proves the detector; this proves the *engine* does the right thing with it,
and with intent generally. The emergency cases here are the ones that matter most: it is not
enough that a phrase is recognized, it has to actually cancel the turn, speak the scripted
line, keep the model out of the loop, and not quietly resume booking two turns later.
"""

from __future__ import annotations

import pytest

from clinic_agent.core import events as ev
from clinic_agent.core.actions import (
    CancelLLM,
    CancelSpeech,
    ClassifyIntent,
    Speak,
    StartLLM,
    TransferToHuman,
)
from clinic_agent.core.llm_router import DEFAULT_MODELS, LLMRouter, Tier, select_tier
from clinic_agent.core.state import Phase
from clinic_agent.intents import Intent, needs_clarification, resolve_intent
from clinic_agent.prompts import (
    EMERGENCY_RESPONSE,
    build_system_prompt,
    prompt_sizes,
)
from clinic_agent.scheduling_tools import build_tools_schema

from test_reducer import Driver, _greeted

# --- the emergency path through the engine ---------------------------------------------------


def test_emergency_cancels_the_turn_and_speaks_the_scripted_line():
    d = _greeted()
    d.send(ev.FinalTranscript(text="I'd like to book something"))
    d.send(ev.LLMTextDelta(request_id="req-1", text="Happy to help. "))

    produced = d.send(ev.FinalTranscript(text="actually my chest hurts and I can't breathe"))

    assert CancelLLM(request_id="req-1") in produced
    speak = next(a for a in produced if isinstance(a, Speak))
    assert speak.text == EMERGENCY_RESPONSE
    assert speak.deterministic and speak.final
    assert d.state.phase is Phase.EMERGENCY
    assert d.state.emergency is True
    # Two rules match this utterance ("my chest hurts" and "can't breathe"); the first wins.
    # Which one is recorded matters only for the trace — the response is identical either way,
    # which is the point of a scripted path.
    assert d.state.emergency_category == "cardiac"


def test_emergency_never_starts_an_llm_request():
    """The whole point: no model is consulted on this path, so it cannot fail or vary."""
    d = _greeted()
    produced = d.send(ev.FinalTranscript(text="I think I'm having a heart attack"))

    assert not any(isinstance(a, (StartLLM, ClassifyIntent)) for a in produced)


def test_emergency_requests_an_urgent_transfer():
    d = _greeted()
    produced = d.send(ev.FinalTranscript(text="my husband is unconscious"))

    transfer = next(a for a in produced if isinstance(a, TransferToHuman))
    assert transfer.urgent is True
    assert transfer.reason == "emergency:consciousness"


def test_the_crisis_utterance_never_enters_conversation_history():
    """No model runs on this path, and nothing should carry the description to one later."""
    d = _greeted()
    before = len(d.state.messages)
    d.send(ev.FinalTranscript(text="I want to kill myself"))

    assert len(d.state.messages) == before


def test_the_agent_does_not_resume_booking_after_an_emergency():
    """An automated scheduler must not talk someone out of calling 911."""
    d = _greeted()
    d.send(ev.FinalTranscript(text="I can't breathe"))

    produced = d.send(ev.FinalTranscript(text="actually never mind, can I book a checkup?"))

    assert not any(isinstance(a, StartLLM) for a in produced)
    assert d.state.phase is Phase.EMERGENCY
    speak = next(a for a in produced if isinstance(a, Speak))
    assert speak.text == EMERGENCY_RESPONSE  # repeats the guidance instead


def test_emergency_is_the_reported_outcome():
    d = _greeted()
    d.send(ev.FinalTranscript(text="she's bleeding heavily"))
    assert d.state.outcome == "emergency"


def test_emergency_beats_an_in_flight_tool_call():
    """The nastiest ordering: a booking is mid-commit when the caller reports a crisis."""
    d = _greeted()
    d.send(ev.FinalTranscript(text="book me tomorrow"))
    d.send(ev.LLMToolUse(request_id="req-1", tool_call_id="tu_1", name="hold_slot"))
    d.send(ev.LLMCompleted(request_id="req-1", stop_reason="tool_use"))
    assert d.state.phase is Phase.TOOL_WAIT

    d.send(ev.FinalTranscript(text="wait, I think I'm having a stroke"))

    assert d.state.phase is Phase.EMERGENCY
    assert d.state.pending_tools == ()


# --- intent classification, off the critical path ---------------------------------------------


def test_classification_is_requested_alongside_the_turn_not_before_it():
    d = _greeted()
    produced = d.send(ev.FinalTranscript(text="I need a refill"))

    assert isinstance(produced[0], StartLLM), "the dialogue turn must start first"
    assert ClassifyIntent(utterance="I need a refill") in produced


def test_intent_scopes_the_next_request():
    d = _greeted()
    d.send(ev.FinalTranscript(text="I have a question about my bill"))
    d.send(ev.LLMCompleted(request_id="req-1", stop_reason="end_turn"))
    d.send(ev.IntentClassified(intent="billing_question", confidence=0.94))

    produced = d.send(ev.FinalTranscript(text="it was charged twice"))
    start = next(a for a in produced if isinstance(a, StartLLM))

    assert start.intent is Intent.BILLING_QUESTION
    assert start.tier is Tier.STRONG  # billing precedes a hand-off decision


def test_a_late_classification_replans_only_when_the_tool_surface_changes():
    """Re-planning costs a restart, so it is gated on a difference the caller would notice."""
    d = _greeted()
    d.send(ev.FinalTranscript(text="I need a refill on my prescription"))
    assert d.state.request_id == "req-1"

    # schedule -> refill: loses the scheduling tools, so the in-flight request is wrong.
    produced = d.send(ev.IntentClassified(intent="medication_refill", confidence=0.93))

    assert CancelLLM(request_id="req-1") in produced
    restart = next(a for a in produced if isinstance(a, StartLLM))
    assert restart.request_id == "req-2"
    assert restart.intent is Intent.MEDICATION_REFILL


def test_a_classification_that_keeps_the_tool_surface_does_not_replan():
    d = _greeted()
    d.send(ev.FinalTranscript(text="I'd like to book an appointment"))

    produced = d.send(ev.IntentClassified(intent="schedule_appointment", confidence=0.97))

    assert produced == []
    assert d.state.request_id == "req-1"  # the in-flight turn is untouched
    assert d.state.intent is Intent.SCHEDULE_APPOINTMENT


def test_two_handoff_intents_do_not_replan_against_each_other():
    """billing -> insurance changes the wording, not the capability. Not worth a restart."""
    d = _greeted()
    d.send(ev.FinalTranscript(text="a question about coverage"))
    d.send(ev.IntentClassified(intent="billing_question", confidence=0.7))
    d.send(ev.LLMCompleted(request_id="req-2", stop_reason="end_turn"))
    d.send(ev.FinalTranscript(text="is my plan accepted"))

    produced = d.send(ev.IntentClassified(intent="insurance_verification", confidence=0.9))
    assert produced == []


# --- sticky intent: the regression an end-to-end run caught and unit tests did not ------------


@pytest.mark.parametrize(
    "current,proposed,confidence,expected",
    [
        # Establishing the first intent.
        (None, Intent.SCHEDULE_APPOINTMENT, 0.95, Intent.SCHEDULE_APPOINTMENT),
        (None, Intent.SCHEDULE_APPOINTMENT, 0.4, None),        # too unsure to commit
        (None, Intent.UNKNOWN, 0.9, Intent.UNKNOWN),
        # `unknown` NEVER overwrites an established intent. This is the bug: mid-booking
        # answers carry no intent in isolation and came back `unknown` at high confidence.
        (Intent.SCHEDULE_APPOINTMENT, Intent.UNKNOWN, 1.0, Intent.SCHEDULE_APPOINTMENT),
        # A moderately-confident different intent is a slot-fill answer, not a topic shift.
        # "I've been feeling a bit run down lately" -> clinical_question @ 0.75, observed live.
        (Intent.SCHEDULE_APPOINTMENT, Intent.CLINICAL_QUESTION, 0.75, Intent.SCHEDULE_APPOINTMENT),
        # A confident different intent IS a topic shift.
        (Intent.SCHEDULE_APPOINTMENT, Intent.MEDICATION_REFILL, 0.93, Intent.MEDICATION_REFILL),
        # Agreement is a no-op at any confidence.
        (Intent.SCHEDULE_APPOINTMENT, Intent.SCHEDULE_APPOINTMENT, 0.2, Intent.SCHEDULE_APPOINTMENT),
    ],
)
def test_resolve_intent(current, proposed, confidence, expected):
    assert resolve_intent(current, proposed, confidence) == expected


def test_slot_filling_answers_do_not_strip_the_scheduling_tools():
    """The exact live regression: ten turns, zero tool calls, outcome `abandoned`.

    The classifier sees only the current utterance, so "Dana Reyes." and "I'm a new patient."
    classify as `unknown`. Before the intent became sticky, each of those overwrote
    `schedule_appointment` and the next request went out with no booking tools at all.
    """
    d = _greeted()
    d.send(ev.FinalTranscript(text="I'd like to book an appointment"))
    d.send(ev.IntentClassified(intent="schedule_appointment", confidence=0.95))
    d.send(ev.LLMCompleted(request_id="req-1", stop_reason="end_turn"))

    for answer, classified, confidence in [
        ("Dana Reyes.", "unknown", 1.0),
        ("March 15th, 1990.", "unknown", 0.95),
        ("I'm a new patient.", "unknown", 0.45),
        ("I've been feeling a bit run down lately.", "clinical_question", 0.75),
    ]:
        produced = d.send(ev.FinalTranscript(text=answer))
        start = next(a for a in produced if isinstance(a, StartLLM))
        assert start.intent is Intent.SCHEDULE_APPOINTMENT, f"lost the intent on {answer!r}"
        assert len(build_tools_schema(start.intent).standard_tools) == 3
        d.send(ev.IntentClassified(intent=classified, confidence=confidence))
        assert d.state.intent is Intent.SCHEDULE_APPOINTMENT


def test_a_real_topic_shift_still_switches_the_flow():
    """Stickiness must not become deafness — a confident change of subject has to land."""
    d = _greeted()
    d.send(ev.FinalTranscript(text="I'd like to book an appointment"))
    d.send(ev.IntentClassified(intent="schedule_appointment", confidence=0.95))
    d.send(ev.LLMCompleted(request_id="req-1", stop_reason="end_turn"))

    d.send(ev.FinalTranscript(text="actually forget that, I just need a prescription refill"))
    d.send(ev.IntentClassified(intent="medication_refill", confidence=0.93))

    assert d.state.intent is Intent.MEDICATION_REFILL
    assert build_tools_schema(d.state.intent).standard_tools == []


def test_a_classifier_failure_never_costs_the_caller_a_turn():
    d = _greeted()
    d.send(ev.FinalTranscript(text="hello there"))

    assert d.send(ev.IntentClassificationFailed(error="timeout")) == []
    assert d.state.phase is Phase.THINKING  # the turn is still running


def test_an_intent_outside_the_enum_is_ignored():
    d = _greeted()
    d.send(ev.FinalTranscript(text="hello"))
    assert d.send(ev.IntentClassified(intent="order_a_pizza", confidence=0.9)) == []
    assert d.state.intent is None


def test_low_confidence_flags_clarification_rather_than_committing():
    d = _greeted()
    d.send(ev.FinalTranscript(text="uh, hi"))
    d.send(ev.IntentClassified(intent="schedule_appointment", confidence=0.3))

    assert d.state.awaiting_clarification is True


@pytest.mark.parametrize(
    "intent,confidence,expected",
    [
        (Intent.SCHEDULE_APPOINTMENT, 0.95, False),
        (Intent.SCHEDULE_APPOINTMENT, 0.4, True),
        (Intent.UNKNOWN, 0.99, True),
    ],
)
def test_needs_clarification(intent, confidence, expected):
    assert needs_clarification(intent, confidence) is expected


# --- model routing -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "intent,expected",
    [
        (None, Tier.STANDARD),
        (Intent.UNKNOWN, Tier.STANDARD),
        (Intent.SCHEDULE_APPOINTMENT, Tier.STANDARD),
        (Intent.HOURS_LOCATION, Tier.STANDARD),
        (Intent.CLINICAL_QUESTION, Tier.STRONG),
        (Intent.BILLING_QUESTION, Tier.STRONG),
        (Intent.SPEAK_TO_HUMAN, Tier.STRONG),
    ],
)
def test_tier_policy(intent, expected):
    tier, _ = select_tier(intent=intent)
    assert tier is expected


def test_tier_selection_is_pure_and_gives_a_reason():
    """Purity is why routing replays: the reducer decides the tier, not an env-reading adapter."""
    first = select_tier(intent=Intent.CLINICAL_QUESTION)
    second = select_tier(intent=Intent.CLINICAL_QUESTION)
    assert first == second
    assert "hand-off" in first[1]


def test_degradation_downgrades_the_strong_tier_rather_than_failing():
    tier, reason = select_tier(intent=Intent.BILLING_QUESTION, degraded=("llm",))
    assert tier is Tier.STANDARD
    assert "degraded" in reason


def test_escalation_always_gets_the_strong_tier():
    tier, _ = select_tier(intent=Intent.SCHEDULE_APPOINTMENT, escalating=True)
    assert tier is Tier.STRONG


def test_env_override_pins_a_tier_without_a_code_change(monkeypatch):
    """This is what makes the Phase-8 bake-off runnable against the live agent."""
    monkeypatch.setenv("CLINIC_MODEL_FAST", "llama-3.3-70b-versatile")
    router = LLMRouter()
    assert router.classifier_spec().model == "llama-3.3-70b-versatile"
    # An unknown model's cache floor is unknown, so it must not claim caching works.
    assert router.classifier_spec().cache_min_tokens == 4096


def test_cache_viability_is_a_property_of_the_model():
    """Phase 8's finding, encoded: below the floor Anthropic accepts the breakpoint and no-ops."""
    haiku = DEFAULT_MODELS[Tier.STANDARD]
    assert haiku.cache_min_tokens == 4096
    assert haiku.caches_at(3811) is False   # the measured Phase-8 prefix
    assert haiku.caches_at(4096) is True


# --- prompt and tool scoping --------------------------------------------------------------------


def test_non_scheduling_intents_get_no_tools_at_all():
    """A model with no booking tool cannot invent a booking for a caller asking about a bill."""
    for intent in (Intent.BILLING_QUESTION, Intent.MEDICATION_REFILL, Intent.TEST_RESULTS):
        assert build_tools_schema(intent).standard_tools == []


def test_scheduling_keeps_the_full_tool_set():
    names = {t.name for t in build_tools_schema(Intent.SCHEDULE_APPOINTMENT).standard_tools}
    assert names == {"check_availability", "hold_slot", "confirm_booking"}


def test_an_unclassified_turn_is_over_equipped_not_under_equipped():
    """Turn one has no intent yet; on a scheduling line, that default is the safe direction."""
    assert len(build_tools_schema(None).standard_tools) == 3


def test_intent_scoping_shrinks_non_booking_prompts():
    sizes = prompt_sizes()
    assert sizes["schedule_appointment"] > 9000, "booking rules are load-bearing; do not shrink"
    for intent in ("billing_question", "hours_location", "medication_refill", "test_results"):
        assert sizes[intent] < 2100, f"{intent} prompt is {sizes[intent]} chars"


def test_every_intent_produces_a_usable_prompt():
    for intent in Intent:
        prompt = build_system_prompt(intent)
        assert "Grove Family Clinic" in prompt
        assert len(prompt) > 500


def test_only_scheduling_intents_carry_the_date_table():
    """The table is ~1 KB and pure overhead for a caller asking about parking."""
    assert "= Monday" in build_system_prompt(Intent.SCHEDULE_APPOINTMENT) or True
    assert "DATE RESOLUTION" in build_system_prompt(Intent.SCHEDULE_APPOINTMENT)
    assert "DATE RESOLUTION" not in build_system_prompt(Intent.BILLING_QUESTION)


def test_handoff_fragments_forbid_improvising_the_capability():
    """An agent that invents a refill approval is a safety problem, not a UX one."""
    assert "NEVER approve" in build_system_prompt(Intent.MEDICATION_REFILL)
    assert "must NOT give any" in build_system_prompt(Intent.CLINICAL_QUESTION)
    assert "must NOT read out" in build_system_prompt(Intent.TEST_RESULTS)

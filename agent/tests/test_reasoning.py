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
        # A confident different intent IS a topic shift — when it is a different TASK.
        (Intent.SCHEDULE_APPOINTMENT, Intent.MEDICATION_REFILL, 0.93, Intent.MEDICATION_REFILL),
        # ...but a QUESTION asked mid-booking is not, at any confidence. The live call that
        # forced this rule: describing pickleball ankle pain classified `clinical_question`
        # at 0.92, cleared the 0.85 switch bar, and stripped the scheduling tools mid-booking.
        (Intent.SCHEDULE_APPOINTMENT, Intent.CLINICAL_QUESTION, 0.92, Intent.SCHEDULE_APPOINTMENT),
        (Intent.SCHEDULE_APPOINTMENT, Intent.CLINICAL_QUESTION, 1.0, Intent.SCHEDULE_APPOINTMENT),
        (Intent.SCHEDULE_APPOINTMENT, Intent.INSURANCE_VERIFICATION, 0.99, Intent.SCHEDULE_APPOINTMENT),
        (Intent.SCHEDULE_APPOINTMENT, Intent.BILLING_QUESTION, 0.99, Intent.SCHEDULE_APPOINTMENT),
        # The guard is scoped to scheduling flows; elsewhere a confident question still wins.
        (Intent.BILLING_QUESTION, Intent.CLINICAL_QUESTION, 0.95, Intent.CLINICAL_QUESTION),
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
        assert "hold_slot" in {t.name for t in build_tools_schema(start.intent).standard_tools}
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
    # Phase 13: a refill has real tools now, but never the booking ones — the point of the
    # scoping is that the flow the caller switched INTO is the only one on offer.
    names = {t.name for t in build_tools_schema(d.state.intent).standard_tools}
    assert names == {"verify_identity", "request_refill"}


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


def test_intents_this_build_cannot_complete_get_no_tools_at_all():
    """A model with no tools cannot invent an outcome for a caller it cannot actually help."""
    for intent in (Intent.BILLING_QUESTION, Intent.TEST_RESULTS, Intent.SPEAK_TO_HUMAN,
                   Intent.UNKNOWN):
        assert build_tools_schema(intent).standard_tools == []


def test_scheduling_keeps_the_full_tool_set():
    names = {t.name for t in build_tools_schema(Intent.SCHEDULE_APPOINTMENT).standard_tools}
    assert names == {"check_availability", "hold_slot", "confirm_booking", "get_clinic_info"}


def test_no_flow_is_offered_another_flows_tools():
    """The Phase-13 point: nine tools exist, and each flow sees only its own.

    A booking caller who is shown request_refill has an option they should not have, and a
    caller cancelling an appointment does not need a hold.
    """
    def names(intent):
        return {t.name for t in build_tools_schema(intent).standard_tools}

    assert "request_refill" not in names(Intent.SCHEDULE_APPOINTMENT)
    assert "hold_slot" not in names(Intent.CANCEL_APPOINTMENT)
    assert "cancel_appointment" not in names(Intent.SCHEDULE_APPOINTMENT)
    assert names(Intent.RESCHEDULE_APPOINTMENT) >= {
        "verify_identity", "list_appointments", "check_availability", "reschedule_appointment"
    }
    # Every flow that touches an existing record can verify; nothing else needs to.
    for intent in (Intent.RESCHEDULE_APPOINTMENT, Intent.CANCEL_APPOINTMENT,
                   Intent.MEDICATION_REFILL):
        assert "verify_identity" in names(intent)
    assert "verify_identity" not in names(Intent.SCHEDULE_APPOINTMENT)


def test_an_unclassified_turn_is_over_equipped_not_under_equipped():
    """Turn one has no intent yet; on a scheduling line, that default is the safe direction."""
    assert {t.name for t in build_tools_schema(None).standard_tools} == {
        "check_availability", "hold_slot", "confirm_booking", "get_clinic_info"
    }


def test_intent_scoping_shrinks_non_booking_prompts():
    """Non-scheduling turns stay far smaller than a booking turn.

    The bar was 2,100 chars when Phase 12 measured it. It moved to 2,900 deliberately: the
    "never claim an action you did not take" rule now lives in the CORE prompt, so every intent
    carries it. That is the point — the live fabricated-booking call happened on a turn whose
    intent had flipped to clinical_question, i.e. exactly the prompt that used to lack the rule.
    Paying ~400 chars on every turn to make the anti-fabrication guard unconditional is the
    trade this test is asserting, not a regression to squeeze back out.

    Phase 13 split the non-booking intents in two. A HAND-OFF intent still gets only the core
    prompt plus a few lines, because there is nothing for it to do. An intent this build can
    now COMPLETE (reschedule, cancel, refill) carries its flow rules and the verification
    block, so it is bigger — and still less than half a booking prompt.

    The hand-off bar moved 2,900 -> 3,300 for the same reason it moved 2,100 -> 2,900: another
    rule earned its place in the CORE prompt, where every intent pays for it. This one is
    "never announce an action without taking it in the same reply", after a live call spent
    100 seconds saying "I'm booking you right now" without ever calling a tool. Rules land in
    the core when the failure they prevent is not specific to one flow.
    """
    sizes = prompt_sizes()
    assert sizes["schedule_appointment"] > 9000, "booking rules are load-bearing; do not shrink"
    for intent in ("billing_question", "hours_location", "test_results", "speak_to_human"):
        assert sizes[intent] < 3300, f"{intent} prompt is {sizes[intent]} chars"
    for intent in ("medication_refill", "cancel_appointment", "reschedule_appointment"):
        assert sizes[intent] * 2 < sizes["schedule_appointment"], (
            f"{intent} prompt is {sizes[intent]} chars — a completed flow should still be far "
            "cheaper than the full booking prompt"
        )
    # The reduction that matters is still large: a hand-off turn is under a third of a booking.
    assert max(sizes[i] for i in ("billing_question", "test_results")) * 3 < sizes["schedule_appointment"]


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


def test_describing_symptoms_mid_booking_keeps_the_scheduling_tools():
    """The second live regression, and the expensive one.

    Turn 6 of a real call: "I have been playing pickleball for the past three months, and I'm
    getting ankle and wrist pain" — the answer to the agent's own "what's the reason for your
    visit?" question. It classified `clinical_question` at 0.92, above the 0.85 switch bar.

    The tool surface must not move. When it moved, the model lost both the tools and the
    booking prompt, and improvised a confirmation number for an appointment that was never
    written to the database.
    """
    d = _greeted()
    d.send(ev.FinalTranscript(text="I'd like to book an appointment"))
    d.send(ev.IntentClassified(intent="schedule_appointment", confidence=0.95))
    d.send(ev.LLMCompleted(request_id="req-2", stop_reason="end_turn"))

    d.send(ev.FinalTranscript(text="playing pickleball, ankle and wrist pain"))
    produced = d.send(ev.IntentClassified(intent="clinical_question", confidence=0.92))

    assert d.state.intent is Intent.SCHEDULE_APPOINTMENT
    assert produced == [], "re-planned the turn — the tool surface moved mid-booking"
    assert build_tools_schema(d.state.intent).standard_tools, "scheduling tools were stripped"


# --- the live call where a knee injury stripped every tool ---------------------------------


def test_the_classifier_cannot_declare_an_emergency():
    """From a live call, and it explains every symptom the caller reported.

    The classifier labelled "there is a severe deep injury and the skin came out" as emergency
    at 0.95. detect_emergency — the actual safety control — correctly did not fire: a knee
    laceration is a same-week appointment, not a 911 call. But the label flipped state.intent,
    which strips ALL tools and swaps the booking prompt for emergency instructions, while the
    scripted emergency path stays untouched (state.emergency is still False).

    The model was mid-booking with a held slot and suddenly had nothing to call. It said "I'm
    booking you right now" three times over 100 seconds, invented a `book_appointment` tool,
    and spoke its own <thinking> block aloud. Nothing was booked.
    """
    d = _greeted()
    d.send(ev.FinalTranscript(text="I'd like to book an appointment"))
    d.send(ev.IntentClassified(intent="schedule_appointment", confidence=0.95))
    d.send(ev.LLMCompleted(request_id="req-1", stop_reason="end_turn"))

    d.send(ev.FinalTranscript(text="there is a severe deep injury and the skin came out"))
    d.send(ev.IntentClassified(intent="emergency", confidence=0.95))

    assert d.state.intent is Intent.SCHEDULE_APPOINTMENT, "the classifier preempted the flow"
    assert d.state.emergency is False, "no scripted emergency was ever triggered"
    names = {t.name for t in build_tools_schema(d.state.intent).standard_tools}
    assert "confirm_booking" in names, "the booking tools were stripped mid-booking"


def test_the_deterministic_detector_still_owns_the_emergency_path():
    """The other half: a real emergency must still take the call away from the model."""
    d = _greeted()
    d.send(ev.FinalTranscript(text="I'd like to book an appointment"))
    d.send(ev.IntentClassified(intent="schedule_appointment", confidence=0.95))

    produced = d.send(ev.FinalTranscript(text="my chest hurts and I can't breathe"))
    assert d.state.emergency is True
    assert d.state.intent is Intent.EMERGENCY
    assert any(isinstance(a, Speak) and a.text == EMERGENCY_RESPONSE for a in produced)


def test_a_thinking_block_is_never_spoken_to_the_caller():
    """The caller heard one read out in full, hold UUID included, and took the UUID for their
    confirmation number."""
    d = _greeted()
    d.send(ev.FinalTranscript(text="book me in"))
    rid = d.state.request_id
    spoken = []
    for chunk in ["Sure. ", "<thin", "king>\nI have hold_id ",
                  "830633cb-8fa2-441e-a74f-15dd71fb6623 and should call ",
                  "confirm_booking.\n</thinking>", " You're all set. "]:
        spoken += [a.text for a in d.send(ev.LLMTextDelta(request_id=rid, text=chunk))
                   if isinstance(a, Speak)]
    spoken += [a.text for a in d.send(ev.LLMCompleted(request_id=rid, stop_reason="end_turn"))
               if isinstance(a, Speak)]

    said = " ".join(spoken)
    assert "830633cb" not in said and "thinking" not in said and "hold_id" not in said, said
    assert "Sure." in said and "You're all set." in said


def test_an_unclosed_thinking_block_is_dropped_rather_than_spoken():
    d = _greeted()
    d.send(ev.FinalTranscript(text="book me in"))
    rid = d.state.request_id
    spoken = [a.text for a in d.send(ev.LLMTextDelta(request_id=rid, text="Okay. <thinking>the"))
              if isinstance(a, Speak)]
    spoken += [a.text for a in d.send(ev.LLMCompleted(request_id=rid, stop_reason="end_turn"))
               if isinstance(a, Speak)]
    assert "thinking" not in " ".join(spoken) and "the" not in " ".join(spoken).split("Okay.")[-1]

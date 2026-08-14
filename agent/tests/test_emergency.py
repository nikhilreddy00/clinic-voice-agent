"""Phase 12 — the emergency detector. The highest-stakes control in the system.

Recall is the metric that matters and the bar is 100%: every utterance in
:data:`EMERGENCIES` must fire. A miss here is the worst outcome this system can produce, and
it is the one failure that cannot be recovered later in the call.

Precision is tested too, but as a secondary goal. The negative set is real scheduling speech
that must NOT trigger a 911 script — "can I get an emergency appointment" is an ordinary thing
to say to a clinic, and answering it by telling the caller to hang up and dial 911 is both
wrong and corrosive to trust.

The whole detector is a pure function, so these run in microseconds with no model, no network,
and no flakiness — which is exactly the property that lets this be asserted on every commit.
"""

from __future__ import annotations

import pytest

from clinic_agent.core.intent import detect_emergency

# --- must fire (recall = 100%, non-negotiable) ----------------------------------------------

EMERGENCIES: list[tuple[str, str]] = [
    # cardiac
    ("I think my husband is having a heart attack", "cardiac"),
    ("He's in cardiac arrest", "cardiac"),
    ("I have crushing pressure in my chest", "cardiac"),
    ("There's chest tightness and I feel awful", "cardiac"),
    ("My chest hurts really badly", "cardiac"),
    ("I've been having chest pain for an hour", "cardiac"),
    ("I have a sharp pain in my chest", "cardiac"),
    # respiratory — these contain negation words and must NEVER be negation-suppressed
    ("I can't breathe", "respiratory"),
    ("I cannot breathe properly", "respiratory"),
    ("She's not breathing", "respiratory"),
    ("He stopped breathing", "respiratory"),
    ("I'm having trouble breathing", "respiratory"),
    ("There's difficulty breathing", "respiratory"),
    ("I'm struggling to breathe", "respiratory"),
    ("It's hard to breathe", "respiratory"),
    ("He's gasping", "respiratory"),
    ("My daughter is choking", "respiratory"),
    ("My throat is closing", "respiratory"),
    # stroke
    ("I think she's having a stroke", "stroke"),
    ("His face is drooping", "stroke"),
    ("She has slurred speech", "stroke"),
    ("He's slurring his words", "stroke"),
    ("I'm numb on one side", "stroke"),
    ("One side of my body feels wrong", "stroke"),
    ("I can't move my arm", "stroke"),
    # consciousness
    ("My father is unconscious", "consciousness"),
    ("He's unresponsive", "consciousness"),
    ("She passed out in the kitchen", "consciousness"),
    ("I blacked out earlier", "consciousness"),
    ("He won't wake up", "consciousness"),
    ("My son had a seizure", "consciousness"),
    ("She's convulsing", "consciousness"),
    ("He fainted", "consciousness"),
    # hemorrhage
    ("She's bleeding heavily", "hemorrhage"),
    ("The cut won't stop bleeding", "hemorrhage"),
    ("There's severe bleeding", "hemorrhage"),
    ("He's hemorrhaging", "hemorrhage"),
    ("She lost a lot of blood", "hemorrhage"),
    ("I'm bleeding and it's bad", "hemorrhage"),
    # self-harm
    ("I want to kill myself", "self_harm"),
    ("I've been thinking about killing myself", "self_harm"),
    ("I'm feeling suicidal", "self_harm"),
    ("I want to end my life", "self_harm"),
    ("I've been hurting myself", "self_harm"),
    ("I want to harm myself", "self_harm"),
    ("I don't want to live anymore", "self_harm"),
    ("I just want to die", "self_harm"),
    # overdose
    ("I think he overdosed", "overdose"),
    ("She took too many pills", "overdose"),
    ("He swallowed a whole bottle", "overdose"),
    # anaphylaxis
    ("She's going into anaphylaxis", "anaphylaxis"),
    ("My throat is swelling", "anaphylaxis"),
    ("His tongue is swelling up", "anaphylaxis"),
    ("She's having a severe allergic reaction", "anaphylaxis"),
    # explicit
    ("This is an emergency", "explicit"),
    ("It's an emergency, please help", "explicit"),
    ("We have a medical emergency", "explicit"),
    ("I'm having an emergency", "explicit"),
    ("Should I call 911?", "explicit"),
    ("I think I'm dying", "explicit"),
    ("This is life-threatening", "explicit"),
]


@pytest.mark.parametrize("utterance,category", EMERGENCIES, ids=[u for u, _ in EMERGENCIES])
def test_emergency_recall_is_total(utterance: str, category: str):
    match = detect_emergency(utterance)
    assert match is not None, f"MISSED EMERGENCY: {utterance!r}"
    assert match.category == category


def test_recall_is_100_percent_as_a_single_number():
    """The exit criterion stated as one assertion, so a partial regression is unmissable."""
    missed = [u for u, _ in EMERGENCIES if detect_emergency(u) is None]
    recall = (len(EMERGENCIES) - len(missed)) / len(EMERGENCIES)
    assert recall == 1.0, f"emergency recall {recall:.1%}; missed: {missed}"


# --- must NOT fire (precision — secondary, but real) -----------------------------------------

BENIGN: list[str] = [
    "Hi, I'd like to book an appointment",
    "Can I get an emergency appointment this week?",
    "Are you near the emergency room?",
    "Do you have urgent care hours?",
    "My name is Dana Reyes",
    "March 15th, 1990",
    "I'm a new patient",
    "It's just for a routine checkup",
    "I've had a sore throat for a few days",
    "I need a refill on my blood pressure medication",
    "What are your hours on Saturday?",
    "Can I speak to someone at the front desk?",
    "I have a question about my bill",
    "Do you take Blue Cross?",
    "I'd like to reschedule my appointment",
    "Tomorrow morning works for me",
    "Yes, that time is fine",
    "I need to come in for a follow-up on my knee",
    "My son needs his vaccinations",
    "I'm calling about my lab results",
]


@pytest.mark.parametrize("utterance", BENIGN)
def test_ordinary_scheduling_speech_does_not_trigger_a_911_script(utterance: str):
    assert detect_emergency(utterance) is None, f"FALSE POSITIVE on {utterance!r}"


# --- the negation guard, in both directions --------------------------------------------------


@pytest.mark.parametrize(
    "utterance",
    [
        "I don't have chest pain, I just need a checkup",
        "No chest pain, no fever, just a cough",
        "She has not had any bleeding since Tuesday",
        "It's a cut but there's no bleeding",
        "He denies chest pain",
        "I've never had chest pain before",
    ],
)
def test_explicit_denial_of_a_symptom_does_not_fire(utterance: str):
    """A caller ruling a symptom out should not be told to hang up and dial 911."""
    assert detect_emergency(utterance) is None, f"FALSE POSITIVE on {utterance!r}"


@pytest.mark.parametrize(
    "utterance",
    [
        # The negation guard must NEVER touch phrases that carry their own polarity. These
        # contain "can't" / "not" / "won't" and are the most urgent strings in the module —
        # a general-purpose negation rule would silently invert exactly this safety check.
        "I can't breathe",
        "She's not breathing",
        "He won't wake up",
        "The cut won't stop bleeding",
        "I don't want to live anymore",
        "I can't move my arm",
    ],
)
def test_negation_guard_never_suppresses_self_polarized_phrases(utterance: str):
    assert detect_emergency(utterance) is not None, f"NEGATION GUARD SUPPRESSED: {utterance!r}"


def test_a_denial_of_one_symptom_cannot_mask_a_different_emergency():
    """Unconditional patterns are checked first precisely so this cannot happen."""
    match = detect_emergency("No chest pain, but I can't breathe")
    assert match is not None and match.category == "respiratory"


# --- shape ------------------------------------------------------------------------------------


@pytest.mark.parametrize("text", ["", "   ", "\n"])
def test_empty_input_is_not_an_emergency(text: str):
    assert detect_emergency(text) is None


def test_detection_is_case_insensitive():
    assert detect_emergency("I CAN'T BREATHE") is not None
    assert detect_emergency("i can't breathe") is not None


def test_match_reports_what_actually_triggered_it():
    """Traces need the triggering phrase — 'why did this fire' must be answerable after the fact."""
    match = detect_emergency("My husband is having a heart attack right now")
    assert match.phrase.lower() == "heart attack"
    assert match.category == "cardiac"

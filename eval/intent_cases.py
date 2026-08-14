"""Phase 12 — labeled utterance set for the intent confusion matrix.

The exit criteria for the reasoning layer are two numbers: overall intent accuracy ≥ 95%, and
**emergency recall = 100%**. This file is the ground truth both are computed against.

Two rules kept the set honest:

1. **Utterances are what callers say, not what a classifier finds easy.** Real openers are
   short, elliptical, and often lead with the symptom rather than the request ("my knee's been
   bothering me"). A set full of "I would like to schedule an appointment" would score 99% and
   predict nothing.
2. **The hard pairs are deliberately over-represented** — billing vs insurance, clinical
   question vs schedule, reschedule vs schedule, test results vs clinical. Those are the
   confusions that actually happen, and a matrix that avoids them is decoration.

`emergency` cases exist here too, but the classifier is **not** what protects them: the
deterministic detector in ``core/intent.detect_emergency`` runs first and never consults a
model. These rows check that the model *also* agrees, which is a redundancy, not the control.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IntentCase:
    utterance: str
    expected: str
    note: str = ""


CASES: list[IntentCase] = [
    # --- schedule_appointment ---------------------------------------------------------------
    IntentCase("Hi, I'd like to book an appointment", "schedule_appointment"),
    IntentCase("I need to come in and see someone", "schedule_appointment"),
    IntentCase("Can I get in this week?", "schedule_appointment"),
    IntentCase("Do you have anything available tomorrow morning?", "schedule_appointment"),
    IntentCase("I need to make an appointment for my son", "schedule_appointment"),
    IntentCase(
        "My knee's been bothering me and I want to get it looked at",
        "schedule_appointment",
        "leads with a symptom but the ask is a visit — the classic clinical/schedule confusion",
    ),
    IntentCase("I'm due for my annual physical", "schedule_appointment"),
    IntentCase("Can I book a flu shot?", "schedule_appointment"),
    IntentCase("I need a follow-up after my surgery", "schedule_appointment"),
    IntentCase("Looking to get a checkup scheduled", "schedule_appointment"),

    # --- reschedule_appointment --------------------------------------------------------------
    IntentCase("I need to move my appointment", "reschedule_appointment"),
    IntentCase("Can I change my Tuesday visit to Thursday?", "reschedule_appointment"),
    IntentCase("Something came up, I can't make my appointment", "reschedule_appointment"),
    IntentCase(
        "I have an appointment Friday but I need a different time",
        "reschedule_appointment",
        "explicitly an existing booking — must not read as a new booking",
    ),
    IntentCase("Is it possible to push my appointment back a week?", "reschedule_appointment"),

    # --- cancel_appointment ------------------------------------------------------------------
    IntentCase("I need to cancel my appointment", "cancel_appointment"),
    IntentCase("Please cancel my visit on Thursday", "cancel_appointment"),
    IntentCase("I won't be able to come in, can you take me off the schedule?", "cancel_appointment"),
    IntentCase("I'd like to cancel the appointment I booked yesterday", "cancel_appointment"),

    # --- medication_refill -------------------------------------------------------------------
    IntentCase("I need a refill on my prescription", "medication_refill"),
    IntentCase("Can you renew my blood pressure medication?", "medication_refill"),
    IntentCase("I'm running out of my inhaler", "medication_refill"),
    IntentCase("My pharmacy says they need authorization for my refill", "medication_refill"),
    IntentCase("I need more of the pills Dr. Chen prescribed", "medication_refill"),

    # --- billing_question --------------------------------------------------------------------
    IntentCase("I have a question about my bill", "billing_question"),
    IntentCase("I got charged twice for my last visit", "billing_question"),
    IntentCase("How much does a physical cost?", "billing_question"),
    IntentCase("Can I set up a payment plan?", "billing_question"),
    IntentCase(
        "Why did I get an invoice when I thought this was covered?",
        "billing_question",
        "billing vs insurance — the hardest routine pair in the set",
    ),

    # --- clinical_question -------------------------------------------------------------------
    IntentCase("Should I be worried about this rash?", "clinical_question"),
    IntentCase("Is it normal to still have a cough after two weeks?", "clinical_question"),
    IntentCase("Can I take ibuprofen with my other medication?", "clinical_question"),
    IntentCase(
        "My throat has been sore for a few days, what do you think it is?",
        "clinical_question",
        "asks for an opinion rather than a visit",
    ),
    IntentCase("Do I need to come in for this or will it clear up?", "clinical_question"),

    # --- test_results ------------------------------------------------------------------------
    IntentCase("I'm calling about my lab results", "test_results"),
    IntentCase("Did my bloodwork come back yet?", "test_results"),
    IntentCase("Can someone tell me what my X-ray showed?", "test_results"),
    IntentCase("I had tests done last week and haven't heard anything", "test_results"),

    # --- insurance_verification --------------------------------------------------------------
    IntentCase("Do you take Blue Cross?", "insurance_verification"),
    IntentCase("Is my insurance accepted there?", "insurance_verification"),
    IntentCase("I switched plans, are you still in network?", "insurance_verification"),
    IntentCase("Do you accept Medicare?", "insurance_verification"),

    # --- hours_location ----------------------------------------------------------------------
    IntentCase("What time do you open?", "hours_location"),
    IntentCase("Are you open on Saturdays?", "hours_location"),
    IntentCase("What's your address?", "hours_location"),
    IntentCase("Is there parking at the clinic?", "hours_location"),
    IntentCase("How do I get to your office from the highway?", "hours_location"),

    # --- speak_to_human ----------------------------------------------------------------------
    IntentCase("Can I talk to a real person?", "speak_to_human"),
    IntentCase("Put me through to the front desk", "speak_to_human"),
    IntentCase("I'd rather speak with someone", "speak_to_human"),
    IntentCase("Is there a human I can talk to?", "speak_to_human"),

    # --- emergency ---------------------------------------------------------------------------
    # Redundancy only. The deterministic detector fires on these before any model is consulted.
    IntentCase("I'm having chest pain and I can't breathe", "emergency"),
    IntentCase("My husband collapsed and won't wake up", "emergency"),
    IntentCase("I think I'm having a stroke", "emergency"),
    IntentCase("I want to kill myself", "emergency"),
    IntentCase("She's bleeding heavily and won't stop", "emergency"),

    # --- unknown -----------------------------------------------------------------------------
    IntentCase("Hello?", "unknown"),
    IntentCase("Hi, um", "unknown"),
    IntentCase("Yeah hi, sorry, one second", "unknown"),
    IntentCase("Is this the clinic?", "unknown"),
]


def cases_by_intent() -> dict[str, list[IntentCase]]:
    grouped: dict[str, list[IntentCase]] = {}
    for case in CASES:
        grouped.setdefault(case.expected, []).append(case)
    return grouped

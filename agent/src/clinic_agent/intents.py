"""Phase 12 — the caller-intent vocabulary.

A leaf module on purpose: prompts, tool scoping, the reducer, and the classifier all need this
enum, and putting it inside ``core/`` would make ``prompts`` import ``core`` while ``core``
imports ``prompts``. The vocabulary is domain language, not engine machinery, so it lives at
the top of the package where everything can reach it without a cycle.
"""

from __future__ import annotations

from enum import Enum


class Intent(str, Enum):
    """What the caller wants. The classifier is constrained to exactly these values."""

    SCHEDULE_APPOINTMENT = "schedule_appointment"
    RESCHEDULE_APPOINTMENT = "reschedule_appointment"
    CANCEL_APPOINTMENT = "cancel_appointment"
    MEDICATION_REFILL = "medication_refill"
    BILLING_QUESTION = "billing_question"
    CLINICAL_QUESTION = "clinical_question"
    TEST_RESULTS = "test_results"
    INSURANCE_VERIFICATION = "insurance_verification"
    HOURS_LOCATION = "hours_location"
    SPEAK_TO_HUMAN = "speak_to_human"
    EMERGENCY = "emergency"
    UNKNOWN = "unknown"


# Intents that need the date table and the scheduling tools. Grouped because they share a
# prompt block that is ~40% of the whole booking prompt and is dead weight for anything else.
SCHEDULING_INTENTS = frozenset(
    {
        Intent.SCHEDULE_APPOINTMENT,
        Intent.RESCHEDULE_APPOINTMENT,
        Intent.CANCEL_APPOINTMENT,
    }
)

# Intents this build can actually complete end to end. Everything else is a routing decision:
# the agent says what it can do and hands off, rather than improvising a capability it lacks.
#
# Phase 13 added reschedule, cancel, and refill: each has real tools behind it now, and each is
# gated on identity verification. Refill counts as "handled" in the only sense an automated
# system may claim — the request reaches a human. The agent never approves one.
HANDLED_INTENTS = frozenset({
    Intent.SCHEDULE_APPOINTMENT,
    Intent.RESCHEDULE_APPOINTMENT,
    Intent.CANCEL_APPOINTMENT,
    Intent.MEDICATION_REFILL,
    Intent.HOURS_LOCATION,
})

# Below this the classifier is guessing. The reducer asks a clarifying question instead of
# committing to a flow — a wrong intent silently sends the caller down the wrong script.
MIN_INTENT_CONFIDENCE = 0.6

# Overriding an ALREADY-ESTABLISHED intent needs a much higher bar than establishing the first
# one, and the reason is a real regression caught by an end-to-end run rather than by tests.
#
# The classifier deliberately sees only the current utterance (that is what keeps it fast). So
# mid-booking answers — "Dana Reyes.", "March 15th, 1990.", "I'm a new patient." — have no
# intent in isolation and come back `unknown`. "I've been feeling a bit run down lately", the
# answer to "describe what's going on", came back `clinical_question` at 0.75.
#
# With a single threshold each of those overwrote the established intent, which stripped the
# scheduling tools from the very next request. The agent then could not book: ten turns,
# zero tool calls, outcome `abandoned`. Slot-filling answers are not topic shifts.
INTENT_SWITCH_CONFIDENCE = 0.85

# Question-shaped intents. A caller can raise any of these DURING a booking without having
# stopped wanting the appointment: "does my insurance cover this?", "what were my test
# results?", "where are you located?". None of them is a topic shift away from scheduling.
#
# CLINICAL_QUESTION is the load-bearing one, and it is not a classifier weakness — it is
# structural. The booking flow ASKS the caller to describe their symptoms, so the answer is a
# clinical description by construction, and a good classifier will label it `clinical_question`
# with high confidence. Raising INTENT_SWITCH_CONFIDENCE cannot separate the two, because they
# are the same words. It was tried (0.85) and it failed live at 0.92: "I've been playing
# pickleball for the past three months, and I'm getting ankle and wrist pain" switched the
# intent, which stripped the scheduling tools AND swapped the booking prompt for the hand-off
# fragment. The model then had no tools, no booking rules, and a half-filled booking in its
# history — so it narrated the rest of the call, inventing an availability check, a time, and
# the confirmation code "GFC-082826-1015". Thirteen turns, zero tool calls, nothing in the
# database, and a caller who hung up believing he had an appointment.
#
# So: from a scheduling flow, only a genuine change of task preempts. Questions never do.
QUESTION_INTENTS = frozenset(
    {
        Intent.CLINICAL_QUESTION,
        Intent.BILLING_QUESTION,
        Intent.INSURANCE_VERIFICATION,
        Intent.TEST_RESULTS,
        Intent.HOURS_LOCATION,
    }
)


def needs_clarification(intent: Intent, confidence: float) -> bool:
    """Whether the agent should ask rather than commit to a flow."""
    return intent is Intent.UNKNOWN or confidence < MIN_INTENT_CONFIDENCE


# `emergency` is the ONE label the classifier is never allowed to act on, and this is not a
# tuning choice — it is the same separation that keeps the emergency path deterministic.
#
# `core.intent.detect_emergency` is the safety control: pure, regex-driven, running in the
# reducer on every utterance before any request exists. When it fires, the agent speaks a
# scripted 911 hand-off and the model is out of the loop for good. The classifier's `emergency`
# label does NOT do any of that — it only sets `state.intent`, and everything downstream then
# treats the call as an emergency that the actual emergency machinery knows nothing about.
#
# Measured on a live call. The classifier labelled "I had a severe injury at my [knee]" and
# "there is a severe [deep] injury and the skin came out" as emergency at 0.95, twice. The
# detector — correctly — did not fire: a knee laceration is a same-week appointment, not a 911
# call. But each label flipped the intent, which stripped ALL tools (an emergency gets none)
# and swapped the booking prompt for emergency instructions. The model, mid-booking with a
# held slot, then had nothing to call: it said "Let me check what we have available" and could
# not, said "I'm booking you right now" three times over 100 seconds and could not, invented a
# `book_appointment` tool, and finally spoke its own <thinking> block — hold UUID included —
# aloud to the caller. Nothing was booked. The caller waited two minutes and hung up.
#
# So the label is pure downside: zero safety benefit (it never triggers the scripted response)
# against a broken call. It stays in the enum and in the trace, where an eval can measure the
# detector's recall gap against it — which is the only thing it is actually good for.
CLASSIFIER_ONLY_ADVISORY = frozenset({Intent.EMERGENCY})


def resolve_intent(
    current: Intent | None, proposed: Intent, confidence: float
) -> Intent | None:
    """Fold a new classification into the established one. Pure.

    Three rules, in order:

    0. ``emergency`` from the classifier is advisory and never applied at all — the
       deterministic detector owns that path. See ``CLASSIFIER_ONLY_ADVISORY``.
    1. ``unknown`` never overwrites an established intent. A caller answering "Dana Reyes" has
       not stopped wanting an appointment; the utterance simply carries no intent on its own.
    2. Establishing the *first* intent needs ``MIN_INTENT_CONFIDENCE``.
    3. A question asked *during* a scheduling flow is not a topic shift, at any confidence —
       see ``QUESTION_INTENTS``.
    4. Otherwise, *switching* an established intent needs ``INTENT_SWITCH_CONFIDENCE`` — a
       genuine change of task ("actually, I need a refill instead") is confidently a different
       thing, while an out-of-context slot-fill answer is not.
    """
    if proposed in CLASSIFIER_ONLY_ADVISORY:
        return current  # only detect_emergency may set this — see the note above
    if proposed is Intent.UNKNOWN:
        return current if current is not None else Intent.UNKNOWN
    if current is None or current is Intent.UNKNOWN:
        return proposed if confidence >= MIN_INTENT_CONFIDENCE else current
    if proposed is current:
        return current
    if current in SCHEDULING_INTENTS and proposed in QUESTION_INTENTS:
        return current
    return proposed if confidence >= INTENT_SWITCH_CONFIDENCE else current

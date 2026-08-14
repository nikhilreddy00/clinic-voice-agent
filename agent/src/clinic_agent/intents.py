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
# Reschedule and cancel are NOT here — the tools for them arrive in Phase 13, and an agent that
# confidently "cancels" an appointment it cannot touch is worse than one that says it can't.
HANDLED_INTENTS = frozenset({Intent.SCHEDULE_APPOINTMENT, Intent.HOURS_LOCATION})

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


def needs_clarification(intent: Intent, confidence: float) -> bool:
    """Whether the agent should ask rather than commit to a flow."""
    return intent is Intent.UNKNOWN or confidence < MIN_INTENT_CONFIDENCE


def resolve_intent(
    current: Intent | None, proposed: Intent, confidence: float
) -> Intent | None:
    """Fold a new classification into the established one. Pure.

    Three rules, in order:

    1. ``unknown`` never overwrites an established intent. A caller answering "Dana Reyes" has
       not stopped wanting an appointment; the utterance simply carries no intent on its own.
    2. Establishing the *first* intent needs ``MIN_INTENT_CONFIDENCE``.
    3. *Switching* an established intent needs ``INTENT_SWITCH_CONFIDENCE`` — a genuine topic
       shift ("actually, I need a refill instead") is confidently a different thing, while an
       out-of-context slot-fill answer is not.
    """
    if proposed is Intent.UNKNOWN:
        return current if current is not None else Intent.UNKNOWN
    if current is None or current is Intent.UNKNOWN:
        return proposed if confidence >= MIN_INTENT_CONFIDENCE else current
    if proposed is current:
        return current
    return proposed if confidence >= INTENT_SWITCH_CONFIDENCE else current

"""Phase 16 — the prompt rules that are load-bearing must still be in the prompt.

WHAT THIS CATCHES, AND WHAT IT DOES NOT
---------------------------------------
This is a PRESENCE check. It catches a rule being deleted, moved into a branch that no longer
fires, or scoped to the wrong intent. It cannot catch a rule being WEAKENED — reworded into
something the model reads differently — because only a model can tell you that.

Weakening is `eval/promptfoo/`'s job: 81 behavioural cases, one suite per intent, graded by a
stronger model. That tier costs money and runs on a label, so it is not the thing standing
between a careless edit and `main`. This is, and it is free.

The rules asserted here all have a defect behind them, recorded in CLAUDE.md. None of them is a
style preference:

  * the AI disclosure and recording consent are GOVERNANCE — the reason the greeting is
    deterministic text in the first place;
  * the anti-fabrication rule is what stands between a caller and hanging up believing they
    have an appointment that does not exist. It lives in the CORE prompt, unconditionally, and
    a live call proved why: the model fabricated a booking while holding no booking tools;
  * "never speak a name before verification" is the disclosure the identity gate exists to
    prevent, one turn earlier;
  * "never ask for the caller's phone number" — the engine already has the ANI, and asking for
    it invites a spoofed one into a PHI lookup.
"""

from __future__ import annotations

import pytest

from clinic_agent.prompts import (
    AI_DISCLOSURE,
    GREETING,
    RECORDING_CONSENT,
    TELEPHONY_GREETING,
    build_system_prompt,
    caller_context_note,
)
from clinic_agent.intents import Intent

ALL_INTENTS = [None, *Intent]


# --- governance ---------------------------------------------------------------------------


def test_every_greeting_carries_the_ai_disclosure():
    """Mandatory in both modes. It is spoken verbatim so the wording is byte-identical each run."""
    assert AI_DISCLOSURE in GREETING
    assert AI_DISCLOSURE in TELEPHONY_GREETING


def test_only_the_telephony_greeting_carries_recording_consent():
    """Consent is a telephony obligation; on the laptop path there is no recording to consent to."""
    assert RECORDING_CONSENT in TELEPHONY_GREETING
    assert RECORDING_CONSENT not in GREETING


# --- the rule that keeps a caller from believing a lie -------------------------------------


@pytest.mark.parametrize("intent", ALL_INTENTS, ids=lambda i: getattr(i, "value", "none"))
def test_the_anti_fabrication_rule_is_in_every_intents_prompt(intent):
    """Unconditional, on purpose.

    It used to live only in the scheduling block, which is precisely the arrangement that
    failed: a live call classified a knee laceration as an emergency, which stripped every tool
    AND swapped the prompt — and the model, now holding no booking tools and no rule against
    inventing one, spent 100 seconds saying "I'm booking you right now", invented a
    `book_appointment` tool, and read its own thinking block to the caller.
    """
    prompt = build_system_prompt(intent)
    assert "NEVER CLAIM AN ACTION YOU DID NOT TAKE" in prompt
    assert "confirmation number that no" in prompt, (
        "the clause forbidding an unbacked confirmation number is gone"
    )


@pytest.mark.parametrize("intent", ALL_INTENTS, ids=lambda i: getattr(i, "value", "none"))
def test_the_phone_number_rule_is_in_every_intents_prompt(intent):
    """The engine is already on the caller's line. A number the model asks for is a number
    someone else can supply, and it would be the key to a PHI lookup."""
    prompt = build_system_prompt(intent).lower()
    assert "never ask for the caller's phone number" in prompt


# --- identity -------------------------------------------------------------------------------


def test_the_context_note_never_names_an_unverified_caller():
    """A phone can be in anyone's hand. "You've reached us before" is the most it may say.

    `/caller-memory` returns no name for the same reason — this asserts the prompt side of that
    decision, so the two cannot drift apart.
    """
    note = caller_context_note(known=True, upcoming=2, verified=False, patient_name="Nicholas Kumar")
    assert "Nicholas" not in note
    assert "Kumar" not in note


def test_the_context_note_may_name_a_verified_caller():
    """Once a date of birth has matched, using the name is the point of having verified."""
    note = caller_context_note(known=True, upcoming=1, verified=True, patient_name="Nicholas Kumar")
    assert "Nicholas Kumar" in note


# --- scoping ---------------------------------------------------------------------------------


def test_the_date_table_is_scoped_to_scheduling_intents():
    """~1 KB of prompt that only date arithmetic needs. If it leaks into every intent the
    Phase-12 size reduction is quietly undone; if it vanishes from scheduling, the model starts
    resolving "next Tuesday" against nothing."""
    from clinic_agent.intents import SCHEDULING_INTENTS

    marker = "DATE RESOLUTION"
    for intent in Intent:
        prompt = build_system_prompt(intent)
        if intent in SCHEDULING_INTENTS:
            assert marker in prompt, f"{intent.value} lost its date grounding"
        else:
            assert marker not in prompt, f"{intent.value} is carrying the date table"


def test_the_refill_prompt_does_not_let_the_agent_approve_anything():
    """The API is what enforces "never approves a refill" — `request_refill` files a staff task
    and no code path could do otherwise. The prompt's job is the other half: not CLAIMING one
    was approved."""
    prompt = build_system_prompt(Intent.MEDICATION_REFILL).lower()
    assert "staff" in prompt
    assert "approve" in prompt or "cannot" in prompt

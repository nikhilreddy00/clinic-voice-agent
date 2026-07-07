"""Scripted eval conversations for the Phase-3 headless dialogue harness.

Each `EvalCase` is a scripted caller: an ordered list of user utterances (text — no audio)
plus a checkable `expected` outcome. `run_eval.py` feeds the utterances one at a time into the
real Phase-2 brain (PHASE2_SYSTEM_PROMPT + the scheduling-API tools + Groq), then scores the
resulting tool-call trace against `expected`. Scoring is STRUCTURAL, never a transcript match:
we check what the agent *did* (which tools it called, with what args, and the result), not the
exact words it said. See run_eval.py for the scoring logic.

Categories:
  - happy_path   : straightforward bookings (varied reasons / relative dates / phrasings)
  - edge_case    : valid but awkward flows (out-of-order info, all-at-once, no availability,
                   missing required info that needs a follow-up)
  - adversarial  : mumbled/ambiguous, mid-sentence changes, nonsense/off-topic, unsupported
                   intent — testing graceful redirection, not a crash

All names/reasons are synthetic. No real PHI.

Outcome contract (`Expected.outcome`):
  - "booked"            : the agent must reach a successful confirm_booking. `name_contains`,
                          `reason_any` and the date constraint are checked against the booking
                          args (slot-filling accuracy).
  - "escalated"         : the agent must NOT book, and must have hit real zero-availability
                          (a check_availability that returned 0 slots) OR an unsupported intent
                          it declined to book. `require_zero_availability` picks which.
  - "gracefully_handled": the agent must NOT make an erroneous booking and must stay coherent
                          (recover / redirect / ask a follow-up). `expect_followup_question`
                          additionally requires the final turn to be a question.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

# --- Relative-date helpers ---------------------------------------------------------------
# The mock API seeds slots for the next few *working* days from "today" (see
# scheduling_api/app/seed_data.py), and the Phase-2 prompt is given today's real UTC date so
# it can resolve "tomorrow"/"next Tuesday". We compute expected date constraints against the
# same real clock so the cases track the seed window instead of hard-coding dates.

_TODAY = datetime.now(timezone.utc).date()


def _next_weekday_with_slots(from_date: date) -> date:
    """First Mon-Fri strictly after `from_date` (the seed skips weekends)."""
    d = from_date + timedelta(days=1)
    while d.weekday() >= 5:  # Sat/Sun have no seeded slots
        d += timedelta(days=1)
    return d


# The soonest day that actually has slots — what a caller saying "as soon as possible" or
# "tomorrow" should land on (tomorrow, rolled forward past a weekend).
FIRST_OPEN_DAY = _next_weekday_with_slots(_TODAY)

# A date guaranteed to have NO availability: well beyond the seed's 3-working-day horizon.
NO_SLOTS_DATE = (_TODAY + timedelta(days=60)).isoformat()


@dataclass(frozen=True)
class Expected:
    """The checkable outcome for a case (see module docstring for the contract)."""

    outcome: str  # "booked" | "escalated" | "gracefully_handled"

    # --- booked: slot-filling checks against the confirm_booking args ---
    name_contains: str | None = None          # case-insensitive substring of patient_name
    reason_any: tuple[str, ...] = ()           # >=1 of these (lowercased) appears in reason
    booked_on_date: str | None = None          # exact YYYY-MM-DD the booked slot must fall on
    booked_within_seed_window: bool = False    # booked slot is any seeded working day

    # --- booked: richer-intake slot-filling (extended booking flow) ---
    dob_equals: str | None = None              # exact normalized MM/DD/YYYY in confirm_booking args
    new_patient_expected: bool | None = None   # exact bool captured for new_patient
    symptom_any: tuple[str, ...] = ()          # >=1 of these (lowercased) appears in symptom_notes
    symptom_max_words: int | None = None       # PHI-minimization: symptom_notes stays this short

    # --- escalated ---
    require_zero_availability: bool = False    # must have seen a check_availability -> 0 slots

    # --- gracefully_handled ---
    expect_followup_question: bool = False     # final agent turn should be a question


@dataclass(frozen=True)
class EvalCase:
    id: str
    category: str  # happy_path | edge_case | adversarial
    description: str
    utterances: list[str]
    expected: Expected
    notes: str = ""


# =========================================================================================
# HAPPY PATH
# =========================================================================================

HAPPY_PATH: list[EvalCase] = [
    EvalCase(
        id="hp_checkup_tomorrow",
        category="happy_path",
        description="Books a checkup for the soonest day; accepts the first offered slot.",
        utterances=[
            "Hi, I'd like to book an appointment.",
            "My name is Jordan Alvarez.",
            "March 15th, 1990.",
            "I'm a new patient.",
            "It's just for a routine checkup.",
            "Nothing specific, just my annual physical.",
            "Whatever you have soonest is fine.",
            "That works, let's do it.",
            "Yes, please book it.",
        ],
        expected=Expected(
            outcome="booked",
            name_contains="jordan",
            reason_any=("checkup", "routine"),
            booked_within_seed_window=True,
        ),
    ),
    EvalCase(
        id="hp_sorethroat_specificday",
        category="happy_path",
        description="Sick visit for a named next working day; accepts an offered time.",
        utterances=[
            "I need to see someone about a sore throat.",
            "Yeah I'd like to book that.",
            "Priya Nair.",
            "3/15/90.",
            "I've been in before.",
            "Just some throat pain and trouble swallowing.",
            f"Can I come in on {FIRST_OPEN_DAY.strftime('%A')}?",
            "Morning works better for me.",
            "Sure, that one's good.",
            "Yes, book it please.",
        ],
        expected=Expected(
            outcome="booked",
            name_contains="priya",
            reason_any=("sore throat", "sick", "throat"),
            booked_on_date=FIRST_OPEN_DAY.isoformat(),
        ),
    ),
    EvalCase(
        id="hp_flushot",
        category="happy_path",
        description="Vaccination booking; short, cooperative caller.",
        utterances=[
            "I want to come in for a flu shot.",
            "Book an appointment, yes.",
            "Sam Okafor.",
            "The fifth of June, nineteen eighty-five.",
            "New patient, first time here.",
            "Nothing's wrong, I just need my flu shot.",
            "Any day this week is fine, earliest you've got.",
            "Great, that time works.",
            "Yes.",
        ],
        expected=Expected(
            outcome="booked",
            name_contains="sam",
            reason_any=("flu", "shot", "vaccin"),
            booked_within_seed_window=True,
        ),
    ),
    EvalCase(
        id="hp_followup_pickalternative",
        category="happy_path",
        description="Follow-up; caller asks for a time and takes a near alternative.",
        utterances=[
            "I'd like to schedule a follow-up visit.",
            "This is Dana Kim.",
            "October 2nd, 1978.",
            "I'm an existing patient — a follow-up from my last visit.",
            "Still some knee soreness I want checked.",
            "It's been aching when I go up stairs for about a week.",
            f"Do you have anything {FIRST_OPEN_DAY.strftime('%A')} around 9?",
            "If 9 is taken, the next closest works.",
            "Okay perfect, that one.",
            "Yes go ahead and book it.",
        ],
        expected=Expected(
            outcome="booked",
            name_contains="dana",
            reason_any=("follow", "follow-up"),
            booked_on_date=FIRST_OPEN_DAY.isoformat(),
        ),
    ),
    EvalCase(
        id="hp_intake_new_patient_wordform",
        category="happy_path",
        description="New patient; DOB spoken in words must normalize to MM/DD/YYYY and be captured.",
        utterances=[
            "Hi, I'd like to book a checkup.",
            "Avery Lindqvist.",
            "The twenty-second of November, nineteen eighty-eight.",
            "I'm a brand new patient.",
            "It's a routine checkup.",
            "Just want a general once-over, nothing's wrong.",
            "Earliest you have is great.",
            "That works.",
            "Yes, book it.",
        ],
        expected=Expected(
            outcome="booked",
            name_contains="avery",
            reason_any=("checkup", "routine"),
            booked_within_seed_window=True,
            dob_equals="11/22/1988",
            new_patient_expected=True,
            symptom_any=("check", "general", "once-over", "nothing"),
        ),
        notes="Verifies spoken-word DOB normalization + new_patient=True capture.",
    ),
    EvalCase(
        id="hp_intake_existing_slashdob",
        category="happy_path",
        description="Existing patient; slash-form DOB captured; new_patient=False.",
        utterances=[
            "I need to book a follow-up.",
            "Yes, an appointment.",
            "Rosa Delgado.",
            "04/09/1971.",
            "I've been a patient here for years.",
            "Follow-up on my blood pressure check.",
            "My blood pressure's been running high and I want it rechecked.",
            "Whatever's soonest works.",
            "Sounds good.",
            "Yes, confirm it.",
        ],
        expected=Expected(
            outcome="booked",
            name_contains="rosa",
            reason_any=("follow", "follow-up"),
            booked_within_seed_window=True,
            dob_equals="04/09/1971",
            new_patient_expected=False,
            symptom_any=("blood pressure", "follow", "pressure"),
        ),
        notes="Verifies slash-form DOB capture + existing-patient (no early-arrival) path.",
    ),
]


# =========================================================================================
# EDGE CASES
# =========================================================================================

EDGE_CASES: list[EvalCase] = [
    EvalCase(
        id="ec_all_at_once",
        category="edge_case",
        description="Caller provides name + reason + preferred day in a single utterance.",
        utterances=[
            f"Hi, this is Marcus Webb, I need a checkup and I'd like to come in "
            f"{FIRST_OPEN_DAY.strftime('%A')} morning if possible.",
            "Sure — date of birth is 07/22/1992, and I'm a new patient.",
            "Just a routine checkup, nothing's bothering me.",
            "Yes that time is good.",
            "Yes, book it.",
        ],
        expected=Expected(
            outcome="booked",
            name_contains="marcus",
            reason_any=("checkup",),
            booked_on_date=FIRST_OPEN_DAY.isoformat(),
        ),
        notes="Tests slot-filling when all fields arrive out of the prompted order.",
    ),
    EvalCase(
        id="ec_out_of_order",
        category="edge_case",
        description="Caller volunteers the day before name/reason are asked.",
        utterances=[
            "Can I get in tomorrow?",
            "Oh, to book an appointment, yes.",
            "It's for a checkup.",
            "Lena Fischer is the name.",
            "Born April 3rd, 1988, and I've been a patient here before.",
            "Just a general check, feeling fine overall.",
            "Yes, the earliest tomorrow works.",
            "Yep, confirm it.",
        ],
        expected=Expected(
            outcome="booked",
            name_contains="lena",
            reason_any=("checkup",),
            booked_within_seed_window=True,
        ),
    ),
    EvalCase(
        id="ec_no_availability",
        category="edge_case",
        description="Caller insists on a date far outside the seeded window -> no slots.",
        utterances=[
            "I'd like to book an appointment.",
            "Chris Donovan.",
            "Just a checkup.",
            f"I can only do {NO_SLOTS_DATE}. Is anything open that day?",
            f"No, it has to be {NO_SLOTS_DATE}, nothing else works for me.",
        ],
        expected=Expected(outcome="escalated", require_zero_availability=True),
        notes="No DB manipulation: the date is beyond the 3-working-day seed horizon.",
    ),
    EvalCase(
        id="ec_missing_reason",
        category="edge_case",
        description="Caller never states a reason; agent should ask before booking.",
        utterances=[
            "I want to book an appointment.",
            "Taylor Brooks.",
            "I'd rather not say what it's about, is that okay?",
        ],
        expected=Expected(outcome="gracefully_handled", expect_followup_question=True),
        notes="A required field is withheld; the agent must follow up, not silently book.",
    ),
    EvalCase(
        id="ec_change_day_midflow",
        category="edge_case",
        description="Caller books, then the offered day doesn't work and they switch.",
        utterances=[
            "Booking an appointment, please.",
            "Omar Haddad.",
            "12/30/1995.",
            "New patient.",
            "A checkup.",
            "No real symptoms, just overdue for a physical.",
            "What have you got tomorrow?",
            "Actually tomorrow's bad — what about the day after?",
            "The earliest one that day works.",
            "Yes, book it.",
        ],
        expected=Expected(
            outcome="booked",
            name_contains="omar",
            reason_any=("checkup",),
            booked_within_seed_window=True,
        ),
    ),
]


# =========================================================================================
# ADVERSARIAL
# =========================================================================================

ADVERSARIAL: list[EvalCase] = [
    EvalCase(
        id="ad_mumbled_vague",
        category="adversarial",
        description="Mumbled, low-information caller; agent must guide to a booking or ask.",
        utterances=[
            "uh yeah hi so like... i dunno, sometime next week i guess?",
            "oh, an appointment yeah",
            "uhh Riley",
            "uh, birthday's like... may 9th, 1991 i think",
            "yeah nah i've been here before",
            "just a normal checkup thing",
            "eh nothing really, just feeling kinda run down",
            "yeah whatever's open, earliest",
            "sure that's fine",
            "yeah ok",
        ],
        expected=Expected(
            outcome="booked",
            name_contains="riley",
            reason_any=("checkup",),
            booked_within_seed_window=True,
        ),
        notes="Ambiguous input should still funnel to a valid booking, not crash.",
    ),
    EvalCase(
        id="ad_midsentence_change",
        category="adversarial",
        description="Caller flips the day mid-sentence twice before settling.",
        utterances=[
            "I need an appointment for a checkup.",
            "Nadia Sokolov.",
            "Date of birth? August 14th, 1983.",
            "I'm a returning patient.",
            "Nothing wrong really, just my routine checkup.",
            "Let's do Tuesday — wait, no, make it Wednesday.",
            "Hmm actually, just give me the earliest you have any day.",
            "Okay that works.",
            "Yes, confirm.",
        ],
        expected=Expected(
            outcome="booked",
            name_contains="nadia",
            reason_any=("checkup",),
            booked_within_seed_window=True,
        ),
        notes="Self-corrections must resolve to the final stated preference.",
    ),
    EvalCase(
        id="ad_nonsense_then_recover",
        category="adversarial",
        description="Off-topic nonsense first, then a real booking request.",
        utterances=[
            "What's your favorite color? Do you like pizza?",
            "haha okay okay. Actually, can I book an appointment?",
            "Gabriel Santos.",
            "Sure, 11/11/1990. First time patient, by the way.",
            "A flu shot.",
            "Nothing's wrong, just need the shot.",
            "Earliest available is fine.",
            "Yes, that works.",
            "Yes book it.",
        ],
        expected=Expected(
            outcome="booked",
            name_contains="gabriel",
            reason_any=("flu", "shot", "vaccin"),
            booked_within_seed_window=True,
        ),
        notes="Redirect gracefully from off-topic chatter, then complete the task.",
    ),
    EvalCase(
        id="ad_pure_nonsense",
        category="adversarial",
        description="Persistent off-topic input; agent must not crash or fabricate a booking.",
        utterances=[
            "asdf jkl; the moon is a hologram",
            "banana banana banana",
            "tell me a story about dragons",
        ],
        expected=Expected(outcome="gracefully_handled"),
        notes="No coherent intent — must stay on-task and never invent a booking.",
    ),
    EvalCase(
        id="ad_unsupported_intent",
        category="adversarial",
        description="Caller wants a billing question, not a booking -> should decline/escalate.",
        utterances=[
            "Hi, I have a question about a charge on my bill.",
            "No, I don't want to book anything, I just want to dispute a payment.",
            "So can you pull up my invoice?",
        ],
        expected=Expected(outcome="escalated"),
        notes="Out-of-scope intent; agent should hand off to staff, never book.",
    ),
    EvalCase(
        id="ad_offtopic_medical_detail",
        category="adversarial",
        description="Caller volunteers detailed medical history; agent must minimize PHI.",
        utterances=[
            "I need to book about my sore throat.",
            "Yes, book me in.",
            "Morgan Reyes.",
            "June 6th, 1975. I'm an existing patient.",
            "It started after a long detailed story about my chronic condition and meds...",
            "Bottom line — my throat's been sore and scratchy for a few days.",
            "Any day works, just the earliest slot you've got.",
            "Yes, that time works.",
            "Yes, book it.",
        ],
        expected=Expected(
            outcome="booked",
            name_contains="morgan",
            reason_any=("sore throat", "throat", "sick"),
            booked_within_seed_window=True,
        ),
        notes="Should capture only a coarse reason, not the clinical narrative.",
    ),
    EvalCase(
        id="ad_wants_human",
        category="adversarial",
        description="Caller explicitly asks for a human agent.",
        utterances=[
            "Can I just talk to a real person please?",
            "Yeah, a human. I don't want to do this with a bot.",
        ],
        expected=Expected(outcome="escalated"),
        notes="Explicit human-handoff request is a global escalation intent.",
    ),
    EvalCase(
        id="ad_intake_symptom_minimization",
        category="adversarial",
        description="Caller volunteers a long medical narrative; symptom_notes must stay brief.",
        utterances=[
            "I want to book about a sore throat.",
            "Yes please.",
            "Devon Marsh.",
            "02/17/1969.",
            "Existing patient.",
            "Well it all started three weeks ago after my flight, then my sinuses, and "
            "my old prescription from Dr. So-and-so, and my cousin had the same thing, and...",
            "Earliest is fine.",
            "Yes, that works.",
            "Yes, book it.",
        ],
        expected=Expected(
            outcome="booked",
            name_contains="devon",
            reason_any=("throat", "sore throat", "sick"),
            booked_within_seed_window=True,
            dob_equals="02/17/1969",
            new_patient_expected=False,
            symptom_any=("throat", "sinus"),
            symptom_max_words=25,
        ),
        notes="PHI minimization on the new symptom field: capture a coarse note, not the saga.",
    ),
]


ALL_CASES: list[EvalCase] = HAPPY_PATH + EDGE_CASES + ADVERSARIAL


def cases_by_category() -> dict[str, list[EvalCase]]:
    out: dict[str, list[EvalCase]] = {}
    for c in ALL_CASES:
        out.setdefault(c.category, []).append(c)
    return out

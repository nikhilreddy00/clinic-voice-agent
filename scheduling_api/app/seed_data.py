"""Synthetic seed data for the mock clinic scheduling API.

Everything here is fictional. No real patients, providers, or PHI.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

# The clinic's wall clock. Slot times below are generated in THIS zone, not UTC.
#
# Phase-9 fix: slots used to be built with tzinfo=timezone.utc, so a "9:00 appointment" was
# 09:00 UTC — 05:00 in the clinic's morning, four or five hours before the clinic opens. The
# agent then offered times no one could attend, and a mid-day caller could find the day's slots
# already "past". Times are business hours in the clinic's own zone; storage is timestamptz, so
# the absolute instant is still unambiguous and DST is handled by the zone rather than by us.
CLINIC_TZ = ZoneInfo("America/New_York")

# Fictional clinic identity (used in greetings / responses).
CLINIC_NAME = "Grove Family Clinic"

# (provider_id, name, specialty)
PROVIDERS: list[tuple[int, str, str]] = [
    (1, "Dr. Elena Rivera", "Family Medicine"),
    (2, "Dr. Marcus Chen", "Pediatrics"),
    (3, "Dr. Aisha Patel", "Internal Medicine"),
]

# Appointment reason categories a slot can be tagged for (coarse, non-clinical).
REASON_CATEGORIES = ["checkup", "follow-up", "sick-visit", "vaccination"]

# Times of day each provider offers slots on a working day.
_SLOT_TIMES = [time(9, 0), time(10, 30), time(13, 0), time(14, 30)]

# How many upcoming working days to generate slots for.
_DAYS_AHEAD = 3


def generate_slots() -> list[tuple[int, int, datetime, str]]:
    """Return synthetic slots as (slot_id, provider_id, start_time, reason_category).

    Slots are generated for the next few working days (Mon-Fri) at fixed times per provider, in
    the CLINIC'S timezone — 9:00 means 9am where the clinic is. slot_id is deterministic so
    seeding is idempotent.

    Returns tz-aware datetimes rather than ISO strings: the column is timestamptz, so handing the
    driver a real datetime keeps the conversion in one place and removes the string-comparison
    behaviour that made the old UTC storage fragile.

    "Today" is also the clinic's today. Deriving it from UTC put the whole window a day out for
    several hours each evening, when it is already tomorrow in UTC but still today in the clinic.
    """
    slots: list[tuple[int, int, datetime, str]] = []
    slot_id = 1
    today = datetime.now(CLINIC_TZ).date()

    days_collected = 0
    day_offset = 1
    while days_collected < _DAYS_AHEAD:
        day = today + timedelta(days=day_offset)
        day_offset += 1
        if day.weekday() >= 5:  # skip Sat/Sun
            continue
        days_collected += 1
        for provider_id, _, _ in PROVIDERS:
            for i, t in enumerate(_SLOT_TIMES):
                start = datetime.combine(day, t, tzinfo=CLINIC_TZ)
                reason = REASON_CATEGORIES[(provider_id + i) % len(REASON_CATEGORIES)]
                slots.append((slot_id, provider_id, start, reason))
                slot_id += 1
    return slots


# Per-tenant curated facts, injected by intent (Phase 13). A fact table rather than a vector
# store: at this volume it is faster, cheaper, and — the part that matters for a clinic — fully
# auditable. Every string here is what the agent is allowed to say on the topic; anything not
# listed is something it must hand to staff rather than infer.
CLINIC_FACTS: dict[str, str] = {
    "hours": (
        "Grove Family Clinic is open Monday through Friday, 8 AM to 5 PM, and closed on "
        "weekends and federal holidays."
    ),
    "location": (
        "Grove Family Clinic is at 214 Grove Street, Suite 3, in Riverton. It's the brick "
        "building next to the public library."
    ),
    "parking": (
        "There's a free patient lot behind the building, entered from Willow Lane, and street "
        "parking on Grove Street."
    ),
    "providers": (
        "The clinic's providers are Dr. Elena Rivera in family medicine, Dr. Marcus Chen in "
        "pediatrics, and Dr. Aisha Patel in internal medicine."
    ),
    "appointment_prep": (
        "Please arrive ten minutes early, and bring a photo ID, your insurance card, and a list "
        "of the medications you take."
    ),
    "insurance": (
        "The clinic accepts most major insurance plans. Staff confirm coverage for a specific "
        "plan before the visit — the agent does not quote coverage."
    ),
}

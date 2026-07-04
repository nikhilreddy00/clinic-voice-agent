"""Synthetic seed data for the mock clinic scheduling API.

Everything here is fictional. No real patients, providers, or PHI.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta, timezone

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


def generate_slots() -> list[tuple[int, int, str, str]]:
    """Return synthetic slots as (slot_id, provider_id, start_time_iso, reason_category).

    Slots are generated for the next few working days (Mon-Fri) at fixed times per provider.
    slot_id is deterministic so seeding is idempotent.
    """
    slots: list[tuple[int, int, str, str]] = []
    slot_id = 1
    today = datetime.now(timezone.utc).date()

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
                start = datetime.combine(day, t, tzinfo=timezone.utc)
                reason = REASON_CATEGORIES[(provider_id + i) % len(REASON_CATEGORIES)]
                slots.append((slot_id, provider_id, start.isoformat(), reason))
                slot_id += 1
    return slots

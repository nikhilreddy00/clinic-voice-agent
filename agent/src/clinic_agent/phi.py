"""Phase 17 — the PHI boundary. One place that says what may be written down.

Everything the agent knows about a caller passes through a tool call, and a tool call's
arguments are the most PHI-dense object in the process: a name, a date of birth, a phone
number, and a sentence of clinical history, all in one dict. Those dicts get logged.

Through Phase 16 the redaction rule was applied BY HAND at each log line —
`dob={'set' if ... else 'unset'}`, `symptom_notes_len=...`, `_redact_phone(...)` — which is
correct wherever it was remembered and silent wherever it was not. It was not remembered for
`patient_name`, in two separate lines, for four phases. A convention that is only as good as
the author's attention is not a boundary; this module is.

    safe_args(arguments)   -> a dict that is safe to log, no matter what is in it
    redact(field, value)   -> one field's safe display form

The rule is deny-by-default over a NAMED field set, not a scan for things that look like PHI:
pattern-matching for names is a losing game, whereas the set of keys this system actually puts
in a tool argument is short, stable, and reviewable in one screen. A key that is not in
`PHI_FIELDS` passes through unchanged, so adding a tool argument that carries PHI means adding
it here — and `tests/test_phi.py` drives every tool with sentinel values through the real
logger, so forgetting fails the build rather than the next call.

**What this does NOT govern.** `logs/traces/` holds full transcripts on purpose (Phase 16): it
is a local debug artifact that never leaves the machine, and the whole live-call debugging loop
depends on it. This module governs the two things that DO leave — loggers and OTel spans (the
span side is `core/otel.ATTRIBUTES`, a closed allowlist enforced the same way). See
`docs/compliance.md`.

**`reason` is deliberately absent.** `reason` / `reason_category` on the scheduling tools is a
coarse, non-clinical category from a fixed four-value list (`REASON_CATEGORIES`), and seeing it
in a log is how a booking flow is debugged. The caller's own spoken words never go there — the
clinical sentence goes in `symptom_notes`, which IS covered. A free-text reason (the one
`cancel_appointment` accepts) must be redacted at the call site with `redact("notes", value)`.
"""

from __future__ import annotations

import os
from typing import Any, Mapping

# Tool-argument keys that carry PHI. Values are replaced by `redact` before anything is logged.
PHI_FIELDS = frozenset({
    # identity
    "patient_name", "name", "phone", "date_of_birth",
    # clinical free text
    "symptom_notes", "notes", "medication", "transcript",
})

# How each field is rendered once redacted. Presence is nearly always what a log line is for
# ("did the model send a DOB at all?"); the value never is.
#   phone  -> last four digits: enough to correlate two lines from one call, not enough to dial
#   text   -> a length, which is what actually helps when a model sends an empty string
#   rest   -> presence only
_LENGTH_FIELDS = frozenset({"symptom_notes", "notes", "transcript"})


def redact_phone(phone: str | None) -> str:
    """Last four digits only. The ANI is an identifier; logs are not a place to keep one."""
    digits = "".join(c for c in str(phone or "") if c.isdigit())
    return f"***{digits[-4:]}" if digits else "unset"


def redact(field: str, value: Any) -> str:
    """One field's safe display form. Safe to call on a non-PHI field; it just returns repr."""
    if field not in PHI_FIELDS:
        return repr(value)
    if field == "phone":
        return redact_phone(value)
    if value is None or value == "":
        return "unset"
    if field in _LENGTH_FIELDS:
        return f"len={len(str(value))}"
    return "set"


def safe_args(arguments: Mapping[str, Any] | None) -> dict[str, Any]:
    """A tool-argument dict with every PHI value replaced. Non-PHI keys pass through.

    The redacted values are strings rather than repr-able objects on purpose: the result is
    meant to be dropped straight into an f-string log line, where `dob=set` reads better than
    `dob='set'`.
    """
    out: dict[str, Any] = {}
    for key, value in (arguments or {}).items():
        out[key] = _Literal(redact(key, value)) if key in PHI_FIELDS else value
    return out


# --- spoken content -----------------------------------------------------------------------
#
# The caller's own words are the densest PHI in the system, and they land in two places: the
# `ASR ▶ transcript received` / `LLM ▶ response generated` console lines, and every event in
# `logs/traces/<call_id>.jsonl`.
#
# Both stay ON by default and that is a deliberate, defensible choice for THIS build: the data
# is synthetic, and the whole live-call debugging loop (inspect_call.py, trace_viewer.py, the
# Tier-1 replay corpus) is built on those traces. Redacting the console while writing the full
# transcript to a file on the same disk would be theater, not a control.
#
# So the control is one switch that governs both. `CLINIC_PHI_LOGS=0` is what a deployment
# handling real PHI sets: the console prints lengths, and traces keep their structure — every
# event, every tool call, every timing — with the words removed. Replay still runs; it just
# cannot show you what was said. See docs/compliance.md.
_OFF = {"0", "false", "no", "off"}

# Event/record keys holding free speech, scrubbed from traces when retention is off.
TEXT_FIELDS = frozenset({"text", "tail", "transcript", "content"})


def retain_transcripts() -> bool:
    """Whether spoken content may be written to the console and to trace files."""
    return os.getenv("CLINIC_PHI_LOGS", "1").strip().lower() not in _OFF


def speech(value: Any) -> str:
    """A transcript or model reply, for a log line: the words, or just their length."""
    return repr(value) if retain_transcripts() else f"«len={len(str(value or ''))}»"


def scrub_event(record: dict[str, Any]) -> dict[str, Any]:
    """A trace record with spoken content and PHI tool arguments removed.

    A no-op when retention is on, which is the default and the only path any current test or
    replay corpus takes.
    """
    if retain_transcripts():
        return record
    out = dict(record)
    for key in TEXT_FIELDS & out.keys():
        out[key] = f"«len={len(str(out[key] or ''))}»"
    if isinstance(out.get("arguments"), dict):
        out["arguments"] = {k: str(v) for k, v in safe_args(out["arguments"]).items()}
    return out


class _Literal(str):
    """A string that reprs as itself, so `f"{safe_args(a)}"` shows `dob=set`, not `dob='set'`."""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return str(self)

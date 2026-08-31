"""Phase 12 — intent classification, and the deterministic emergency path.

Two mechanisms live here and they are deliberately not the same kind of thing.

**Emergency detection is a pure function and never involves a model.** ``detect_emergency()``
is called by the reducer on every finalized caller utterance, before any LLM request exists.
A caller saying they cannot breathe must get the same scripted response on every call, on every
model, during a provider outage, and at 3 a.m. when the API is timing out. Routing that through
a language model would make the single highest-stakes control in the system probabilistic and
non-replayable. Because it is pure, it also appears in every recorded trace and every replay,
so the Phase-16 eval can assert on it forever.

**Intent classification is a model call**, because deciding between "billing question" and
"insurance verification" genuinely needs language understanding. It runs *in parallel* with the
dialogue turn, never in series — the classifier arriving late is a re-plan, not a stall.

### The recall/precision trade, stated plainly

Emergency recall is the non-negotiable metric: a missed emergency is the worst thing this system
can do. Precision is deliberately sacrificed to it — a false positive tells a caller who wanted
a checkup to hang up and dial 911, which is bad, but it is recoverable in a way that the other
error is not.

That said, unlimited false positives are their own harm (they erode trust and delay real care),
so a narrow negation guard suppresses *explicit denials* of a small set of symptom nouns:
"I don't have chest pain" should not trigger. The guard is applied **only** to phrases in
:data:`NEGATABLE_PATTERNS`. It is emphatically NOT applied to phrases that carry their own
polarity — "can't breathe" and "not breathing" contain negation words and are the most urgent
strings in the whole module. A general-purpose negation rule would suppress exactly those, which
is how this kind of safety check gets silently inverted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..intents import (
    HANDLED_INTENTS,
    MIN_INTENT_CONFIDENCE,
    SCHEDULING_INTENTS,
    Intent,
    needs_clarification,
)

__all__ = [
    "HANDLED_INTENTS",
    "MIN_INTENT_CONFIDENCE",
    "SCHEDULING_INTENTS",
    "CLASSIFIER_SYSTEM_PROMPT",
    "CLASSIFIER_TOOL",
    "EmergencyMatch",
    "Intent",
    "classifier_messages",
    "detect_emergency",
    "needs_clarification",
]


@dataclass(frozen=True)
class EmergencyMatch:
    """A fired emergency rule. ``phrase`` is kept so traces show exactly what triggered."""

    category: str
    phrase: str


def _compile(patterns: dict[str, list[str]]) -> list[tuple[str, re.Pattern[str]]]:
    return [
        (category, re.compile(pattern, re.IGNORECASE))
        for category, group in patterns.items()
        for pattern in group
    ]


# Phrases that fire unconditionally. Every one of these either carries its own polarity or is
# not something a caller says about a situation that is fine.
EMERGENCY_PATTERNS = _compile(
    {
        "cardiac": [
            r"\bheart attack\b",
            r"\bcardiac arrest\b",
            r"\bcrushing (?:pain|pressure)\b",
            r"\bchest (?:pressure|tightness)\b",
            r"\bmy chest (?:is tight|feels tight|hurts)\b",
            # "my chest is crushing" — the promptfoo suite found this one. The existing
            # `crushing (pain|pressure)` requires the noun the caller often leaves out.
            r"\bchest (?:is|feels) crushing\b",
            r"\bcrushing (?:feeling|sensation)\b",
        ],
        "respiratory": [
            r"\bcan'?t breathe\b",
            r"\bcannot breathe\b",
            r"\bnot breathing\b",
            r"\bstopped breathing\b",
            r"\b(?:trouble|difficulty|struggling) breathing\b",
            r"\bstruggling to breathe\b",
            r"\bhard to breathe\b",
            r"\bgasping\b",
            r"\bchoking\b",
            r"\bthroat is closing\b",
            r"\bthroat closing\b",
        ],
        "stroke": [
            r"\bstroke\b",
            r"\bface (?:is )?drooping\b",
            r"\bslurred speech\b",
            r"\bslurring\b",
            r"\bnumb on one side\b",
            r"\bone side of (?:my|his|her|their) (?:body|face)\b",
            r"\bcan'?t move (?:my |his |her )?(?:arm|leg|face|side)\b",
        ],
        "consciousness": [
            r"\bunconscious\b",
            r"\bunresponsive\b",
            r"\bpassed out\b",
            r"\bblacked out\b",
            r"\bwon'?t wake up\b",
            r"\bseizure\b",
            r"\bconvulsing\b",
            r"\bfainted\b",
            # Imminent, not past. "passed out" and "fainted" were covered; a caller who is
            # still conscious enough to say it is about to happen was not.
            r"\b(?:about to|going to|gonna|think i'?m gonna) (?:pass out|faint|black out)\b",
            r"\bpassing out\b",
            r"\bfainting\b",
        ],
        "hemorrhage": [
            r"\bbleeding (?:heavily|badly|a lot)\b",
            r"\bwon'?t stop bleeding\b",
            r"\bsevere bleeding\b",
            r"\bhemorrhag\w*\b",
            r"\blost a lot of blood\b",
            r"\bblood everywhere\b",
        ],
        "self_harm": [
            # Inflections matter: "hurting myself" is as urgent as "hurt myself", and a
            # test-set miss on exactly that is why these carry an explicit (?:ing)?.
            r"\bkill(?:ing)? (?:myself|himself|herself|themselves)\b",
            r"\bsuicid\w*\b",
            r"\bend(?:ing)? (?:my|his|her) life\b",
            r"\bhurt(?:ing)? (?:myself|himself|herself|themselves)\b",
            r"\bharm(?:ing)? (?:myself|himself|herself|themselves)\b",
            r"\bwant to die\b",
            r"\bdon'?t want to (?:live|be here)\b",
        ],
        "overdose": [
            r"\boverdos\w*\b",
            r"\btook too many (?:pills|of)\b",
            r"\bswallowed (?:a |the )?(?:whole )?bottle\b",
        ],
        "anaphylaxis": [
            r"\banaphyla\w*\b",
            r"\bthroat (?:is )?swelling\b",
            r"\btongue (?:is )?swelling\b",
            r"\bsevere allergic reaction\b",
        ],
        "explicit": [
            # Deliberately NOT bare "emergency": "can I get an emergency appointment" and
            # "are you near the emergency room" are ordinary scheduling speech, and firing a
            # 911 script at those callers would be both wrong and corrosive to trust.
            r"\bthis is an emergency\b",
            r"\bit'?s an emergency\b",
            r"\bmedical emergency\b",
            r"\bhaving an emergency\b",
            r"\bcall 911\b",
            r"\bshould i (?:call )?911\b",
            r"\b(?:i'?m|he'?s|she'?s|they'?re) dying\b",
            r"\blife[- ]threatening\b",
        ],
    }
)

# Symptom nouns a caller may explicitly deny. ONLY these are subject to the negation guard.
NEGATABLE_PATTERNS = _compile(
    {
        "cardiac": [r"\bchest pains?\b", r"\bpain in (?:my|his|her) chest\b"],
        "hemorrhage": [r"\bbleeding\b"],
    }
)

# An explicit denial immediately before the symptom, within two words. Narrow on purpose: a
# wide window starts swallowing real reports ("no, I have chest pain").
_DENIAL = re.compile(
    r"(?:\bno\b|\bnot\b|n'?t have\b|\bwithout\b|\bdenies\b|\bnever had\b)\W+(?:\w+\W+){0,2}$",
    re.IGNORECASE,
)


# A few phrases describe either a crisis happening now or a memory of one. "I passed out" is an
# emergency; "I passed out flyers at the health fair" and "I fainted once as a teenager" are not.
#
# This guard is deliberately confined to those three past-tense phrases and is NOT a general
# rule, for the same reason the negation guard is not: applied to "can't breathe" or "not
# breathing", a rule like this would suppress the most urgent strings in the module. Widening
# either guard is how a safety check gets silently inverted.
#
# The bias is still towards firing. A missed emergency is unbounded harm; a false positive is a
# caller told to hang up and dial 911 when they did not need to, which is bad service and safe.
# So the guard requires EXPLICIT evidence — a past-tense marker or a direct object — never an
# absence of urgency cues.
_HISTORICAL_SENSITIVE = re.compile(r"\b(?:passed out|blacked out|fainted)\b", re.IGNORECASE)

_HISTORY_MARKER = re.compile(
    r"\b(?:once|twice|a few times|years? ago|months? ago|weeks? ago|last (?:week|month|year)|"
    r"as a (?:kid|child|teenager|teen)|when i was|in the past|history of|used to|"
    r"back in \d{4}|previously)\b",
    re.IGNORECASE,
)

# "passed out X" where X is a thing being handed round — the verb is transitive and has nothing
# to do with consciousness.
_PASSED_OUT_OBJECT = re.compile(
    r"\bpassed out\s+(?:\w+\s+){0,2}"
    r"(?:flyers?|leaflets?|pamphlets?|copies|forms?|papers?|samples?|candy|snacks?|cards?)\b",
    re.IGNORECASE,
)


def _is_recollection(text: str, phrase: str) -> bool:
    """True when a consciousness phrase is describing the past rather than the present."""
    if not _HISTORICAL_SENSITIVE.fullmatch(phrase.strip()):
        return False
    return bool(_HISTORY_MARKER.search(text) or _PASSED_OUT_OBJECT.search(text))


def detect_emergency(text: str) -> EmergencyMatch | None:
    """Return the first matching emergency rule, or None. Pure — no I/O, no model, no clock.

    Called by the reducer on every finalized caller utterance. Order matters only in that
    unconditional patterns are checked before negatable ones, so a denial of one symptom can
    never mask an unrelated unconditional phrase in the same sentence.
    """
    if not text or not text.strip():
        return None

    for category, pattern in EMERGENCY_PATTERNS:
        match = pattern.search(text)
        if match and not _is_recollection(text, match.group(0)):
            return EmergencyMatch(category=category, phrase=match.group(0))

    for category, pattern in NEGATABLE_PATTERNS:
        match = pattern.search(text)
        if match and not _DENIAL.search(text[: match.start()]):
            return EmergencyMatch(category=category, phrase=match.group(0))

    return None


# --- intent classification ------------------------------------------------------------------

CLASSIFIER_TOOL = {
    "name": "classify_intent",
    "description": (
        "Record what the caller wants. Call this exactly once, for every utterance."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "intent": {
                "type": "string",
                "enum": [i.value for i in Intent],
                "description": (
                    "The caller's primary intent. Use 'unknown' when the utterance is a "
                    "greeting, filler, or genuinely ambiguous — do not guess a specific intent."
                ),
            },
            "confidence": {
                "type": "number",
                "description": "0.0-1.0. Below 0.6 the agent asks a clarifying question.",
            },
        },
        "required": ["intent", "confidence"],
    },
}

CLASSIFIER_SYSTEM_PROMPT = """\
You classify the intent of a caller phoning a family medical clinic's scheduling line.

Return exactly one intent via the classify_intent tool. Be literal: classify what the caller
actually said, not what they might mean next.

  schedule_appointment    booking a NEW visit
  reschedule_appointment  moving an EXISTING booked visit
  cancel_appointment      cancelling an existing visit
  medication_refill       prescription refill or renewal
  billing_question        bills, charges, payment, cost of a visit
  clinical_question       medical advice about symptoms or treatment
  test_results            lab or imaging results
  insurance_verification  whether a plan is accepted, coverage questions
  hours_location          opening hours, address, directions, parking
  speak_to_human          explicitly asking for a person or the front desk
  emergency               an urgent medical crisis happening now
  unknown                 greeting, filler, or too ambiguous to place

Confidence is your own certainty. Use below 0.6 when the utterance could reasonably belong to
two intents; the agent will ask a clarifying question rather than guess.
"""


def classifier_messages(utterance: str) -> list[dict]:
    """The single-turn message list for a classification request.

    Only the utterance is sent — no conversation history. That keeps the classifier's prompt at
    a few hundred tokens (it runs on every topic shift and its latency budget is ~150 ms), and
    it keeps classification honest: the point is what the caller *just said*, not a re-reading
    of the whole call.
    """
    return [{"role": "user", "content": utterance}]


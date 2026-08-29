"""Is the caller finished, or just thinking?

Deepgram ends a turn after ``endpointing`` milliseconds of silence — 300 here, chosen for
latency. A caller who pauses mid-sentence to gather their thoughts therefore *ends their turn*,
and the agent answers a fragment. On the first successfully-booked call through this engine
that happened seven times in twenty-seven turns:

    caller: "Well, there is a severe pain in the"     agent: "Take your time — I'm listening."
    caller: "How about"                               agent: "I'm all ears — what day?"
    caller: "in"                                      agent: "Take your time."

Each one costs a full LLM and TTS round trip, and — the part the caller actually feels — talks
over someone who was mid-thought. The model papering over it politely ("take your time") made
the symptom pleasant while leaving the cause alone.

The obvious fix is to raise the endpointing window, and it is the wrong one: it slows down
every turn, including the great majority that were already correct. This is the cheaper trade —
keep the fast window, and hold the flush *only* for utterances that are visibly unfinished.
Turns that were already right stay exactly as fast as they were.

Pure and deterministic on purpose. Turn-taking decides when a caller gets interrupted, and that
is not a thing to make probabilistic — the same reasoning as the emergency detector.
"""

from __future__ import annotations

import re

# Deepgram punctuates (``punctuate=true``), so a terminal mark is a strong finished signal.
TERMINAL = (".", "?", "!")

# Words that cannot end an English sentence. A transcript ending here is mid-clause: the caller
# is still assembling it. Kept narrow — a false positive costs one grace window of latency,
# but a word wrongly listed here would delay every turn that legitimately ends with it.
CONTINUATION_WORDS = frozenset(
    {
        # articles and determiners
        "a", "an", "the", "my", "your", "his", "her", "their", "our", "this", "that", "these",
        "those", "some", "any",
        # prepositions
        "about", "at", "by", "for", "from", "in", "into", "of", "on", "onto", "to", "with",
        "without", "over", "under", "near", "since", "until", "during",
        # conjunctions and subordinators
        "and", "or", "but", "so", "because", "if", "when", "while", "as", "than", "though",
        "although", "whether",
        # auxiliaries and copulas
        "am", "is", "are", "was", "were", "be", "been", "being", "has", "have", "had", "do",
        "does", "did", "will", "would", "can", "could", "should", "might", "must",
        # pronouns and fillers that trail off
        "i", "we", "they", "it", "there", "um", "uh", "er", "like", "well", "just", "really",
        "very", "kind", "sort", "maybe",
    }
)

_WORD = re.compile(r"[a-z']+")

# Two tiers, because the evidence comes in two strengths and one window cannot serve both.
#
# STRONG — a trailing comma or a word that cannot end an English sentence ("in", "the", "with",
# "Well,"). The caller is unambiguously mid-clause, so buy them real time.
#
# WEAK — no terminal punctuation at all. `punctuate=true` means Deepgram declined to close the
# sentence, which is suggestive but not conclusive: "December eight two thousand" and "Nikki
# Kumarati" are complete answers that simply never got a full stop. A short window is cheap
# insurance; a long one would tax every name and date of birth in the call.
STRONG_GRACE_S = 1.6
WEAK_GRACE_S = 0.7


def looks_unfinished(text: str) -> bool:
    """True when the transcript is unambiguously mid-clause (the STRONG signal).

    >>> looks_unfinished("Well, there is a severe pain in the")
    True
    >>> looks_unfinished("in")
    True
    >>> looks_unfinished("Well,")
    True
    >>> looks_unfinished("It's been two weeks.")
    False
    >>> looks_unfinished("December eight two thousand")
    False
    """
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.endswith(TERMINAL):
        return False
    if stripped.endswith(","):
        return True
    words = _WORD.findall(stripped.lower())
    return bool(words) and words[-1] in CONTINUATION_WORDS


def grace_seconds(text: str) -> float:
    """How long to keep listening after Deepgram calls the turn over. 0.0 means flush now.

    >>> grace_seconds("It's been two weeks.")
    0.0
    >>> grace_seconds("I got some of")
    1.6
    >>> grace_seconds("December eight two thousand")
    0.7
    """
    stripped = text.strip()
    if not stripped:
        return 0.0
    if looks_unfinished(stripped):
        return STRONG_GRACE_S
    if stripped.endswith(TERMINAL):
        return 0.0
    return WEAK_GRACE_S

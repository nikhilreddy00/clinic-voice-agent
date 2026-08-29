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


def looks_unfinished(text: str) -> bool:
    """True when the caller is mid-sentence and the turn should not end yet.

    >>> looks_unfinished("Well, there is a severe pain in the")
    True
    >>> looks_unfinished("in")
    True
    >>> looks_unfinished("Well,")
    True
    >>> looks_unfinished("It's been two weeks.")
    False
    >>> looks_unfinished("Seven.")
    False
    >>> looks_unfinished("December eight")
    False
    """
    stripped = text.strip()
    if not stripped:
        return False

    # Deepgram closed the sentence itself. Trust it — this is the common case and it must stay
    # on the fast path.
    if stripped.endswith(TERMINAL):
        return False

    # A trailing comma is a caller drawing breath mid-list, never an ending.
    if stripped.endswith(","):
        return True

    words = _WORD.findall(stripped.lower())
    if not words:
        return False
    return words[-1] in CONTINUATION_WORDS

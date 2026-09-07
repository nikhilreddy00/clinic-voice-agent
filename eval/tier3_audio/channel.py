"""What a phone line does to what a caller said.

A pure function: `(text, condition, seed) -> text`. No audio, no ASR, no network, no clock.

WHY THIS EXISTS INSTEAD OF SYNTHESISED AUDIO IN THE OFFLINE PATH
----------------------------------------------------------------
Tier 3 is meant to answer "does the agent still book an appointment when it cannot hear the
caller cleanly". The obvious way to ask that is to synthesise speech, degrade it, and run it
through Deepgram — which is what `--live` does, and it bills Cartesia and Deepgram per case.

Doing that offline is not a cheaper version of the same test, it is a different and empty one:
with a fake STT the harness hands the session a scripted string, so the audio is generated,
thrown away, and the transcript arrives perfect. That measures the plumbing and calls it
robustness.

So the offline path models the OUTPUT of a bad channel rather than its input. Every corruption
below is one this project has actually seen in `logs/traces/`, and applying them to the real
Tier-2 cases exercises the part that matters — the dialogue's recovery — through the real
prompt, the real tools and real Postgres.

WHAT IT DOES NOT MEASURE, stated plainly so nobody quotes it as if it did: Deepgram's actual
error rate on an accent, in noise, or over a codec. Only `--live` can say that, and it has not
been run.

The corruption model is deliberately crude in one specific way: it is a per-word process with
no language model behind it, so it cannot produce the plausible-but-wrong whole phrases a real
ASR invents. Real errors are therefore somewhat HARDER than these, not easier — worth knowing
when a case passes here.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

# Digit and number confusions, taken from real traces. The teens/tens pair is the one that
# matters most on this line: it lands in dates of birth and appointment times, where being
# wrong by a decade is not a rounding error.
_TEENS_TENS = {
    "thirteen": "thirty", "fourteen": "forty", "fifteen": "fifty", "sixteen": "sixty",
    "seventeen": "seventy", "eighteen": "eighty", "nineteen": "ninety",
}

NUMBER_CONFUSIONS: dict[str, str] = {
    **_TEENS_TENS,
    **{v: k for k, v in _TEENS_TENS.items()},          # and the other direction
    # ORDINALS MATTER MORE THAN CARDINALS HERE, and leaving them out was the first thing the
    # preview showed: a caller does not say "December fifteen", they say "December fifteenth".
    # Dates of birth and appointment days are almost entirely ordinals, and they are the two
    # places where being wrong by a decade is not a rounding error.
    **{f"{k}th": f"{v[:-1]}ieth" for k, v in _TEENS_TENS.items()},
    **{f"{v[:-1]}ieth": f"{k}th" for k, v in _TEENS_TENS.items()},
    "two": "to", "four": "for", "eight": "ate", "one": "won",
}

# Homophones and near-misses seen on this line. "can't"/"can" is the one with teeth: it inverts
# a sentence, and `core/intent.NEGATABLE_PATTERNS` exists because of that class of error.
HOMOPHONES: dict[str, str] = {
    "can't": "can", "cannot": "can", "won't": "want", "don't": "do",
    "there": "their", "here": "hear", "no": "know", "knew": "new",
    "week": "weak", "meet": "meat", "flu": "flew", "sore": "saw",
}

# Words a caller says that a noisy line most often loses entirely: short, unstressed, and
# usually the ones carrying the grammar rather than the content.
DROPPABLE = frozenset({
    "a", "an", "the", "is", "it", "to", "of", "and", "for", "at", "on", "in",
    "my", "i", "um", "uh", "so", "just", "please", "okay",
})


@dataclass(frozen=True)
class Condition:
    """One named channel, and how badly it mangles a sentence.

    Probabilities are per WORD. They compound across a long utterance, which is the intended
    behaviour — a caller reciting a date of birth has more surface for the line to damage than
    one saying "yes".
    """

    name: str
    drop: float = 0.0            # a droppable word disappears
    confuse_number: float = 0.0  # a number becomes its confusable neighbour
    homophone: float = 0.0       # a word becomes its homophone
    truncate_tail: float = 0.0   # the last few words are cut off entirely
    confidence: float = 0.97     # what the STT would report, for the low-confidence ladder
    note: str = ""

    # Voices and noise levels the LIVE path would sweep. Inert offline, and kept here so the
    # two paths cannot describe different conditions by the same name.
    voices: tuple[str, ...] = ()
    snr_db: float | None = None


CONDITIONS: dict[str, Condition] = {
    "clean": Condition(
        name="clean", confidence=0.97,
        note="the control. If a case fails here it is not an audio problem.",
        snr_db=None,
    ),
    "mild": Condition(
        name="mild", drop=0.04, confuse_number=0.05, homophone=0.03, confidence=0.90,
        note="a decent mobile call: the odd article lost, the occasional teens/tens slip.",
        snr_db=20.0,
    ),
    "noisy": Condition(
        name="noisy", drop=0.12, confuse_number=0.18, homophone=0.10, truncate_tail=0.10,
        confidence=0.72,
        note="a car, a corridor, a speakerphone. Sustained low confidence here is what the "
             "Phase-15 ladder escalates on, so this condition also exercises the hand-off.",
        snr_db=10.0,
    ),
    "clipped": Condition(
        name="clipped", truncate_tail=0.45, drop=0.03, confidence=0.88,
        note="endpointing cutting the caller off mid-sentence. `core/endpointing.py` exists to "
             "buy grace here, so a case that survives this is evidence FOR the grace rule.",
        snr_db=15.0,
    ),
}

# Live-path voice sweep. Not used offline; declared once so `--live` and the offline conditions
# cannot drift into describing different experiments by the same names.
LIVE_VOICES: tuple[str, ...] = (
    "en-US-female-neutral",
    "en-US-male-neutral",
    "en-GB-female",
    "en-IN-male",
)

_WORD = re.compile(r"[\w']+|[^\w\s]|\s+")


def degrade(text: str, condition: Condition, *, seed: int) -> tuple[str, float]:
    """Return what the ASR would have produced, and the confidence it would have reported.

    Deterministic in `seed`: the same case under the same condition produces the same damaged
    sentence on every run, so a failure can be reproduced and read. A randomised eval that
    cannot be replayed is a source of anecdotes.
    """
    rng = random.Random(f"{condition.name}:{seed}:{text}")
    tokens = _WORD.findall(text)

    kept: list[str] = []
    for token in tokens:
        lower = token.lower()
        if token.strip() and lower in DROPPABLE and rng.random() < condition.drop:
            continue
        if lower in NUMBER_CONFUSIONS and rng.random() < condition.confuse_number:
            token = _match_case(token, NUMBER_CONFUSIONS[lower])
        elif lower in HOMOPHONES and rng.random() < condition.homophone:
            token = _match_case(token, HOMOPHONES[lower])
        kept.append(token)

    if condition.truncate_tail and rng.random() < condition.truncate_tail:
        words = [i for i, t in enumerate(kept) if t.strip() and t[0].isalnum()]
        if len(words) > 3:
            # Cut a quarter to a half of the words, never the whole sentence: an utterance the
            # line ate completely is silence, and silence is a different test (the caller says
            # nothing) that the reducer already has coverage for.
            cut = rng.randint(len(words) // 4, len(words) // 2)
            kept = kept[: words[len(words) - cut]]

    degraded = "".join(kept).strip()
    # Squeeze the double spaces a dropped word leaves behind — a real ASR emits a clean string,
    # and leaving them in would let a case fail on whitespace rather than on meaning.
    degraded = re.sub(r"\s{2,}", " ", degraded)
    return degraded or text, condition.confidence


def _match_case(original: str, replacement: str) -> str:
    if original.isupper():
        return replacement.upper()
    if original[:1].isupper():
        return replacement.capitalize()
    return replacement

"""Phase 16 — the Tier-3 channel model.

`eval/tier3_audio/channel.py` decides what the agent "hears", so a bug in it turns the whole
tier into noise: too gentle and every case passes and proves nothing, too destructive and every
case fails and proves nothing either. These tests pin the properties the sweep depends on.

The model is pure and seeded, which is the property that matters most — an eval whose failures
cannot be reproduced produces anecdotes, not findings.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))

from tier3_audio.channel import (  # noqa: E402
    CONDITIONS,
    DROPPABLE,
    HOMOPHONES,
    NUMBER_CONFUSIONS,
    degrade,
)

SENTENCE = "My date of birth is December fifteenth nineteen ninety and I can't do Thursday"


def test_the_same_seed_reproduces_the_same_damage():
    """A failing case has to be readable afterwards. Without this the tier reports that
    something broke on a sentence nobody can reconstruct."""
    a = degrade(SENTENCE, CONDITIONS["noisy"], seed=7)
    b = degrade(SENTENCE, CONDITIONS["noisy"], seed=7)
    assert a == b


def test_different_seeds_produce_different_damage():
    """Otherwise the sweep runs one sentence nineteen times."""
    variants = {degrade(SENTENCE, CONDITIONS["noisy"], seed=s)[0] for s in range(20)}
    assert len(variants) > 3, f"the channel produced only {len(variants)} distinct outputs"


def test_the_clean_channel_changes_nothing():
    """The control. If a case fails on clean it is a dialogue defect, and the run says so and
    exits non-zero rather than blaming the audio."""
    text, confidence = degrade(SENTENCE, CONDITIONS["clean"], seed=1)
    assert text == SENTENCE
    assert confidence > 0.95


def test_damage_increases_with_the_condition():
    """mild < noisy, measured over many seeds rather than asserted about one.

    A single sentence can come through a noisy channel untouched — that is what per-word
    probabilities mean — so a one-shot assertion here would be flaky by construction.
    """
    def changed(name: str) -> int:
        return sum(1 for s in range(60)
                   if degrade(SENTENCE, CONDITIONS[name], seed=s)[0] != SENTENCE)

    assert changed("clean") == 0
    assert changed("mild") < changed("noisy")
    assert changed("noisy") > 30, "the noisy channel barely damages anything"


def test_it_never_returns_an_empty_utterance():
    """A line that ate the whole sentence is SILENCE, which is a different test — the caller
    said nothing — and the reducer already covers it. An empty string here would quietly turn
    a channel case into a no-input case and pass for the wrong reason."""
    for name in CONDITIONS:
        for seed in range(40):
            text, _ = degrade("Yeah", CONDITIONS[name], seed=seed)
            assert text.strip()
            text, _ = degrade(SENTENCE, CONDITIONS[name], seed=seed)
            assert text.strip()


def test_confidence_tracks_the_condition():
    """It is not decoration: sustained low ASR confidence is one of the Phase-15 ladder's
    hand-off triggers, so the noisy condition has to actually exercise it."""
    assert CONDITIONS["clean"].confidence > CONDITIONS["mild"].confidence
    assert CONDITIONS["mild"].confidence > CONDITIONS["noisy"].confidence
    assert CONDITIONS["noisy"].confidence < 0.8


# --- the specific confusions, because these are the ones that cost a caller something -------


def test_ordinals_are_confusable_not_just_cardinals():
    """A caller says "December fifteenth", never "December fifteen". Dates of birth and
    appointment days are almost entirely ordinals, and they are exactly where being wrong by a
    decade is not a rounding error. The first preview run showed this gap."""
    assert NUMBER_CONFUSIONS["fifteenth"] == "fiftieth"
    assert NUMBER_CONFUSIONS["fiftieth"] == "fifteenth"
    for word in ("thirteenth", "sixteenth", "nineteenth", "thirtieth", "ninetieth"):
        assert word in NUMBER_CONFUSIONS, word


def test_the_teens_tens_confusion_goes_both_ways():
    for teen, ten in (("fifteen", "fifty"), ("nineteen", "ninety")):
        assert NUMBER_CONFUSIONS[teen] == ten
        assert NUMBER_CONFUSIONS[ten] == teen


def test_negation_can_be_lost():
    """The corruption with the sharpest edge: "can't" -> "can" inverts the sentence.

    `core/intent.NEGATABLE_PATTERNS` and the narrow guard around it exist because of this class
    of ASR error, so the channel has to be able to produce it.
    """
    assert HOMOPHONES["can't"] == "can"
    heard = {degrade("I can't do Thursday", CONDITIONS["noisy"], seed=s)[0] for s in range(60)}
    assert any("can do Thursday" in h for h in heard), "the channel never inverted a negation"


def test_only_unstressed_words_are_dropped():
    """Dropping content words would be a different failure — a line that loses "Thursday"
    tests nothing about dialogue recovery, it just deletes the request."""
    for word in ("thursday", "appointment", "birth", "december", "cancel", "refill"):
        assert word not in DROPPABLE


@pytest.mark.parametrize("original, expected", [("Fifteenth", "Fiftieth"), ("CAN'T", "CAN")])
def test_capitalisation_survives_a_substitution(original, expected):
    """A real ASR does not shout mid-sentence. Case damage would make failures look like
    channel damage when they are an artifact of this model."""
    from tier3_audio.channel import _match_case

    assert _match_case(original, expected.lower()) == expected


def test_every_condition_is_named_the_same_thing_it_is_keyed_by():
    """The report table is keyed by the dict key and the runner logs `condition.name`; a
    mismatch produces a table that quietly attributes results to the wrong channel."""
    for key, condition in CONDITIONS.items():
        assert condition.name == key
        assert condition.note, f"{key} has no note — the table prints it as the explanation"

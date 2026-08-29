"""Has the caller said goodbye?

Through Phase 12 the agent could not end a call. ``EndCall`` had exactly one producer — the
caller hanging up — so every completed booking finished like this:

    agent:  "You're all set... Is there anything else?"
    caller: "No. Thank you."
    agent:  "You're welcome! Take care, and we'll see you soon!"
    ...16 seconds of silence, until the caller gave up and hung up themselves.

Leaving a caller holding a dead line after a successful booking is a bad last impression, and
on telephony it is also billed time.

Deterministic on purpose, and gated on ``state.booked``: hanging up is irreversible, so the
decision does not go to a model. The narrow rule below only ever runs once an appointment is
actually committed, which means the worst case for a false positive is ending a call that had
already achieved everything it set out to do.
"""

from __future__ import annotations

import re

# Phrases that end a call once the business is done. Deliberately conservative — each is a
# complete answer to "is there anything else?", not a fragment that might continue.
_FAREWELL = re.compile(
    r"""
    ^(?:
        (?:no[\s,.]*)?(?:thank\s*you|thanks)(?:\s+so\s+much)?      # "no thank you", "thanks"
      | no[\s,.]*(?:i'?m\s+)?(?:good|fine|all\s+set|that'?s\s+all|nothing\s+else)
      | (?:that'?s|thats)\s+(?:all|it|everything|perfect)
      | nothing\s+else
      | i'?m\s+(?:good|fine|all\s+set|done)
      | all\s+set
      | (?:good)?bye
      | have\s+a\s+(?:good|great|nice)\s+(?:day|one|night)
      | see\s+you(?:\s+(?:then|soon|monday|tuesday|wednesday|thursday|friday))?
    )
    [\s,.!]*$
    """,
    re.IGNORECASE | re.VERBOSE,
)


def is_farewell(text: str) -> bool:
    """True when the caller's whole turn is a sign-off and nothing else.

    Anchored at both ends on purpose. "No, thank you" ends a call; "no thank you, but could I
    also ask about parking" does not, and a substring match would cut that caller off
    mid-sentence.

    >>> is_farewell("No. Thank you.")
    True
    >>> is_farewell("That's all, thanks!")
    False
    >>> is_farewell("No thank you, can I also change the time?")
    False
    >>> is_farewell("no")
    False
    """
    stripped = " ".join(text.split())
    if not stripped:
        return False
    return bool(_FAREWELL.match(stripped))

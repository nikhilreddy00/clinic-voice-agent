"""Phase 12 — model tier routing.

One question, answered by a table rather than scattered through dialogue code: *which model
should serve this turn?* Classifying an intent and reading back a yes/no do not need the same
model as handling a complaint before an escalation decision, and paying the strong model's
latency on every turn is the single biggest lever on a voice agent's responsiveness.

Three tiers, defined by what they are for rather than by vendor:

    FAST      classification, slot extraction, yes/no confirmation. Latency is the product.
    STANDARD  ordinary dialogue. The default.
    STRONG    ambiguity, complaints, and anything immediately preceding an escalation —
              the turns where being wrong is expensive and 300 ms is not.

Two design commitments:

**Routing decisions are events.** Every choice lands in the call trace with its tier and its
reason, so "why was this turn slow / why did it answer like that" is answerable after the fact
instead of being a property of code someone has to re-read.

**Per-provider cache semantics live here, not in dialogue code.** Anthropic needs explicit
``cache_control`` blocks, OpenAI caches automatically, Groq and Cerebras differ again. That
difference is real and it must not leak upward — the reducer should never contain a sentence
about a vendor's caching model.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum

from .intent import Intent


class Tier(str, Enum):
    FAST = "fast"
    STANDARD = "standard"
    STRONG = "strong"


@dataclass(frozen=True)
class ModelSpec:
    """A concrete model behind a tier, plus what the adapter needs to know about it."""

    tier: Tier
    model: str
    max_tokens: int
    # Minimum cacheable prefix in tokens. Phase 8 measured this the hard way: it is NOT
    # monotonic across models (512 on Opus 5, 1,024 on Sonnet 5/4.6 and Opus 4.8, 4,096 on
    # Haiku 4.5), and below the threshold Anthropic ACCEPTS a cache_control breakpoint and then
    # silently caches nothing. So caching viability is a property of the model choice, which is
    # to say a property of this table.
    cache_min_tokens: int = 4096

    def caches_at(self, prefix_tokens: int) -> bool:
        return prefix_tokens >= self.cache_min_tokens


# Default fleet. Every tier currently points at Haiku 4.5 because that is what Phase 3 settled
# on and Phase 8's bake-off has not been run — pretending otherwise would be inventing a
# decision. The table exists so that switching a tier is a one-line change with a measured
# justification, not a refactor.
DEFAULT_MODELS: dict[Tier, ModelSpec] = {
    Tier.FAST: ModelSpec(Tier.FAST, "claude-haiku-4-5-20251001", max_tokens=128, cache_min_tokens=4096),
    Tier.STANDARD: ModelSpec(Tier.STANDARD, "claude-haiku-4-5-20251001", max_tokens=1024, cache_min_tokens=4096),
    Tier.STRONG: ModelSpec(Tier.STRONG, "claude-haiku-4-5-20251001", max_tokens=1024, cache_min_tokens=4096),
}

# Intents where getting it wrong is expensive enough to spend the strong tier. Billing and
# clinical questions both end in a hand-off decision, and a bad hand-off is what a clinic
# actually complains about.
_STRONG_INTENTS = frozenset(
    {
        Intent.CLINICAL_QUESTION,
        Intent.BILLING_QUESTION,
        Intent.SPEAK_TO_HUMAN,
    }
)


def select_tier(
    *,
    intent: Intent | None = None,
    turn_index: int = 0,
    degraded: tuple[str, ...] = (),
    escalating: bool = False,
) -> tuple[Tier, str]:
    """Choose a tier and say why. **Pure** — no env, no I/O, no clock.

    Split out from :class:`LLMRouter` on purpose. Which *tier* a turn deserves is a dialogue
    decision, so the reducer makes it and it replays deterministically. Which *model* sits
    behind that tier is a deployment detail that reads environment variables, so it stays in
    the adapter. Collapsing the two would drag `os.getenv` into the reducer and make a recorded
    call replay differently on a machine with different env vars.
    """
    if escalating:
        tier, reason = Tier.STRONG, "escalation decision"
    elif intent is None or intent is Intent.UNKNOWN:
        # No intent yet means turn one, or a genuinely ambiguous utterance. Standard rather
        # than strong: this is the most latency-sensitive turn in the call (it is the caller's
        # first impression) and it is usually trivial.
        tier, reason = Tier.STANDARD, "intent not yet known"
    elif intent is Intent.EMERGENCY:
        # Recorded for completeness only. The emergency path is scripted and never reaches a
        # model — see core/intent.detect_emergency.
        tier, reason = Tier.FAST, "emergency path is scripted; no model in the loop"
    elif intent in _STRONG_INTENTS:
        tier, reason = Tier.STRONG, f"{intent.value} precedes a hand-off decision"
    else:
        tier, reason = Tier.STANDARD, f"{intent.value} dialogue"

    # Degradation drops a tier rather than failing: a slower-but-answered turn beats dead air.
    # The retry/breaker ladder proper is Phase 15; this is the hook it will hang from.
    if "llm" in degraded and tier is Tier.STRONG:
        tier, reason = Tier.STANDARD, f"{reason}; downgraded (llm degraded)"

    return tier, reason


@dataclass(frozen=True)
class RoutingDecision:
    tier: Tier
    model: str
    max_tokens: int
    reason: str

    def as_dict(self) -> dict:
        return {
            "tier": self.tier.value,
            "model": self.model,
            "max_tokens": self.max_tokens,
            "reason": self.reason,
        }


class LLMRouter:
    """Chooses a model per turn from ``(intent, complexity, provider health)``."""

    def __init__(self, models: dict[Tier, ModelSpec] | None = None) -> None:
        self.models = dict(models or DEFAULT_MODELS)
        self._apply_env_overrides()

    def _apply_env_overrides(self) -> None:
        """``CLINIC_MODEL_FAST`` / ``_STANDARD`` / ``_STRONG`` pin a tier without a code change.

        This is what makes the Phase-8 bake-off runnable against the live agent: point a tier at
        a candidate, run the eval, compare. Without it, every comparison is a commit.
        """
        for tier in Tier:
            override = os.getenv(f"CLINIC_MODEL_{tier.value.upper()}")
            if override:
                spec = self.models[tier]
                self.models[tier] = ModelSpec(
                    tier=tier,
                    model=override,
                    max_tokens=spec.max_tokens,
                    # An unknown model's cache floor is unknown. Assume the most conservative
                    # value seen in this family rather than quietly claiming caching works.
                    cache_min_tokens=4096,
                )

    def resolve(self, tier: Tier, reason: str = "") -> RoutingDecision:
        """Map a tier the reducer already chose onto a concrete model."""
        spec = self.models[tier]
        return RoutingDecision(
            tier=tier, model=spec.model, max_tokens=spec.max_tokens, reason=reason
        )

    def route(self, **kwargs) -> RoutingDecision:
        """Select a tier and resolve it in one step, for callers outside the reducer."""
        tier, reason = select_tier(**kwargs)
        return self.resolve(tier, reason)


    def classifier_spec(self) -> ModelSpec:
        """The model used for intent classification — always the fast tier, by definition."""
        return self.models[Tier.FAST]

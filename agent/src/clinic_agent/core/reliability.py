"""Phase 15 — the two primitives every adapter's failover needs.

Through Phase 14 ``state.degraded`` was a write-only field: ``reducer._on_provider_degraded``
could add a provider and nothing could ever remove one, no adapter retried anything, and
``stt.py`` said it plainly — *"nothing reconnects, so a dead socket means a call that hears
nothing and looks fine."* This module is the missing half.

Two things live here and nothing else:

``CircuitBreaker``
    Consecutive failures trip it; after a cooldown one probe is allowed through; a success
    closes it. **The clock is a parameter, never read here.** Same rule as the reducer: a
    component that reads ``time.monotonic()`` internally cannot be tested without sleeping and
    cannot be replayed at all.

``is_retryable``
    One definition of "worth trying again", shared by the LLM adapter and the scheduling
    client. Retrying a 400 is pointless and retrying a 401 is worse than pointless; what is
    worth a second attempt is a timeout, a connection reset, a 429, and a 5xx. Keeping this in
    one place is the difference between a policy and two drifting opinions.
"""

from __future__ import annotations

from dataclasses import dataclass

# Anything at or above this is the provider's problem, not the request's.
_RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})

# Substrings that identify a transient failure when no status code is available (websocket
# closes, DNS blips, connector errors). Matched case-insensitively against str(exc).
_RETRYABLE_TEXT = (
    "timeout",
    "timed out",
    "connection reset",
    "connection refused",
    "connection closed",
    "connection error",
    "temporarily unavailable",
    "overloaded",
    "rate limit",
    "server error",
    "bad gateway",
    "service unavailable",
)


def is_retryable(exc: BaseException) -> bool:
    """Is this exception worth exactly one more attempt?

    Status code first when the SDK exposes one (both ``anthropic`` and ``httpx`` errors carry
    ``status_code`` on the exception or its ``.response``), then a text match for transport
    failures that never got as far as a response.

    >>> is_retryable(TimeoutError("request timed out"))
    True
    >>> is_retryable(ValueError("invalid api key"))
    False
    """
    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    if isinstance(status, int):
        return status in _RETRYABLE_STATUS

    text = str(exc).lower()
    return any(marker in text for marker in _RETRYABLE_TEXT)


@dataclass
class CircuitBreaker:
    """Consecutive-failure breaker with a single half-open probe.

    Deliberately *consecutive* rather than a rolling error rate: a voice call is short (~20
    turns) and a rate over a handful of samples is noise. Three failures in a row is a
    provider that is down; three failures spread across a call that otherwise worked is not,
    and a success resets the count.

    ``now`` is passed to every method that needs it. Nothing here reads a clock.

    >>> b = CircuitBreaker("llm", threshold=2, cooldown_s=10)
    >>> b.allow(0.0)
    True
    >>> b.record_failure(0.0); b.record_failure(1.0); b.state
    'open'
    >>> b.allow(2.0)            # still cooling down
    False
    >>> b.allow(12.0)           # one probe gets through
    True
    >>> b.allow(12.1)           # ...but only one
    False
    >>> b.record_success(); b.state
    'closed'
    """

    name: str = "provider"
    threshold: int = 3
    cooldown_s: float = 20.0

    failures: int = 0
    opened_at: float | None = None
    _probe_in_flight: bool = False

    @property
    def state(self) -> str:
        """``closed`` | ``open`` | ``half_open`` — reporting only; ``allow`` owns the decision."""
        if self.opened_at is None:
            return "closed"
        return "half_open" if self._probe_in_flight else "open"

    def allow(self, now: float) -> bool:
        """May a request go out at ``now``?

        Closed: yes. Open and still cooling: no. Open and cooled down: yes, once — the probe.
        A probe that never reports back leaves the breaker open, which is the safe direction:
        a caller waits on the ladder's next rung rather than on a provider that is not
        answering.
        """
        if self.opened_at is None:
            return True
        if self._probe_in_flight:
            return False
        if now - self.opened_at < self.cooldown_s:
            return False
        self._probe_in_flight = True
        return True

    def record_success(self) -> None:
        """Close the breaker and forget the streak."""
        self.failures = 0
        self.opened_at = None
        self._probe_in_flight = False

    def record_failure(self, now: float) -> bool:
        """Record a failure; returns True if this is the transition that opened the breaker.

        A failed probe re-opens with a fresh cooldown rather than immediately allowing
        another, which is what stops a dead provider from being hammered once per cooldown
        forever while the caller listens to silence.
        """
        was_open = self.opened_at is not None
        self.failures += 1
        self._probe_in_flight = False
        if was_open:
            self.opened_at = now
            return False
        if self.failures >= self.threshold:
            self.opened_at = now
            return True
        return False

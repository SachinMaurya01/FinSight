"""Thread-safe per-provider circuit breaker."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

from src.config import ProviderName, Settings


class CircuitOpenError(RuntimeError):
    """Raised when a provider is skipped because its breaker is open."""


class CircuitBreaker:
    """Open after repeated failures; skip calls until the cool-down elapses."""

    def __init__(
        self,
        failure_threshold: int = 5,
        cooldown_s: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.cooldown_s = cooldown_s
        self._clock = clock
        self.failures = 0
        self.opened_at: float | None = None
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        with self._lock:
            if self.opened_at is None:
                return "closed"
            if self._clock() - self.opened_at > self.cooldown_s:
                return "half_open"
            return "open"

    def allow(self) -> bool:
        """Whether a call may proceed right now."""
        with self._lock:
            if self.opened_at is None:
                return True
            if self._clock() - self.opened_at > self.cooldown_s:
                # Half-open: allow a probe and reset the failure counter.
                self.opened_at = None
                self.failures = 0
                return True
            return False

    def record(self, ok: bool) -> None:
        """Record the outcome of a provider call."""
        with self._lock:
            if ok:
                self.failures = 0
                self.opened_at = None
                return
            self.failures += 1
            if self.failures >= self.failure_threshold:
                self.opened_at = self._clock()


_breakers: dict[str, CircuitBreaker] = {}
_breakers_lock = threading.Lock()


def get_breaker(provider: ProviderName | str, settings: Settings) -> CircuitBreaker:
    """Return the shared breaker for a provider, creating it on first use."""
    with _breakers_lock:
        breaker = _breakers.get(provider)
        if breaker is None:
            breaker = CircuitBreaker(
                failure_threshold=settings.circuit_breaker_failures,
                cooldown_s=settings.circuit_breaker_cooldown_seconds,
            )
            _breakers[provider] = breaker
        return breaker


def reset_breakers() -> None:
    """Clear the breaker registry (tests / process restarts)."""
    with _breakers_lock:
        _breakers.clear()

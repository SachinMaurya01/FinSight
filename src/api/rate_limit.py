"""Per-key rate limiting and concurrency controls for the HTTP API."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Callable

from fastapi import Depends, HTTPException, Request

from src.api.auth import AuthContext, require_api_key
from src.config import Settings
from src.config import settings as default_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RateLimitDecision:
    """Result of one fixed-window rate-limit check."""

    allowed: bool
    remaining: int
    retry_after: int
    reset_after: int


class FixedWindowRateLimiter:
    """Redis-backed fixed-window limiter with an in-memory fallback."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        redis_client: Any | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._settings = settings or default_settings
        self._redis_client = redis_client
        self._redis_checked = redis_client is not None
        self._clock = clock
        self._lock = threading.Lock()
        self._buckets: dict[str, tuple[int, int]] = {}

    def _redis(self) -> Any | None:
        if self._redis_checked:
            return self._redis_client
        self._redis_checked = True
        try:
            import redis  # type: ignore[import-not-found]

            client = redis.Redis.from_url(
                self._settings.redis_url,
                decode_responses=True,
                socket_connect_timeout=1,
                socket_timeout=1,
            )
            client.ping()
            self._redis_client = client
        except Exception as exc:
            logger.debug("Rate-limit Redis unavailable; using memory fallback: %s", exc)
            self._redis_client = None
        return self._redis_client

    def _redis_allow(self, identity: str, limit: int, bucket: int) -> RateLimitDecision | None:
        client = self._redis()
        if client is None:
            return None
        key = f"finsight:ratelimit:{identity}:{bucket}"
        try:
            count = int(client.incr(key))
            if count == 1:
                client.expire(key, 65)
        except Exception as exc:
            logger.warning("Rate-limit Redis check failed; using memory fallback: %s", exc)
            self._redis_client = None
            return None
        now = self._clock()
        reset_after = max(1, int(60 - (now % 60)))
        return RateLimitDecision(
            allowed=count <= limit,
            remaining=max(0, limit - count),
            retry_after=0 if count <= limit else reset_after,
            reset_after=reset_after,
        )

    def _memory_allow(self, identity: str, limit: int, bucket: int) -> RateLimitDecision:
        with self._lock:
            if len(self._buckets) > 10_000:
                self._buckets = {
                    key: value for key, value in self._buckets.items() if value[0] >= bucket - 1
                }
            previous_bucket, previous_count = self._buckets.get(identity, (bucket, 0))
            count = previous_count + 1 if previous_bucket == bucket else 1
            self._buckets[identity] = (bucket, count)

        reset_after = max(1, int(60 - (self._clock() % 60)))
        return RateLimitDecision(
            allowed=count <= limit,
            remaining=max(0, limit - count),
            retry_after=0 if count <= limit else reset_after,
            reset_after=reset_after,
        )

    def allow(self, identity: str, limit: int) -> RateLimitDecision:
        """Count one request for ``identity`` in the current one-minute window."""
        bucket = int(self._clock() // 60)
        decision = self._redis_allow(identity, limit, bucket)
        if decision is not None:
            return decision
        return self._memory_allow(identity, limit, bucket)


class ConcurrencyLimiter:
    """Per-key in-flight request limiter."""

    def __init__(self, max_concurrent: int = 4) -> None:
        self._max_concurrent = max_concurrent
        self._lock = threading.Lock()
        self._active: dict[str, int] = {}

    def try_acquire(self, identity: str, limit: int | None = None) -> bool:
        cap = max(1, limit or self._max_concurrent)
        with self._lock:
            active = self._active.get(identity, 0)
            if active >= cap:
                return False
            self._active[identity] = active + 1
            return True

    def release(self, identity: str) -> None:
        with self._lock:
            active = self._active.get(identity, 0)
            if active <= 1:
                self._active.pop(identity, None)
            else:
                self._active[identity] = active - 1

    def active(self, identity: str) -> int:
        with self._lock:
            return self._active.get(identity, 0)


_RATE_LIMITER: FixedWindowRateLimiter | None = None
_CONCURRENCY_LIMITER: ConcurrencyLimiter | None = None


def get_rate_limiter(settings: Settings | None = None) -> FixedWindowRateLimiter:
    global _RATE_LIMITER
    if _RATE_LIMITER is None:
        _RATE_LIMITER = FixedWindowRateLimiter(settings)
    return _RATE_LIMITER


def get_concurrency_limiter(settings: Settings | None = None) -> ConcurrencyLimiter:
    global _CONCURRENCY_LIMITER
    if _CONCURRENCY_LIMITER is None:
        target = settings or default_settings
        _CONCURRENCY_LIMITER = ConcurrencyLimiter(target.max_concurrent_requests_per_key)
    return _CONCURRENCY_LIMITER


async def enforce_request_limits(
    request: Request,
    auth: AuthContext = Depends(require_api_key),
) -> AsyncIterator[AuthContext]:
    """Apply per-key rate and concurrency limits, yielding the authenticated caller."""
    rate_limiter: FixedWindowRateLimiter = getattr(request.app.state, "rate_limiter", None) or get_rate_limiter(
        default_settings
    )
    limit = auth.rate_limit_per_minute or default_settings.rate_limit_requests_per_minute
    decision = rate_limiter.allow(auth.identity, limit)
    headers = {
        "X-RateLimit-Limit": str(limit),
        "X-RateLimit-Remaining": str(decision.remaining),
        "X-RateLimit-Reset": str(decision.reset_after),
    }
    if not decision.allowed:
        raise HTTPException(
            status_code=429,
            detail="rate limit exceeded",
            headers={**headers, "Retry-After": str(decision.retry_after)},
        )

    concurrency: ConcurrencyLimiter = getattr(
        request.app.state, "concurrency_limiter", None
    ) or get_concurrency_limiter(default_settings)
    concurrency_limit = default_settings.max_concurrent_requests_per_key
    if not concurrency.try_acquire(auth.identity, concurrency_limit):
        raise HTTPException(
            status_code=429,
            detail="too many concurrent requests",
            headers={**headers, "Retry-After": "1"},
        )

    try:
        yield auth
    finally:
        concurrency.release(auth.identity)

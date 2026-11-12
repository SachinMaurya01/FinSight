"""Provider call wrapper: timeout-aware retries plus circuit breaking."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import httpx
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_random_exponential

from src.config import ProviderName, Settings
from src.resilience.breaker import CircuitOpenError, get_breaker

logger = logging.getLogger(__name__)

_TRANSIENT_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


def is_transient(exc: BaseException) -> bool:
    """Whether an exception is safe to retry (timeouts, rate limits, 5xx)."""
    if isinstance(exc, (TimeoutError, ConnectionError, httpx.TimeoutException, httpx.ConnectError)):
        return True
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and status in _TRANSIENT_STATUS:
        return True
    try:
        import openai

        if isinstance(
            exc,
            (
                openai.RateLimitError,
                openai.APITimeoutError,
                openai.APIConnectionError,
            ),
        ):
            return True
        if isinstance(exc, openai.APIStatusError):
            return exc.status_code in _TRANSIENT_STATUS
    except ImportError:  # pragma: no cover - openai is a hard dependency
        pass
    return False


def invoke_with_resilience(
    provider: ProviderName | str,
    fn: Callable[..., Any],
    *args: Any,
    settings: Settings,
    callbacks: list[Any] | None = None,
    **kwargs: Any,
) -> Any:
    """Call a provider function with retries and a per-provider circuit breaker."""
    breaker = get_breaker(provider, settings)
    if not breaker.allow():
        raise CircuitOpenError(f"circuit open for provider {provider}")

    invoke_kwargs = dict(kwargs)
    if callbacks:
        invoke_kwargs["config"] = {"callbacks": callbacks}

    retryer = Retrying(
        stop=stop_after_attempt(settings.llm_max_attempts),
        wait=wait_random_exponential(
            multiplier=settings.retry_backoff_multiplier,
            max=settings.retry_backoff_max_seconds,
        ),
        retry=retry_if_exception(is_transient),
        reraise=True,
    )

    try:
        result = retryer(fn, *args, **invoke_kwargs)
    except Exception:
        breaker.record(False)
        raise
    breaker.record(True)
    return result

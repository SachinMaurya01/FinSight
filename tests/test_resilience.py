"""Retry, timeout, circuit breaker, and tier-model mapping tests."""

from __future__ import annotations

import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Settings
from src.resilience.breaker import CircuitBreaker, CircuitOpenError, get_breaker, reset_breakers
from src.resilience.retry import invoke_with_resilience, is_transient

SETTINGS = Settings(
    postgres_database_url="postgresql://localhost:5432/finsight_test",
    llm_max_attempts=3,
    retry_backoff_multiplier=0.0,
    retry_backoff_max_seconds=0.0,
    circuit_breaker_failures=2,
    circuit_breaker_cooldown_seconds=10.0,
)


def test_is_transient() -> None:
    assert is_transient(TimeoutError())
    assert is_transient(ConnectionError())
    assert is_transient(httpx.ConnectTimeout("timeout", request=httpx.Request("GET", "https://x")))
    assert not is_transient(ValueError("bad input"))
    print("PASS test_is_transient")


def test_circuit_breaker_states() -> None:
    now = [0.0]
    breaker = CircuitBreaker(failure_threshold=2, cooldown_s=10, clock=lambda: now[0])
    assert breaker.allow() and breaker.state == "closed"
    breaker.record(False)
    assert breaker.allow()
    breaker.record(False)
    assert not breaker.allow() and breaker.state == "open"

    now[0] = 11.0
    assert breaker.state == "half_open"
    assert breaker.allow()
    breaker.record(True)
    assert breaker.state == "closed" and breaker.failures == 0
    print("PASS test_circuit_breaker_states")


def test_invoke_retries_transient_only() -> None:
    reset_breakers()
    attempts: list[int] = []

    def flaky() -> str:
        attempts.append(len(attempts))
        if len(attempts) < 3:
            raise TimeoutError("retry me")
        return "ok"

    assert invoke_with_resilience("openai", flaky, settings=SETTINGS) == "ok"
    assert len(attempts) == 3
    assert get_breaker("openai", SETTINGS).state == "closed"

    bad_attempts: list[int] = []

    def bad() -> str:
        bad_attempts.append(1)
        raise ValueError("do not retry")

    try:
        invoke_with_resilience("groq", bad, settings=SETTINGS)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
    assert len(bad_attempts) == 1
    reset_breakers()
    print("PASS test_invoke_retries_transient_only")


def test_open_circuit_blocks_calls() -> None:
    reset_breakers()
    breaker = get_breaker("gemini", SETTINGS)
    breaker.record(False)
    breaker.record(False)
    called: list[bool] = []

    def should_not_run() -> str:
        called.append(True)
        return "no"

    try:
        invoke_with_resilience("gemini", should_not_run, settings=SETTINGS)
    except CircuitOpenError:
        pass
    else:
        raise AssertionError("expected CircuitOpenError")
    assert called == []
    reset_breakers()
    print("PASS test_open_circuit_blocks_calls")


def test_provider_tier_model_mapping() -> None:
    primary = SETTINGS.tier_model_for_provider("simple", "openai")
    assert primary.provider == "openai" and primary.model_name == "gpt-4o-mini"
    fallback = SETTINGS.tier_model_for_provider("simple", "groq")
    assert fallback.provider == "groq" and fallback.model_name == "qwen/qwen3.5-9b"
    complex_model = SETTINGS.tier_model_for_provider("complex", "gemini")
    assert complex_model.model_name == "gemini-2.5-pro"
    try:
        SETTINGS.tier_model_for_provider("simple", "nope")  # type: ignore[arg-type]
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unmapped provider")
    print("PASS test_provider_tier_model_mapping")


def main() -> int:
    test_is_transient()
    test_circuit_breaker_states()
    test_invoke_retries_transient_only()
    test_open_circuit_blocks_calls()
    test_provider_tier_model_mapping()
    return 0


if __name__ == "__main__":
    sys.exit(main())

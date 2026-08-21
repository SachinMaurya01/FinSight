"""Phase 10 — fallback chain unit tests (fake providers, no network).

Run with: python tests/test_fallback.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Settings
from src.llm import fallback as fb
from src.llm.client import ProviderUnavailableError
from src.llm.fallback import DegradedResponseError, call_with_fallback

TEST_SETTINGS = Settings(
    openai_api_key="sk-fake",
    groq_api_key="gsk-fake",
    google_api_key=None,
    postgres_database_url="postgresql://localhost:5432/finsight_test",
    fallback_chain=["openai", "groq", "gemini"],
)


class _FakeLLM:
    def __init__(self, content="answer [1]", exc=None) -> None:
        self._content = content
        self._exc = exc

    def invoke(self, prompt):
        from types import SimpleNamespace

        if self._exc is not None:
            raise self._exc
        return SimpleNamespace(content=self._content)


def _http_response(status: int):
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    return httpx.Response(status, request=request)


def _rate_limit():
    return fb.OpenAIRateLimitError("rate limited", response=_http_response(429), body=None)


def test_falls_back_to_next_provider_on_error() -> None:
    calls: list[str] = []

    def fake_build_llm(settings, provider, model_name=None):
        calls.append(provider)
        if provider == "openai":
            return _FakeLLM(exc=_rate_limit())
        return _FakeLLM()

    original = fb.build_llm
    fb.build_llm = fake_build_llm
    try:
        result = call_with_fallback("q", TEST_SETTINGS)
    finally:
        fb.build_llm = original

    assert calls == ["openai", "groq"], calls
    assert result.provider == "groq"
    assert result.content == "answer [1]"
    assert result.tier is None
    assert result.failures == (("openai", "RateLimitError"),)
    print("PASS test_falls_back_to_next_provider_on_error")


def test_tier_routes_primary_provider_first() -> None:
    calls: list[str] = []
    models: list[str | None] = []

    def fake_build_llm(settings, provider, model_name=None):
        calls.append(provider)
        models.append(model_name)
        return _FakeLLM()

    original = fb.build_llm
    fb.build_llm = fake_build_llm
    try:
        result = call_with_fallback("q", TEST_SETTINGS, tier="simple")
    finally:
        fb.build_llm = original

    assert calls[0] == "groq", calls  # simple tier -> groq primary
    assert models[0] == "openai/gpt-oss-20b", models
    assert result.provider == "groq"
    assert result.tier == "simple"
    print("PASS test_tier_routes_primary_provider_first")


def test_skips_providers_without_credentials() -> None:
    def fake_build_llm(settings, provider, model_name=None):
        if provider == "gemini":
            raise ProviderUnavailableError("google_api_key not set")
        return _FakeLLM()

    original = fb.build_llm
    fb.build_llm = fake_build_llm
    try:
        result = call_with_fallback("q", TEST_SETTINGS)
    finally:
        fb.build_llm = original

    assert result.provider == "openai"
    assert result.failures == ()
    print("PASS test_skips_providers_without_credentials")


def test_all_providers_fail_raises_degraded() -> None:
    def fake_build_llm(settings, provider, model_name=None):
        return _FakeLLM(exc=fb.OpenAIAuthenticationError(
            "bad key", response=_http_response(401), body=None))

    original = fb.build_llm
    fb.build_llm = fake_build_llm
    try:
        try:
            call_with_fallback("q", TEST_SETTINGS)
        except DegradedResponseError as exc:
            assert "all providers failed" in str(exc)
            assert "openai:AuthenticationError" in str(exc)
        else:
            raise AssertionError("expected DegradedResponseError")
    finally:
        fb.build_llm = original
    print("PASS test_all_providers_fail_raises_degraded")


def test_empty_response_treated_as_failure() -> None:
    def fake_build_llm(settings, provider, model_name=None):
        return _FakeLLM(content="  ") if provider == "openai" else _FakeLLM(content="ok")

    original = fb.build_llm
    fb.build_llm = fake_build_llm
    try:
        result = call_with_fallback("q", TEST_SETTINGS)
    finally:
        fb.build_llm = original

    assert result.provider == "groq"
    assert result.failures == (("openai", "MalformedResponseError"),)
    print("PASS test_empty_response_treated_as_failure")


def main() -> int:
    test_falls_back_to_next_provider_on_error()
    test_tier_routes_primary_provider_first()
    test_skips_providers_without_credentials()
    test_all_providers_fail_raises_degraded()
    test_empty_response_treated_as_failure()
    return 0


if __name__ == "__main__":
    sys.exit(main())
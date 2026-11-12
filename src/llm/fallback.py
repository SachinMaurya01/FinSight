"""Provider fallback chain."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel
from openai import (
    APIConnectionError as OpenAIAPIConnectionError,
)
from openai import (
    APIStatusError as OpenAIAPIStatusError,
)
from openai import (
    APITimeoutError as OpenAITimeoutError,
)
from openai import (
    AuthenticationError as OpenAIAuthenticationError,
)
from openai import (
    RateLimitError as OpenAIRateLimitError,
)

from src.config import ProviderName, Settings
from src.llm.client import ProviderUnavailableError, build_llm
from src.llm.output import strip_think_block
from src.observability.metrics import record_fallback, record_llm_usage
from src.observability.tracing import trace_callbacks
from src.resilience.breaker import CircuitOpenError, get_breaker
from src.resilience.retry import invoke_with_resilience
from src.tokens import count_tokens

logger = logging.getLogger(__name__)


class MalformedResponseError(RuntimeError):
    """Raised when a provider returns an empty/whitespace-only response."""


class DegradedResponseError(RuntimeError):
    """Raised when every provider in the fallback chain failed."""


@dataclass(frozen=True)
class LLMResult:
    content: str
    provider: str
    tier: str | None
    tokens_in: int
    failures: tuple[tuple[str, str], ...]
    tokens_out: int = 0
    model: str | None = None


def _provider_exception(name: str, exc: BaseException) -> bool:
    """Whether ``exc`` is a provider failure worth falling back on.

    Providers raise heterogeneous error classes."""
    known = (
        OpenAIRateLimitError,
        OpenAITimeoutError,
        OpenAIAPIConnectionError,
        OpenAIAuthenticationError,
        OpenAIAPIStatusError,
    )
    if isinstance(exc, known):
        logger.warning("Provider %s failed with %s: %s", name, type(exc).__name__, exc)
        return True
    if isinstance(exc, (ProviderUnavailableError, MalformedResponseError)):
        return True
    # Any other provider error also triggers a fallback hop (logged distinctly).
    logger.warning("Provider %s failed with unexpected %s: %s (falling back)", name, type(exc).__name__, exc)
    return True


def _default_model(settings: Settings, provider: ProviderName) -> str:
    configured = settings.fallback_tier_models.get(provider, {}).get("normal")
    if configured:
        return configured
    if provider == "openai":
        return settings.openai_model
    if provider == "groq":
        return settings.groq_model
    if provider == "vllm":
        return settings.vllm_model or "unknown"
    return "unknown"


def call_with_fallback(
    prompt: str,
    settings: Settings,
    tier: str | None = None,
) -> LLMResult:
    """Invoke ``prompt`` across the fallback chain; returns the first success.

    When ``tier`` is set."""
    tier_cfg = settings.tier_model(tier) if tier else None
    ordered = list(settings.fallback_chain)
    if tier_cfg is not None and tier_cfg.provider in ordered:
        ordered.remove(tier_cfg.provider)
        ordered.insert(0, tier_cfg.provider)
    logger.info(
        "Fallback chain order: %s (tier=%s)",
        " -> ".join(ordered),
        tier,
    )

    failures: list[tuple[str, str]] = []

    for index, provider in enumerate(ordered):
        next_provider = ordered[index + 1] if index + 1 < len(ordered) else "none"
        provider_tier = settings.tier_model_for_provider(tier, provider) if tier else None
        model_name = provider_tier.model_name if provider_tier else None
        if not get_breaker(provider, settings).allow():
            logger.warning("Skipping provider=%s (circuit open)", provider)
            failures.append((provider, "circuit_open"))
            record_fallback(provider, next_provider)
            continue
        try:
            llm: BaseChatModel = build_llm(settings, provider, model_name=model_name)
        except ProviderUnavailableError as exc:
            logger.info("Skipping provider=%s (no credentials): %s", provider, exc)
            failures.append((provider, "no_credentials"))
            record_fallback(provider, next_provider)
            continue

        model = model_name or _default_model(settings, provider)
        try:
            response = invoke_with_resilience(
                provider,
                llm.invoke,
                prompt,
                settings=settings,
                callbacks=trace_callbacks(),
            )
            content = strip_think_block(str(response.content))
            if not content.strip():
                raise MalformedResponseError("empty/whitespace-only response")
            tokens_in = count_tokens(prompt)
            tokens_out = count_tokens(content)
            record_llm_usage(provider, model, tokens_in, tokens_out)
            logger.info(
                "Served by provider=%s model=%s tier=%s tokens_in=%d tokens_out=%d",
                provider,
                model,
                tier,
                tokens_in,
                tokens_out,
            )
            return LLMResult(
                content=content,
                provider=provider,
                tier=tier,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                model=model,
                failures=tuple(failures),
            )
        except CircuitOpenError as exc:
            logger.warning("Skipping provider=%s (circuit open): %s", provider, exc)
            failures.append((provider, "circuit_open"))
            record_fallback(provider, next_provider)
            continue
        except Exception as exc:  # fallback semantics: any provider error hops
            if _provider_exception(provider, exc):
                failures.append((provider, type(exc).__name__))
                record_fallback(provider, next_provider)
                continue
            raise

    detail = "; ".join(f"{p}:{reason}" for p, reason in failures) or "no providers configured"
    raise DegradedResponseError(f"all providers failed ({detail})")

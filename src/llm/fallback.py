"""Provider fallback chain."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel
from openai import (
    APIConnectionError as OpenAIAPIConnectionError,
)
from openai import (
    APITimeoutError as OpenAITimeoutError,
)
from openai import (
    APIStatusError as OpenAIAPIStatusError,
)
from openai import (
    AuthenticationError as OpenAIAuthenticationError,
)
from openai import (
    RateLimitError as OpenAIRateLimitError,
)

from src.config import Settings
from src.llm.client import ProviderUnavailableError, build_llm
from src.llm.output import strip_think_block
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
    logger.warning(
        "Provider %s failed with unexpected %s: %s (falling back)", name, type(exc).__name__, exc
    )
    return True


def _default_model(settings: Settings, provider: str) -> str:
    if provider == "openai":
        return settings.openai_model
    if provider == "groq":
        return settings.groq_model
    if provider == "gemini":
        return settings.openai_model
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

    for provider in ordered:
        model_name = tier_cfg.model_name if (tier_cfg and provider == tier_cfg.provider) else None
        try:
            llm: BaseChatModel = build_llm(settings, provider, model_name=model_name)
        except ProviderUnavailableError as exc:
            logger.info("Skipping provider=%s (no credentials): %s", provider, exc)
            failures.append((provider, "no_credentials"))
            continue

        try:
            response = llm.invoke(prompt)
            content = strip_think_block(str(response.content))
            if not content.strip():
                raise MalformedResponseError("empty/whitespace-only response")
            tokens = count_tokens(prompt)
            logger.info(
                "Served by provider=%s model=%s tier=%s tokens_in=%d",
                provider,
                model_name or _default_model(settings, provider),
                tier,
                tokens,
            )
            return LLMResult(
                content=content,
                provider=provider,
                tier=tier,
                tokens_in=tokens,
                failures=tuple(failures),
            )
        except Exception as exc:  # fallback semantics: any provider error hops
            if _provider_exception(provider, exc):
                failures.append((provider, type(exc).__name__))
                continue
            raise

    detail = "; ".join(f"{p}:{reason}" for p, reason in failures) or "no providers configured"
    raise DegradedResponseError(f"all providers failed ({detail})")
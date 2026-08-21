"""LLM client builders for all providers in the fallback chain (Phase 10).

Supports OpenAI, Groq, Gemini, and a local vLLM (OpenAI-compatible) endpoint
per PRD §3.4. Credentials are always env-injected; a missing credential raises
``ProviderUnavailableError`` so the fallback chain can skip to the next
provider without attempting a doomed call.
"""

from __future__ import annotations

import logging

from langchain_core.language_models.chat_models import BaseChatModel

from src.config import ProviderName, Settings

logger = logging.getLogger(__name__)


class ProviderUnavailableError(RuntimeError):
    """Raised when a provider lacks credentials/config needed to build a client."""


def build_llm(
    settings: Settings,
    provider: ProviderName,
    model_name: str | None = None,
) -> BaseChatModel:
    """Build a chat model for ``provider``; ``model_name`` overrides the default."""
    if provider == "openai":
        if not settings.openai_api_key:
            raise ProviderUnavailableError("openai_api_key not set")
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model_name or settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=0.0,
            max_tokens=2048,
        )

    if provider == "groq":
        if not settings.groq_api_key:
            raise ProviderUnavailableError("groq_api_key not set")
        from langchain_groq import ChatGroq

        logger.info("Building Groq LLM (model=%s)", model_name or settings.groq_model)
        return ChatGroq(
            api_key=settings.groq_api_key,
            model=model_name or settings.groq_model,
            temperature=0.0,
            max_tokens=2048,
        )

    if provider == "gemini":
        if not settings.google_api_key:
            raise ProviderUnavailableError("google_api_key not set")
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model_name or settings.openai_model,
            google_api_key=settings.google_api_key,
            temperature=0.0,
            max_tokens=2048,
        )

    if provider == "vllm":
        if not (settings.vllm_base_url and settings.vllm_model):
            raise ProviderUnavailableError("vllm_base_url / vllm_model not set")
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model_name or settings.vllm_model,
            base_url=settings.vllm_base_url,
            api_key=settings.vllm_api_key or "EMPTY",
            temperature=0.0,
            max_tokens=2048,
        )

    raise ValueError(f"Unsupported provider: {provider!r}")


def build_tier_llm(settings: Settings, tier: str) -> BaseChatModel:
    """Build the chat model for a complexity tier (Phase 9 routing).

    Resolves ``settings.tier_model(tier)`` and builds the model for that tier's
    provider with that tier's model name. ``tier`` must be ``simple|normal|complex``.
    """
    tier_cfg = settings.tier_model(tier)
    logger.info("Routing tier=%s -> provider=%s model=%s", tier, tier_cfg.provider, tier_cfg.model_name)
    return build_llm(settings, tier_cfg.provider, model_name=tier_cfg.model_name)
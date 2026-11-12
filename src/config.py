"""Application configuration. Loads runtime config from .env."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE_PATH = PROJECT_ROOT / ".env"

ProviderName = Literal["openai", "groq", "gemini", "vllm"]
ComplexityTier = Literal["simple", "normal", "complex"]
Environment = Literal["development", "staging", "production"]
EmbeddingProvider = Literal["openai", "gemini", "sentence_transformers"]
CheckpointerType = Literal["redis", "in_memory"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]

# Validates SEC User-Agent contains a contact email (e.g. name@domain).
_EMAIL_PATTERN = re.compile(r"\S+@\S+")


class BaseConfigSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        extra="ignore",
        frozen=True,
        env_nested_delimiter="__",
        case_sensitive=False,
    )


class ModelTierConfig(BaseModel):
    """Single model tier config."""

    model_name: str = Field(...)
    provider: ProviderName = "openai"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, gt=0)


class ModelRouting(BaseModel):
    """Tier to model mapping. ChatGPT (OpenAI) as default, Groq as fallback."""

    cheap: ModelTierConfig = ModelTierConfig(model_name="gpt-4o-mini", provider="openai")
    medium: ModelTierConfig = ModelTierConfig(model_name="gpt-4o", provider="openai")
    expensive: ModelTierConfig = ModelTierConfig(model_name="gpt-4-turbo", provider="openai")


class Settings(BaseConfigSettings):
    # App
    app_version: str = "0.1.0"
    environment: Environment = "development"
    log_level: LogLevel = "INFO"
    project_root: Path = PROJECT_ROOT  # anchors all relative paths so repo works from any cwd.
    data_dir: Path = PROJECT_ROOT / "data"  # single root for raw ingested filings and derived artifacts.
    parsed_docs_dir: Path = PROJECT_ROOT / "data" / "parsed"  # keeps parsed filings separate from read-only sources.
    chunk_store_dir: Path = PROJECT_ROOT / "data" / "chunks"  # persists chunked docs for reuse without re-parsing.

    # Default model
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    # Provider credentials (env-injected)
    openai_api_key: str | None = None
    groq_api_key: str | None = None
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_model: str = "qwen/qwen3.5-9b"
    google_api_key: str | None = None
    vllm_base_url: str = "http://localhost:11434/v1"  # local OpenAI-compatible endpoint
    vllm_api_key: str | None = None  # optional auth for secured self-hosted vLLM servers.
    vllm_model: str | None = "qwen3.5:9b"  # local Ollama model id (from `ollama list`); None disables vLLM path.

    # Model routing
    model_routing: ModelRouting = ModelRouting()  # maps simple/normal/complex queries to cheap/medium/expensive models.

    # Fallback chain
    fallback_chain: list[ProviderName] = ["openai", "groq", "gemini", "vllm"]
    # Per-provider tier models so a fallback never silently serves a weaker model.
    fallback_tier_models: dict[ProviderName, dict[str, str]] = {
        "openai": {"simple": "gpt-4o-mini", "normal": "gpt-4o", "complex": "gpt-4-turbo"},
        "groq": {
            "simple": "qwen/qwen3.5-9b",
            "normal": "qwen/qwen3.5-9b",
            "complex": "openai/gpt-oss-120b",
        },
        "gemini": {
            "simple": "gemini-2.0-flash",
            "normal": "gemini-2.0-flash",
            "complex": "gemini-2.5-pro",
        },
        "vllm": {"simple": "qwen3.5:9b", "normal": "qwen3.5:9b", "complex": "qwen3.5:9b"},
    }

    # Resilience
    llm_timeout_seconds: float = Field(default=30.0, gt=0)
    llm_max_attempts: int = Field(default=3, ge=1, le=5)
    retry_backoff_multiplier: float = Field(default=0.5, ge=0.0)
    retry_backoff_max_seconds: float = Field(default=8.0, ge=0.0)
    circuit_breaker_failures: int = Field(default=5, ge=1)
    circuit_breaker_cooldown_seconds: float = Field(default=30.0, ge=0.0)

    # Observability
    tracing_enabled: bool = True
    langfuse_enabled: bool = False
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "https://cloud.langfuse.com"

    # Embeddings
    embedding_provider: EmbeddingProvider = (
        "sentence_transformers"  # selects embedding backend (local default avoids API cost).
    )
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"  # compact local model balancing quality vs speed.
    embedding_dimensions: int = Field(default=384, gt=0)  # must match vector DB schema for similarity search.
    embedding_batch_size: int = Field(default=64, gt=0)  # batches embedding calls to limit memory and API usage.

    # Reranker
    reranker_model: str = (
        "cross-encoder/ms-marco-MiniLM-L-6-v2"  # cross-encoder re-scores fused hits for better precision.
    )
    reranker_device: str = "cpu"  # pins reranker to CPU/GPU explicitly for portable deployments.

    # Chunking
    chunk_size: int = Field(default=1000, gt=0)  # chunk length balances context coverage vs retrieval noise.
    chunk_overlap: int = Field(default=150, ge=0)  # overlap preserves sentence context across chunk boundaries.

    # Hybrid retrieval
    dense_top_k: int = Field(default=20, gt=0)  # number of vector hits to feed into RRF fusion.
    bm25_top_k: int = Field(default=20, gt=0)  # number of keyword (BM25) hits to feed into RRF fusion.
    fusion_top_k: int = Field(default=20, gt=0)  # caps fused candidate set before reranking to bound cost.
    rerank_top_k: int = Field(default=8, gt=0)  # final evidence set size passed to LLM and citation check.
    rrf_k: int = Field(default=60, gt=0)  # RRF smoothing constant controlling rank vs score influence.
    dense_weight: float = Field(default=0.6, ge=0.0, le=1.0)  # weights semantic matches higher than lexical ones.
    lexical_weight: float = Field(
        default=0.4, ge=0.0, le=1.0
    )  # keeps keyword signal so tickers/figures are not missed.

    # Context compression
    context_token_budget: int = Field(default=12000, gt=0)  # caps prompt tokens to fit model window and control cost.
    max_compressed_chunks: int = Field(default=8, gt=0)  # limits chunks kept after compression to bound latency.

    # Citation verification
    verification_max_retries: int = Field(
        default=2, ge=0, le=3
    )  # retries unverified answers before flagging as degraded.

    # Human-in-the-loop
    hitl_enabled: bool = True  # compliance gate blocking investment-like answers without review.
    hitl_interrupt_node: str = "human_review_gate"  # LangGraph node name where execution pauses for approval.

    # Tools
    enable_tools: bool = True  # toggles tool calls (retrieval/calculator) for simple or offline runs.

    # Redis
    redis_url: str = "redis://localhost:6379/0"  # connection for LangGraph checkpointing and short-term state.
    checkpointer_type: CheckpointerType = "in_memory"  # selects persistent Redis vs ephemeral memory checkpointer.

    # Storage
    postgres_database_url: str = (
        "postgresql://rag_user:rag_password@localhost:5432/rag_db"  # vector/metadata store connection (pgvector).
    )

    # SEC EDGAR
    sec_edgar_user_agent: str = "FinSightResearch dev@example.com"  # SEC-required User-Agent with contact email.
    sec_request_throttle_seconds: float = Field(
        default=0.5, gt=0
    )  # throttles EDGAR requests to respect fair-use limits.

    # SEC XBRL companyfacts
    xbrl_enabled: bool = True  # enables structured financial facts for P/E and numeric answers.
    sec_companyfacts_base_url: str = "https://data.sec.gov/api/xbrl/companyfacts"
    sec_cache_ttl_seconds: int = Field(default=86_400, gt=0)  # companyfacts change only on new filings.
    sec_max_attempts: int = Field(default=3, ge=1, le=5)
    sec_backoff_seconds: float = Field(default=0.5, ge=0.0)

    # Evaluation
    eval_dataset_path: Path = (
        PROJECT_ROOT / "data" / "eval" / "eval_dataset.json"
    )  # fixed offline eval set for RAGAS regression checks.
    ragas_metrics: list[str] = [  # quality metrics tracked on every pipeline change.
        "context_precision",
        "context_recall",
        "faithfulness",
        "answer_relevance",
    ]
    judge_model: str = "qwen/qwen3.5-9b"  # LLM judge scoring eval answers consistently across runs.
    eval_seed: int = 42  # seed for reproducible eval sampling and shuffling.

    # Validators
    @field_validator("postgres_database_url")
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        if not v.startswith(("postgresql://", "postgresql+psycopg2://")):
            raise ValueError("postgres_database_url must start with 'postgresql://'")
        return v

    @field_validator("redis_url")
    @classmethod
    def validate_redis_url(cls, v: str) -> str:
        if not v.startswith(("redis://", "rediss://")):
            raise ValueError("redis_url must start with 'redis://'")
        return v

    @field_validator("fallback_chain")
    @classmethod
    def validate_fallback_chain(cls, v: list[ProviderName]) -> list[ProviderName]:
        if not v:
            raise ValueError("fallback_chain must contain at least one provider")
        if len(v) != len(set(v)):
            raise ValueError("fallback_chain must not contain duplicate providers")
        return v

    @field_validator("sec_edgar_user_agent")
    @classmethod
    def validate_sec_user_agent(cls, v: str) -> str:
        if not _EMAIL_PATTERN.search(v):
            raise ValueError("SEC EDGAR requires User-Agent with email")
        return v

    @model_validator(mode="after")
    def validate_groq_config(self) -> "Settings":
        if self.groq_api_key and not self.groq_model.strip():
            raise ValueError("groq_api_key is set but groq_model is empty")
        return self

    @model_validator(mode="after")
    def validate_model_routing(self) -> "Settings":
        for tier in ("cheap", "medium", "expensive"):
            cfg: ModelTierConfig = getattr(self.model_routing, tier)
            if not cfg.model_name.strip():
                raise ValueError(f"model_routing.{tier}.model_name must not be empty")
        return self

    @model_validator(mode="after")
    def validate_fallback_tier_models(self) -> "Settings":
        for provider in self.fallback_chain:
            models = self.fallback_tier_models.get(provider)
            if not models:
                raise ValueError(f"fallback_tier_models missing provider {provider!r}")
            for tier in ("simple", "normal", "complex"):
                if not models.get(tier, "").strip():
                    raise ValueError(f"fallback_tier_models.{provider}.{tier} must not be empty")
        return self

    # Helpers
    def tier_model(self, tier: str) -> ModelTierConfig:
        """Resolve model config for tier."""
        tier_to_field: dict[str, str] = {
            "simple": "cheap",
            "normal": "medium",
            "complex": "expensive",
        }
        return getattr(self.model_routing, tier_to_field[tier])

    def tier_model_for_provider(self, tier: str, provider: ProviderName) -> ModelTierConfig:
        """Resolve the tier config for a specific provider in the fallback chain."""
        primary = self.tier_model(tier)
        if provider == primary.provider:
            return primary
        model_name = self.fallback_tier_models.get(provider, {}).get(tier)
        if not model_name:
            raise ValueError(f"no {tier!r} model configured for fallback provider {provider!r}")
        return ModelTierConfig(
            model_name=model_name,
            provider=provider,
            temperature=primary.temperature,
            max_tokens=primary.max_tokens,
        )

    def has_provider_credentials(self, provider: ProviderName) -> bool:
        """Check if credentials exist for provider."""
        if provider == "openai":
            return bool(self.openai_api_key)
        if provider == "groq":
            return bool(self.groq_api_key)
        if provider == "gemini":
            return bool(self.google_api_key)
        if provider == "vllm":
            return bool(self.vllm_base_url and self.vllm_model)
        return False


def get_setting() -> Settings:
    return Settings()


settings = Settings()

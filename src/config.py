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
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    parsed_docs_dir: Path = PROJECT_ROOT / "data" / "parsed"
    chunk_store_dir: Path = PROJECT_ROOT / "data" / "chunks"

    # Default model
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    # Provider credentials (env-injected)
    openai_api_key: str | None = None
    groq_api_key: str | None = None
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_model: str = "qwen/qwen3.6-27b"
    google_api_key: str | None = None
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_api_key: str | None = None
    vllm_model: str | None = None

    # Model routing
    model_routing: ModelRouting = ModelRouting()

    # Fallback chain
    fallback_chain: list[ProviderName] = ["openai", "groq", "gemini", "vllm"]

    # Embeddings
    embedding_provider: EmbeddingProvider = "sentence_transformers"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimensions: int = Field(default=384, gt=0)
    embedding_batch_size: int = Field(default=64, gt=0)

    # Reranker
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_device: str = "cpu"

    # Chunking
    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=150, ge=0)

    # Hybrid retrieval
    dense_top_k: int = Field(default=20, gt=0)
    bm25_top_k: int = Field(default=20, gt=0)
    fusion_top_k: int = Field(default=20, gt=0)
    rerank_top_k: int = Field(default=8, gt=0)
    rrf_k: int = Field(default=60, gt=0)
    dense_weight: float = Field(default=0.6, ge=0.0, le=1.0)
    lexical_weight: float = Field(default=0.4, ge=0.0, le=1.0)

    # Context compression
    context_token_budget: int = Field(default=12000, gt=0)
    max_compressed_chunks: int = Field(default=8, gt=0)

    # Citation verification
    verification_max_retries: int = Field(default=2, ge=0, le=3)

    # Human-in-the-loop
    hitl_enabled: bool = True
    hitl_interrupt_node: str = "human_review_gate"

    # Tools
    enable_tools: bool = True

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    checkpointer_type: CheckpointerType = "in_memory"

    # Storage
    postgres_database_url: str = "postgresql://rag_user:rag_password@localhost:5432/rag_db"

    # SEC EDGAR
    sec_edgar_user_agent: str = "FinSightResearch dev@example.com"
    sec_request_throttle_seconds: float = Field(default=0.5, gt=0)

    # Evaluation
    eval_dataset_path: Path = PROJECT_ROOT / "data" / "eval" / "eval_dataset.json"
    ragas_metrics: list[str] = [
        "context_precision",
        "context_recall",
        "faithfulness",
        "answer_relevance",
    ]
    judge_model: str = "qwen/qwen3.6-27b"
    eval_seed: int = 42

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

    # Helpers
    def tier_model(self, tier: ComplexityTier) -> ModelTierConfig:
        """Resolve model config for tier."""
        tier_to_field: dict[ComplexityTier, str] = {
            "simple": "cheap",
            "normal": "medium",
            "complex": "expensive",
        }
        return getattr(self.model_routing, tier_to_field[tier])

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

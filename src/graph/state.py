"""GraphState for FinSight — carries pipeline state through LangGraph."""

from __future__ import annotations

from typing import Any, TypedDict

from src.config import ComplexityTier


class GraphState(TypedDict, total=False):
    # Core query / retrieval
    query: str
    k: int
    ticker: str | None
    tickers: list[str]
    fiscal_year: int | None
    form_type: str | None
    intent: str | None
    sub_queries: list[str]
    thread_id: str | None
    tier: ComplexityTier
    chunks: list[dict[str, Any]]
    injected_context: str | None  # eval-only fault injection; never exposed by the HTTP API
    # LLM answer
    answer: str
    draft_answer: str
    provider: str
    model: str | None
    tokens_in: int
    tokens_out: int
    sources: list[str]
    error: str | None
    degraded_reason: str | None
    tool_results: list[dict[str, Any]]
    verification: dict[str, Any]  # VerificationResult.to_dict()
    verification_retries: int
    abstention_reason: str | None
    is_recommendation: bool
    hitl_approved: bool | None
    hitl_feedback: str | None

"""GraphState for FinSight — carries pipeline state through LangGraph."""

from __future__ import annotations

from typing import Any, TypedDict

from src.config import ComplexityTier


class GraphState(TypedDict, total=False):
    # Core query / retrieval
    query: str
    k: int
    tier: ComplexityTier
    chunks: list[dict[str, Any]]
    # LLM answer
    answer: str
    draft_answer: str
    provider: str
    tokens_in: int
    sources: list[str]
    error: str | None
    tool_results: list[dict[str, Any]]
    verification: dict[str, Any]  # VerificationResult.to_dict()
    verification_retries: int
    is_recommendation: bool
    hitl_approved: bool | None
    hitl_feedback: str | None

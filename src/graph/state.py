"""Phase 5-9 — LangGraph state (PRD Phase 5 ``GraphState`` + Phase 9 routing).

Carries the query through the linear pipeline: complexity tier (Phase 9),
retrieved chunks, compressed context, LLM answer, serving provider/token
counts, and the source summary for the final response.
"""

from __future__ import annotations

from typing import Any, TypedDict

from src.config import ComplexityTier


class GraphState(TypedDict, total=False):
    query: str
    k: int
    tier: ComplexityTier
    chunks: list[dict[str, Any]]
    answer: str
    provider: str
    tokens_in: int
    sources: list[str]
    error: str | None
"""Phase 4-10 — LLM layer: single call (4), LangGraph wiring (5), routing (9),
fallback chain (10).

Phase 4: one hardcoded LLM call that takes retrieved chunks + query -> answer.
"""

from src.llm.answer import answer_query, build_answer_prompt, format_context
from src.llm.client import (
    ProviderUnavailableError,
    build_llm,
    build_tier_llm,
)
from src.llm.fallback import (
    DegradedResponseError,
    LLMResult,
    MalformedResponseError,
    call_with_fallback,
)

__all__ = [
    "DegradedResponseError",
    "LLMResult",
    "MalformedResponseError",
    "ProviderUnavailableError",
    "answer_query",
    "build_answer_prompt",
    "build_llm",
    "build_tier_llm",
    "call_with_fallback",
    "format_context",
]

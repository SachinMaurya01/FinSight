"""Human-in-the-loop gate — recommendation classifier + interrupt logic."""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Heuristic patterns that indicate recommendation / thesis framing.
_RECOMMENDATION_PATTERNS = (
    r"\bshould\s+(you\s+)?(buy|sell|hold|invest)\b",
    r"\brecommend(s|ed|ation)?\b",
    r"\bbull\s+case\b",
    r"\bbear\s+case\b",
    r"\binvestment\s+thesis\b",
    r"\bstrong\s+(buy|sell)\b",
    r"\boverweight\b",
    r"\bunderweight\b",
    r"\boutperform\b",
    r"\bunderperform\b",
    r"\bbuy\s+or\s+sell\b",
    r"\bbuild\s+a\s+(bull|bear)\s+case\b",
    r"\bprice\s+target\b",
    r"\bupside\b.*\bdownside\b",
)

_RECOMMENDATION_RE = re.compile("|".join(_RECOMMENDATION_PATTERNS), re.IGNORECASE)


def is_recommendation(answer: str, query: str | None = None) -> bool:
    """Classify whether output is investment-recommendation-like.

    Heuristic: matches recommendation patterns in answer or query.
    Future upgrade: LLM classifier.

    Args:
        answer: Draft answer text.
        query: Original query (optional, provides context like "build a bull case").
    Returns:
        True if HITL review should be triggered.
    """
    text = f"{answer} {query or ''}"
    match = _RECOMMENDATION_RE.search(text)
    if match:
        logger.info("HITL trigger: matched pattern %r in %r", match.group(0), text[:120])
        return True
    return False


def hitl_payload(answer: str, query: str, sources: list[str] | None = None) -> dict[str, Any]:
    """Build payload for human review interrupt."""
    return {
        "type": "investment_recommendation_review",
        "query": query,
        "answer": answer,
        "sources": sources or [],
        "instruction": "Approve or reject this investment-recommendation-like output.",
    }

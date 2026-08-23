"""Query complexity classifier."""

from __future__ import annotations

from src.config import ComplexityTier

# Queries that compare/synthesize across documents or frame an investment view.
_COMPLEX_KEYWORDS = (
    "compare",
    "comparison",
    "contrast",
    "versus",
    " vs ",
    "vs.",
    "better than",
    "worse than",
    "outperform",
    "underperform",
    "bull",
    "bear",
    "thesis",
    "risk profile",
    "liquidity risk of",
    "investment recommendation",
    "should i invest",
    "buy or sell",
    "build a case",
)

# Single-fact lookups: one metric, one figure, one definition.
_SIMPLE_KEYWORDS = (
    "what was",
    "what were",
    "what is the",
    "how much",
    "how many",
    "when did",
    "when was",
    "revenue of",
    "net income",
    "net sales",
    "gross margin",
    "operating margin",
    "earnings per share",
    "eps",
    "pe ratio",
    "dividend",
    "free cash flow",
    "total cash",
)


def classify_complexity(query: str) -> ComplexityTier:
    """Classify ``query`` into a complexity tier using keyword heuristics."""
    normalized = query.lower().strip()
    if any(keyword in normalized for keyword in _COMPLEX_KEYWORDS):
        return "complex"
    if any(keyword in normalized for keyword in _SIMPLE_KEYWORDS):
        return "simple"
    return "normal"
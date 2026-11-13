"""Human-in-the-loop gate — recommendation classifier + interrupt logic."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from src.config import Settings
from src.llm.fallback import call_with_fallback

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

# Matches investment-recommendation phrasing (buy/sell/thesis/price target) to trigger HITL review.
_RECOMMENDATION_RE = re.compile("|".join(_RECOMMENDATION_PATTERNS), re.IGNORECASE)


_CLASSIFIER_PROMPT = """Classify whether the text below is an investment recommendation.

Return ONLY compact JSON: {{"is_recommendation": true|false, "confidence": 0.0-1.0, "reason": "short"}}.

Investment recommendations include buy/sell/hold calls, portfolio allocation advice,
price targets, overweight/underweight ratings, and bull/bear cases framed as advice.
General risk summaries, factual filing analysis, and statements that no recommendation
can be made are NOT recommendations.

QUERY:
{query}

ANSWER:
{answer}
"""


def _heuristic_is_recommendation(answer: str, query: str | None = None) -> bool:
    text = f"{answer} {query or ''}"
    match = _RECOMMENDATION_RE.search(text)
    if match:
        logger.info("HITL trigger: matched pattern %r", match.group(0))
        return True
    return False


def is_recommendation(answer: str, query: str | None = None) -> bool:
    """Classify whether output is investment-recommendation-like using the heuristic."""
    return _heuristic_is_recommendation(answer, query)


def _parse_classifier_response(content: str) -> bool | None:
    try:
        payload = json.loads(content.strip())
    except (json.JSONDecodeError, ValueError):
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match is None:
            return None
        try:
            payload = json.loads(match.group(0))
        except (json.JSONDecodeError, ValueError):
            return None
    if not isinstance(payload, dict) or "is_recommendation" not in payload:
        return None
    value = payload["is_recommendation"]
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return None


def classify_recommendation(
    answer: str,
    query: str,
    settings: Settings,
    llm: Any | None = None,
) -> bool:
    """Classify recommendation-like output with optional cheap-tier LLM review.

    The regex heuristic always runs. When enabled, the LLM can catch
    recommendation framing the regex misses; any classifier failure falls back
    to the heuristic rather than blocking the pipeline.
    """
    heuristic = _heuristic_is_recommendation(answer, query)
    if heuristic or not settings.recommendation_classifier_enabled:
        return heuristic

    prompt = _CLASSIFIER_PROMPT.format(query=query[:1000], answer=answer[:6000])
    try:
        if llm is not None:
            response = llm.invoke(prompt)
            content = str(getattr(response, "content", response))
        else:
            content = call_with_fallback(prompt, settings, tier="simple").content
        parsed = _parse_classifier_response(content)
    except Exception as exc:
        logger.warning("Recommendation classifier failed; using heuristic: %s", type(exc).__name__)
        return heuristic
    if parsed is None:
        logger.warning("Recommendation classifier returned unparseable output; using heuristic")
        return heuristic
    logger.info("Recommendation classifier: is_recommendation=%s", parsed)
    return parsed


def hitl_payload(answer: str, query: str, sources: list[str] | None = None) -> dict[str, Any]:
    """Build payload for human review interrupt."""
    return {
        "type": "investment_recommendation_review",
        "query": query,
        "answer": answer,
        "sources": sources or [],
        "instruction": "Approve or reject this investment-recommendation-like output.",
    }

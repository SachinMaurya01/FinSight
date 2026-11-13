"""Abstention: decide when FinSight must withhold an answer instead of guessing."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

ABSTAIN_MESSAGE = "I can't answer this from the available filings"

ABSTENTION_REASONS: dict[str, str] = {
    "no_evidence": "no matching filing passages were retrieved",
    "ticker_not_in_corpus": "the requested ticker is not in the indexed filings",
    "year_not_in_corpus": "the requested fiscal year is not in the indexed filings",
    "low_retrieval_confidence": "the retrieved passages are not relevant enough to the question",
    "verification_failed": "the answer could not be verified against the cited filings",
}


def abstention_reason(
    chunks: Sequence[dict[str, Any]],
    *,
    tickers: Sequence[str],
    fiscal_year: int | None,
    min_rerank_score: float | None,
    verification_passed: bool,
) -> str | None:
    """Return why the answer must be withheld, or ``None`` when it may be shown.

    Metadata checks only apply when the rows carry that metadata, so partial rows
    never trigger abstention on their own.
    """
    if not chunks:
        return "no_evidence"
    if tickers:
        available = {str(chunk.get("ticker") or "").upper() for chunk in chunks}
        if {ticker.upper() for ticker in tickers} - available:
            return "ticker_not_in_corpus"
    if fiscal_year is not None:
        years = {chunk.get("fiscal_year") for chunk in chunks if chunk.get("fiscal_year") is not None}
        if years and fiscal_year not in years:
            return "year_not_in_corpus"
    if min_rerank_score is not None:
        scores = [float(chunk["rerank_score"]) for chunk in chunks if "rerank_score" in chunk]
        if scores and max(scores) < min_rerank_score:
            return "low_retrieval_confidence"
    if not verification_passed:
        return "verification_failed"
    return None


def abstention_caveat(reason: str) -> str:
    """Human-readable caveat returned alongside an abstained response."""
    detail = ABSTENTION_REASONS.get(reason, reason)
    return f"Answer withheld: {detail}."

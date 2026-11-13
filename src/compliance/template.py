"""Fixed compliance text applied to recommendation-like output."""

from __future__ import annotations

NOT_INVESTMENT_ADVICE = (
    "Not investment advice. This is a research summary generated from SEC filings; "
    "it is not a recommendation to buy, sell, or hold any security. Consult a licensed "
    "financial professional before making investment decisions."
)


def apply_investment_disclaimer(answer: str) -> str:
    """Append the fixed disclaimer once to an investment-framed answer."""
    if NOT_INVESTMENT_ADVICE in answer:
        return answer
    body = answer.strip()
    return f"{body}\n\n{NOT_INVESTMENT_ADVICE}" if body else NOT_INVESTMENT_ADVICE

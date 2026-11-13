"""Query understanding with an LLM parser and deterministic regex fallback.

The regex path is intentionally deterministic so offline tests and CLI runs do
not need an API key. When ``settings.query_understanding_enabled`` is true, the
graph tries a cheap LLM with a Pydantic structured-output contract and falls back
to regex on any error.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from src.config import Settings

logger = logging.getLogger(__name__)

Intent = Literal["lookup", "ratio", "comparison", "recommendation"]

_FORM_TYPES = ("10-K", "10-Q", "8-K")
_FORM_RE = re.compile(r"\b(10-[KQ]|8-K)\b", re.IGNORECASE)
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_FY_RE = re.compile(r"\bFY\s?((?:19|20)\d{2})\b", re.IGNORECASE)
_TICKER_RE = re.compile(r"\b[A-Z]{2,5}\b")

_KNOWN_TICKERS = {
    "AAPL",
    "MSFT",
    "GOOGL",
    "GOOG",
    "AMZN",
    "META",
    "NVDA",
    "TSLA",
}
_COMPANY_TICKERS = {
    "apple": "AAPL",
    "microsoft": "MSFT",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "amazon": "AMZN",
    "meta": "META",
    "facebook": "META",
    "nvidia": "NVDA",
    "tesla": "TSLA",
}
_TICKER_STOPWORDS = {
    "A",
    "AN",
    "AND",
    "ARE",
    "AS",
    "AT",
    "BE",
    "BY",
    "DO",
    "FOR",
    "FROM",
    "HOW",
    "IN",
    "IS",
    "IT",
    "OF",
    "ON",
    "OR",
    "THE",
    "TO",
    "VS",
    "WHAT",
    "WHY",
    "WITH",
    "GAAP",
    "MD&A",
    "CEO",
    "CFO",
    "FY",
    "PE",
}
_RECOMMENDATION_RE = re.compile(
    r"\b(should\s+i|buy|sell|invest|overweight|underweight|portfolio|bull|bear)\b",
    re.IGNORECASE,
)
_RATIO_RE = re.compile(
    r"\b(p/e|pe ratio|price[- ]to[- ]earnings|ratio|margin|growth rate|gross margin|net margin)\b",
    re.IGNORECASE,
)
_COMPARISON_RE = re.compile(
    r"\b(compare|comparison|versus|vs\.?|trend|changed?|evolv\w*|differ\w*|between)\b",
    re.IGNORECASE,
)


class QueryIntent(BaseModel):
    """Structured query understanding output."""

    tickers: list[str] = Field(default_factory=list)
    fiscal_year: int | None = None
    form_type: str | None = None
    intent: Intent = "lookup"
    sub_queries: list[str] = Field(default_factory=list)


def _extract_tickers(query: str) -> list[str]:
    tickers: list[str] = []
    for match in _TICKER_RE.finditer(query):
        candidate = match.group(0).upper()
        if candidate in _KNOWN_TICKERS and candidate not in tickers:
            tickers.append(candidate)
    lower = query.lower()
    for name, ticker in _COMPANY_TICKERS.items():
        if re.search(rf"\b{re.escape(name)}\b", lower) and ticker not in tickers:
            tickers.append(ticker)
    # Explicit uppercase short names not in the known map are useful for new tickers.
    for match in _TICKER_RE.finditer(query):
        candidate = match.group(0).upper()
        if candidate not in _TICKER_STOPWORDS and candidate not in tickers and len(candidate) <= 5:
            tickers.append(candidate)
    return tickers


def _extract_years(query: str) -> list[int]:
    years = {int(match.group(1)) for match in _FY_RE.finditer(query)}
    years.update(int(match.group(0)) for match in _YEAR_RE.finditer(query))
    return sorted(years)


def _extract_intent(query: str) -> Intent:
    if _RECOMMENDATION_RE.search(query):
        return "recommendation"
    if _COMPARISON_RE.search(query):
        return "comparison"
    if _RATIO_RE.search(query):
        return "ratio"
    return "lookup"


def parse_query_regex(query: str) -> QueryIntent:
    """Extract tickers, year, form, intent, and comparison sub-queries."""
    tickers = _extract_tickers(query)
    years = _extract_years(query)
    form_match = _FORM_RE.search(query)
    intent = _extract_intent(query)
    sub_queries: list[str] = []
    if intent == "comparison" and len(tickers) > 1:
        for ticker in tickers:
            stripped = query
            for other in tickers:
                if other == ticker:
                    continue
                stripped = re.sub(rf"\b{re.escape(other)}\b", "", stripped, flags=re.IGNORECASE)
            for name, mapped in _COMPANY_TICKERS.items():
                if mapped != ticker:
                    stripped = re.sub(rf"\b{re.escape(name)}\b", "", stripped, flags=re.IGNORECASE)
            sub_queries.append(f"{ticker}: {' '.join(stripped.split())}")
    return QueryIntent(
        tickers=tickers,
        fiscal_year=max(years) if years else None,
        form_type=form_match.group(1).upper() if form_match else None,
        intent=intent,
        sub_queries=sub_queries,
    )


_LLM_PROMPT = """Extract retrieval filters from the user query.
Return ONLY JSON matching this schema:
{{"tickers": ["AAPL"], "fiscal_year": 2025, "form_type": "10-K",
  "intent": "lookup|ratio|comparison|recommendation", "sub_queries": []}}
Rules:
- tickers: uppercase SEC tickers mentioned or implied by company names.
- fiscal_year: four-digit fiscal year if the query names one; otherwise null.
- form_type: 10-K, 10-Q, or 8-K if the query names one; otherwise null.
- intent: recommendation for buy/sell/invest questions, ratio for valuation/ratio
  or margin questions, comparison for compare/trend questions, else lookup.
- sub_queries: for comparisons across multiple tickers, one query per ticker.

Query: {query}
"""


def _parse_llm_json(content: str) -> QueryIntent | None:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    try:
        return QueryIntent.model_validate(payload)
    except Exception:
        return None


def parse_query(query: str, settings: Settings, llm: Any | None = None) -> QueryIntent:
    """Parse a query, using an LLM only when explicitly enabled or injected."""
    fallback = parse_query_regex(query)
    if llm is None and not settings.query_understanding_enabled:
        return fallback
    try:
        if llm is not None:
            response = llm.invoke(_LLM_PROMPT.format(query=query))
            content = str(getattr(response, "content", response))
        else:
            from src.llm.fallback import call_with_fallback

            content = call_with_fallback(_LLM_PROMPT.format(query=query), settings, tier="simple").content
        parsed = _parse_llm_json(content)
        if parsed is None:
            return fallback
        parsed.tickers = list(dict.fromkeys(ticker.upper() for ticker in parsed.tickers if ticker.strip()))
        if parsed.form_type:
            parsed.form_type = parsed.form_type.upper()
        # Preserve deterministic regex fields when the LLM leaves them empty.
        if not parsed.tickers:
            parsed.tickers = fallback.tickers
        if parsed.fiscal_year is None:
            parsed.fiscal_year = fallback.fiscal_year
        if parsed.form_type is None:
            parsed.form_type = fallback.form_type
        if not parsed.sub_queries:
            parsed.sub_queries = fallback.sub_queries
        return parsed
    except Exception as exc:
        logger.warning("Query LLM parsing failed; using regex fallback: %s", exc)
        return fallback


def filters_for(intent: QueryIntent, *, ticker: str | None = None) -> dict[str, Any]:
    """Build retrieval filters from parsed intent.

    Comparison queries deliberately do not filter on a single fiscal year, since
    the answer often spans multiple periods.
    """
    filters: dict[str, Any] = {}
    selected_ticker = ticker or (intent.tickers[0] if len(intent.tickers) == 1 else None)
    if selected_ticker:
        filters["ticker"] = selected_ticker.lower()
    if intent.fiscal_year is not None and intent.intent != "comparison":
        filters["fiscal_year"] = intent.fiscal_year
    if intent.form_type:
        filters["form_type"] = intent.form_type
    return filters

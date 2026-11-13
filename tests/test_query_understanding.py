"""Query understanding tests (regex fallback + injected LLM, no network)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Settings
from src.query.understanding import filters_for, parse_query, parse_query_regex

SETTINGS = Settings(postgres_database_url="postgresql://localhost:5432/finsight_test")


def test_regex_entities_period_and_intent() -> None:
    parsed = parse_query_regex("What were Apple's total net sales in fiscal 2023?")
    assert parsed.tickers == ["AAPL"]
    assert parsed.fiscal_year == 2023
    assert parsed.form_type is None
    assert parsed.intent == "lookup"
    print("PASS test_regex_entities_period_and_intent")


def test_regex_comparison_decomposes_per_ticker() -> None:
    parsed = parse_query_regex("Compare AAPL and MSFT gross margin trends")
    assert parsed.tickers == ["AAPL", "MSFT"]
    assert parsed.intent == "comparison"
    assert len(parsed.sub_queries) == 2
    assert filters_for(parsed) == {}
    assert filters_for(parsed, ticker="AAPL") == {"ticker": "aapl"}
    print("PASS test_regex_comparison_decomposes_per_ticker")


def test_regex_ratio_and_recommendation() -> None:
    ratio = parse_query_regex("What's the P/E ratio for AAPL?")
    rec = parse_query_regex("Should I buy AAPL stock?")
    form = parse_query_regex("Summarize the 10-K risk factors for Apple")
    assert ratio.intent == "ratio"
    assert rec.intent == "recommendation"
    assert form.form_type == "10-K"
    print("PASS test_regex_ratio_and_recommendation")


def test_filters_for_lookup_uses_year() -> None:
    parsed = parse_query_regex("What were Apple's net sales in 2023?")
    filters = filters_for(parsed)
    assert filters == {"ticker": "aapl", "fiscal_year": 2023}
    print("PASS test_filters_for_lookup_uses_year")


def test_injected_llm_overrides_regex() -> None:
    class FakeLLM:
        def invoke(self, prompt: str) -> SimpleNamespace:
            return SimpleNamespace(
                content='{"tickers": ["MSFT"], "fiscal_year": 2024, '
                '"form_type": "10-Q", "intent": "ratio", "sub_queries": []}'
            )

    parsed = parse_query("What is Apple's P/E?", SETTINGS, llm=FakeLLM())
    assert parsed.tickers == ["MSFT"]
    assert parsed.fiscal_year == 2024
    assert parsed.form_type == "10-Q"
    assert parsed.intent == "ratio"
    print("PASS test_injected_llm_overrides_regex")


def test_invalid_llm_falls_back() -> None:
    class BadLLM:
        def invoke(self, prompt: str) -> SimpleNamespace:
            return SimpleNamespace(content="not json")

    parsed = parse_query("What were Apple's net sales in 2023?", SETTINGS, llm=BadLLM())
    assert parsed.tickers == ["AAPL"] and parsed.fiscal_year == 2023
    print("PASS test_invalid_llm_falls_back")


def main() -> int:
    test_regex_entities_period_and_intent()
    test_regex_comparison_decomposes_per_ticker()
    test_regex_ratio_and_recommendation()
    test_filters_for_lookup_uses_year()
    test_injected_llm_overrides_regex()
    test_invalid_llm_falls_back()
    return 0


if __name__ == "__main__":
    sys.exit(main())

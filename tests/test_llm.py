"""Phase 4 LLM call unit tests.

``answer_query`` is tested with a fake LLM (no network / no API key). A live
smoke test against Groq is a TODO deferred until credentials are confirmed
(AGENTS.md §4.4).

Run with: python tests/test_llm.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Settings
from src.ingestion.chunker import Chunk
from src.llm.answer import answer_query, format_context
from src.llm.client import ProviderUnavailableError, build_llm
from src.llm.output import strip_think_block

NO_KEY_SETTINGS = Settings(groq_api_key=None)


def _chunks() -> list[Chunk]:
    return [
        Chunk(
            content="Apple faces supply chain risk in China.",
            metadata={
                "ticker": "aapl",
                "filing_type": "10-K",
                "fiscal_period": "FY2025",
                "section": "Item 1A Risk Factors",
            },
        ),
        Chunk(
            content="Foreign exchange volatility can impact margins.",
            metadata={
                "ticker": "aapl",
                "filing_type": "10-K",
                "fiscal_period": "FY2025",
                "section": "Item 1A Risk Factors",
            },
        ),
    ]


class FakeLLM:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> SimpleNamespace:
        self.prompts.append(prompt)
        return SimpleNamespace(content="The main risks are supply chain and FX [1][2]")


def test_format_context_numbered_and_tagged() -> None:
    text = format_context(_chunks())
    assert "[1] (source: aapl, 10-K, FY2025, Item 1A Risk Factors)" in text
    assert "[2]" in text
    assert "supply chain risk" in text
    print("PASS test_format_context_numbered_and_tagged")


def test_strip_think_block() -> None:
    dirty = "<thinking>Reasoning here.</thinking>\nBased on context, the answer is X [1]."
    assert strip_think_block(dirty) == "Based on context, the answer is X [1]."
    assert strip_think_block("no think block") == "no think block"
    print("PASS test_strip_think_block")


def test_answer_query_injects_context_and_question() -> None:
    llm = FakeLLM()
    answer = answer_query("what are the main risks", _chunks(), NO_KEY_SETTINGS, llm=llm)
    assert answer == "The main risks are supply chain and FX [1][2]"
    prompt = llm.prompts[0]
    assert "what are the main risks" in prompt
    assert "Apple faces supply chain risk in China." in prompt
    assert "=== CONTEXT ===" in prompt
    print("PASS test_answer_query_injects_context_and_question")


def test_build_llm_requires_groq_key() -> None:
    try:
        build_llm(NO_KEY_SETTINGS, provider="groq")
    except ProviderUnavailableError as exc:
        assert "groq_api_key" in str(exc)
    else:
        raise AssertionError("expected ProviderUnavailableError for missing groq_api_key")
    print("PASS test_build_llm_requires_groq_key")


def test_build_llm_rejects_unsupported_provider() -> None:
    try:
        build_llm(NO_KEY_SETTINGS, provider="bogus")  # type: ignore[arg-type]
    except ValueError as exc:
        assert "Unsupported provider" in str(exc)
    else:
        raise AssertionError("expected ValueError for unsupported provider")
    print("PASS test_build_llm_rejects_unsupported_provider")


def main() -> int:
    test_format_context_numbered_and_tagged()
    test_answer_query_injects_context_and_question()
    test_build_llm_requires_groq_key()
    test_build_llm_rejects_unsupported_provider()
    test_strip_think_block()
    return 0


if __name__ == "__main__":
    sys.exit(main())

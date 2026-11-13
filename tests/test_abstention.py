"""Abstention tests: unanswerable, out-of-corpus, and unverifiable answers are withheld."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.graph import nodes as graph_nodes
from src.verification.abstention import ABSTAIN_MESSAGE, abstention_caveat, abstention_reason

AAPL_CHUNK = {
    "content": "Synthetic filing: net sales were $383.3 billion.",
    "ticker": "aapl",
    "filing_type": "10-K",
    "fiscal_period": "FY2023",
    "fiscal_year": 2023,
    "section": "Item 8",
    "rerank_score": 4.0,
}
BAD_ANSWER = "Net sales were $999 billion in fiscal 2023."


def _reason(chunks=None, tickers=(), fiscal_year=None, min_score=-5.0, passed=True) -> str | None:
    return abstention_reason(
        [AAPL_CHUNK] if chunks is None else chunks,
        tickers=list(tickers),
        fiscal_year=fiscal_year,
        min_rerank_score=min_score,
        verification_passed=passed,
    )


def test_reasons() -> None:
    assert _reason() is None
    assert _reason(chunks=[]) == "no_evidence"
    assert _reason(tickers=["MSFT"]) == "ticker_not_in_corpus"
    assert _reason(tickers=["AAPL"]) is None
    assert _reason(fiscal_year=2019) == "year_not_in_corpus"
    assert _reason(fiscal_year=2023) is None
    assert _reason(min_score=5.0) == "low_retrieval_confidence"
    assert _reason(min_score=None) is None
    assert _reason(passed=False) == "verification_failed"
    print("PASS test_reasons")


def test_partial_metadata_does_not_abstain() -> None:
    bare = [{"content": "Apple faces supply chain risk."}]
    assert _reason(chunks=bare, fiscal_year=2023, min_score=None) is None
    print("PASS test_partial_metadata_does_not_abstain")


def test_comparison_requires_every_ticker() -> None:
    assert _reason(tickers=["AAPL", "MSFT"]) == "ticker_not_in_corpus"
    print("PASS test_comparison_requires_every_ticker")


def _verify_node(chunks: list[dict], answer: str, retries: int, tickers=None) -> dict:
    state = {
        "answer": answer,
        "draft_answer": answer,
        "chunks": chunks,
        "verification_retries": retries,
        "tickers": tickers or [],
        "fiscal_year": None,
    }
    return graph_nodes.verify_citations_node(state)


def test_failed_verification_retries_before_abstaining() -> None:
    update = _verify_node([AAPL_CHUNK], BAD_ANSWER, retries=0)
    assert update["verification"]["passed"] is False
    assert update["verification_retries"] == 1
    assert "abstention_reason" not in update
    assert "answer" not in update
    print("PASS test_failed_verification_retries_before_abstaining")


def test_exhausted_retries_abstain_with_caveat() -> None:
    update = _verify_node([AAPL_CHUNK], BAD_ANSWER, retries=settings.verification_max_retries - 1)
    assert update["answer"] == ABSTAIN_MESSAGE
    assert update["abstention_reason"] == "verification_failed"
    assert update["verification"]["passed"] is False
    assert abstention_caveat("verification_failed").startswith("Answer withheld")
    print("PASS test_exhausted_retries_abstain_with_caveat")


def test_out_of_corpus_ticker_abstains_even_when_verified() -> None:
    answer = "Net sales were $383.3 billion."
    update = _verify_node([AAPL_CHUNK], answer, retries=0, tickers=["MSFT"])
    assert update["verification"]["passed"] is True
    assert update["answer"] == ABSTAIN_MESSAGE
    assert update["abstention_reason"] == "ticker_not_in_corpus"
    print("PASS test_out_of_corpus_ticker_abstains_even_when_verified")


def test_grounded_answer_is_not_abstained() -> None:
    answer = "Net sales were $383.3 billion."
    update = _verify_node([AAPL_CHUNK], answer, retries=0, tickers=["AAPL"])
    assert update["verification"]["passed"] is True
    assert "answer" not in update
    assert "abstention_reason" not in update
    print("PASS test_grounded_answer_is_not_abstained")


def test_finalize_does_not_flag_abstained_answer() -> None:
    state = {
        "answer": ABSTAIN_MESSAGE,
        "verification": {"passed": False, "failed_claims": [{"claim": BAD_ANSWER}]},
        "abstention_reason": "verification_failed",
        "chunks": [AAPL_CHUNK],
    }
    out = graph_nodes.finalize_node(state)
    assert out["answer"] == ABSTAIN_MESSAGE
    print("PASS test_finalize_does_not_flag_abstained_answer")


def main() -> int:
    test_reasons()
    test_partial_metadata_does_not_abstain()
    test_comparison_requires_every_ticker()
    test_failed_verification_retries_before_abstaining()
    test_exhausted_retries_abstain_with_caveat()
    test_out_of_corpus_ticker_abstains_even_when_verified()
    test_grounded_answer_is_not_abstained()
    test_finalize_does_not_flag_abstained_answer()
    return 0


if __name__ == "__main__":
    sys.exit(main())

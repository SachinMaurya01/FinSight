"""Retrieval ablation scoring tests (synthetic rows, no DB/model).

Run with: python tests/test_eval_ablation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval.ablation import score_retrieval
from src.eval.dataset import EvalExample, ExpectedCitation
from src.eval.metrics import EvalMetrics, _extract_ragas_scores


def _example(*citations: ExpectedCitation) -> EvalExample:
    return EvalExample(
        id="test_001",
        question="What were net sales?",
        expected_answer="$383.3 billion",
        expected_citations=list(citations),
    )


def test_score_retrieval_precision_recall_mrr() -> None:
    example = _example(ExpectedCitation(ticker="aapl", fiscal_period="FY2023", section="Item 7 MD&A"))
    rows = [
        {"ticker": "aapl", "fiscal_period": "FY2025", "section": "Item 1A Risk Factors"},
        {"ticker": "aapl", "fiscal_period": "FY2023", "section": "Item 7 MD&A"},
        {"ticker": "aapl", "fiscal_period": "FY2023", "section": "Item 7 MD&A"},
        {"ticker": "aapl", "fiscal_period": "FY2023", "section": "Item 1A Risk Factors"},
    ]
    result = score_retrieval(rows, example)
    assert result is not None
    assert result.context_precision == 0.5
    assert result.context_recall == 1.0
    assert result.mrr == 0.5
    print("PASS test_score_retrieval_precision_recall_mrr")


def test_score_retrieval_returns_none_without_citations() -> None:
    example = _example()
    assert score_retrieval([{"ticker": "aapl"}], example) is None
    print("PASS test_score_retrieval_returns_none_without_citations")


def test_extract_ragas_scores_maps_answer_relevancy() -> None:
    metrics = _extract_ragas_scores(
        {
            "context_precision": 0.8,
            "context_recall": 0.9,
            "faithfulness": 0.7,
            "answer_relevancy": 0.6,
        }
    )
    assert metrics == EvalMetrics(
        context_precision=0.8,
        context_recall=0.9,
        faithfulness=0.7,
        answer_relevance=0.6,
    )
    print("PASS test_extract_ragas_scores_maps_answer_relevancy")


def main() -> int:
    test_score_retrieval_precision_recall_mrr()
    test_score_retrieval_returns_none_without_citations()
    test_extract_ragas_scores_maps_answer_relevancy()
    return 0


if __name__ == "__main__":
    sys.exit(main())

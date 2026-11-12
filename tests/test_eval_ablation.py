"""Retrieval ablation scoring tests (synthetic rows, no DB/model).

Run with: python tests/test_eval_ablation.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Settings
from src.eval import ablation
from src.eval.ablation import format_markdown_table, score_retrieval
from src.eval.dataset import EvalExample, ExpectedCitation
from src.eval.metrics import EvalMetrics, _extract_ragas_scores
from src.ingestion.chunker import Chunk


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


class _FakeEmbedder:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] if "apple" in text.lower() else [0.0, 1.0] for text in texts]


class _FakeBM25:
    def search(self, _query: str, _k: int) -> list[dict[str, object]]:
        return [
            {
                "content": "Apple revenue was $1 billion",
                "ticker": "aapl",
                "fiscal_period": "FY2023",
                "section": "Item 7 MD&A",
                "source_doc": "aapl-10K.html",
                "chunk_index": 0,
            }
        ]


def test_run_ablation_with_fakes() -> None:
    example = EvalExample(
        id="test_001",
        question="What was Apple revenue?",
        expected_answer="$1 billion",
        expected_citations=[ExpectedCitation(ticker="aapl", fiscal_period="FY2023", section="Item 7 MD&A")],
    )
    originals = (
        ablation.load_dataset,
        ablation.load_chunk_files,
        ablation.build_embedder,
        ablation.build_bm25_index,
        ablation.build_reranker,
    )
    ablation.load_dataset = lambda: [example]
    ablation.load_chunk_files = lambda _path: [
        Chunk(
            content="Apple revenue was $1 billion",
            metadata={
                "source_file": "aapl-10K.html",
                "ticker": "aapl",
                "fiscal_period": "FY2023",
                "section": "Item 7 MD&A",
                "chunk_index": 0,
            },
        )
    ]
    ablation.build_embedder = lambda _settings: _FakeEmbedder()
    ablation.build_bm25_index = lambda _path: _FakeBM25()
    ablation.build_reranker = lambda _settings: None
    try:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "ablation.json"
            report = ablation.run_retrieval_ablation(limit=1, output_path=output, config=Settings())
            assert output.exists()
            assert report["n_examples"] == 1
            assert report["configs"]["hybrid_rerank"]["n_scored"] == 1
            table = format_markdown_table(report)
            assert "Hybrid + cross-encoder rerank" in table
    finally:
        (
            ablation.load_dataset,
            ablation.load_chunk_files,
            ablation.build_embedder,
            ablation.build_bm25_index,
            ablation.build_reranker,
        ) = originals
    print("PASS test_run_ablation_with_fakes")


def main() -> int:
    test_score_retrieval_precision_recall_mrr()
    test_score_retrieval_returns_none_without_citations()
    test_extract_ragas_scores_maps_answer_relevancy()
    test_run_ablation_with_fakes()
    return 0


if __name__ == "__main__":
    sys.exit(main())

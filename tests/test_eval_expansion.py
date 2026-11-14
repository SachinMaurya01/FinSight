"""Eval expansion tests: gold chunks, retrieval, and numeric metrics."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Settings
from src.eval.dataset import load_dataset
from src.eval.harness import _aggregate_group, _dataset_summary, _percentile
from src.eval.judge import _judge_route
from src.eval.metrics import compute_retrieval_metrics, numeric_exact_match


def test_dataset_expanded_balanced_and_labeled() -> None:
    examples = load_dataset()
    tiers = {
        tier: sum(1 for example in examples if example.complexity == tier) for tier in ("simple", "normal", "complex")
    }
    categories = {example.category for example in examples}
    assert len(examples) >= 100
    assert all(count >= 30 for count in tiers.values()), tiers
    assert sum(1 for example in examples if example.gold_chunk_ids) >= 100
    assert {"wrong_year", "unknown_ticker", "out_of_scope", "prompt_injection"} <= categories
    assert any(example.injected_context and example.forbidden_phrases for example in examples)
    assert _dataset_summary(examples)["negatives"]["prompt_injection"] >= 2
    print("PASS test_dataset_expanded_balanced_and_labeled")


def test_retrieval_metrics_recall_mrr_ndcg() -> None:
    gold = ["aapl-10K.html#183", "aapl-10K2.html#182"]
    rows = [
        {"source_doc": "aapl-10K.html", "chunk_index": 1},
        {"source_doc": "aapl-10K.html", "chunk_index": 183},
        {"source_doc": "aapl-10K2.html", "chunk_index": 182},
    ]
    metrics = compute_retrieval_metrics(rows, gold, k=3)
    assert metrics is not None
    assert metrics.recall_at_k == 1.0
    assert metrics.mrr == 0.5
    assert 0.0 < metrics.ndcg_at_k < 1.0
    assert compute_retrieval_metrics(rows, []) is None
    print("PASS test_retrieval_metrics_recall_mrr_ndcg")


def test_numeric_exact_match_is_unit_aware() -> None:
    assert numeric_exact_match("Net sales were $383.3 billion.", "Net sales were $383,285 million.") == 1.0
    assert numeric_exact_match("Net sales were $383 billion.", "Net sales were $383,285 million.") == 1.0
    assert numeric_exact_match("Growth was 24.5%.", "Growth was 24.5%.") == 1.0
    assert numeric_exact_match("Growth was 12%.", "Growth was 24.5%.") == 0.0
    assert numeric_exact_match("No numbers here.", "No numbers here either.") is None
    print("PASS test_numeric_exact_match_is_unit_aware")


def test_judge_route_uses_different_model() -> None:
    settings = Settings(openai_api_key="test", groq_api_key="test")
    order, overrides = _judge_route(settings, "openai", "gpt-4o")
    assert order[0] == "groq"
    assert overrides["groq"] != "gpt-4o"

    groq_only = Settings(groq_api_key="test")
    order, overrides = _judge_route(groq_only, "groq", "qwen/qwen3.5-9b")
    assert order[0] == "groq"
    assert overrides["groq"] != "qwen/qwen3.5-9b"
    print("PASS test_judge_route_uses_different_model")


def test_group_aggregation_includes_latency_and_cost() -> None:
    items = [
        {
            "metrics": {
                "context_precision": 1.0,
                "context_recall": 1.0,
                "faithfulness": 1.0,
                "answer_relevance": 1.0,
            },
            "numeric_accuracy": 1.0,
            "judge": {"score": 1.0},
            "latency_ms": 100.0,
            "cost_usd": 0.01,
        },
        {
            "metrics": {
                "context_precision": 0.5,
                "context_recall": 0.5,
                "faithfulness": 0.5,
                "answer_relevance": 0.5,
            },
            "numeric_accuracy": 0.5,
            "judge": {"score": 0.5},
            "latency_ms": 300.0,
            "cost_usd": 0.03,
        },
    ]
    aggregate = _aggregate_group(items)
    assert aggregate["n"] == 2
    assert aggregate["numeric_accuracy"] == 0.75
    assert aggregate["latency"]["p50_ms"] == 200.0
    assert aggregate["latency"]["p95_ms"] == 290.0
    assert aggregate["cost_usd_total"] == 0.04
    assert _percentile([], 50) is None
    print("PASS test_group_aggregation_includes_latency_and_cost")


def test_generated_gold_ids_point_at_real_chunks() -> None:
    from src.config import settings
    from src.retrieval.search import load_chunk_files

    available = {
        f"{chunk.metadata.get('source_file')}#{chunk.metadata.get('chunk_index')}"
        for chunk in load_chunk_files(settings.chunk_store_dir)
    }
    generated = [example for example in load_dataset() if example.id.startswith(("tbl_", "txt_"))]
    assert generated
    assert all(set(example.gold_chunk_ids) <= available for example in generated)
    print("PASS test_generated_gold_ids_point_at_real_chunks")


def main() -> int:
    test_dataset_expanded_balanced_and_labeled()
    test_retrieval_metrics_recall_mrr_ndcg()
    test_numeric_exact_match_is_unit_aware()
    test_judge_route_uses_different_model()
    test_group_aggregation_includes_latency_and_cost()
    test_generated_gold_ids_point_at_real_chunks()
    return 0


if __name__ == "__main__":
    sys.exit(main())

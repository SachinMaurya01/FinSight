"""Evaluation harness."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from src.config import settings
from src.eval.dataset import load_dataset
from src.eval.judge import evaluate_numeric_judge
from src.eval.metrics import compute_metrics

logger = logging.getLogger(__name__)


def run_eval(limit: int | None = None, output_path: Path | None = None) -> dict[str, Any]:
    """Run eval dataset through the pipeline and return aggregated results."""
    examples = load_dataset()
    if limit:
        examples = examples[:limit]

    from src.graph import ask
    from src.graph.checkpoint import get_checkpointer  # noqa: F401  # keep import for side-effect

    results: list[dict[str, Any]] = []
    total_metrics = {"context_precision": 0, "context_recall": 0, "faithfulness": 0, "answer_relevance": 0}
    judge_scores: list[float] = []

    start = time.time()
    for ex in examples:
        logger.info("Eval %s: %r", ex.id, ex.question)
        try:
            state = ask(ex.question, k=settings.rerank_top_k)
            answer = state.get("answer", "")
            chunks = state.get("chunks", [])
            contexts = [c.get("content", "") for c in chunks]
            expected_citations = [c.model_dump() for c in ex.expected_citations]

            metrics = compute_metrics(ex.question, answer, chunks, expected_citations, contexts)
            judge = evaluate_numeric_judge(ex.question, answer, ex.expected_answer, contexts)

            total_metrics["context_precision"] += metrics.context_precision
            total_metrics["context_recall"] += metrics.context_recall
            total_metrics["faithfulness"] += metrics.faithfulness
            total_metrics["answer_relevance"] += metrics.answer_relevance
            judge_scores.append(judge.score)

            results.append(
                {
                    "id": ex.id,
                    "question": ex.question,
                    "expected_answer": ex.expected_answer,
                    "answer": answer,
                    "complexity": ex.complexity,
                    "is_recommendation": ex.is_recommendation,
                    "verification": state.get("verification"),
                    "tier": state.get("tier"),
                    "provider": state.get("provider"),
                    "metrics": metrics.__dict__,
                    "judge": {"score": judge.score, "reasoning": judge.reasoning},
                }
            )
        except Exception as exc:
            logger.exception("Eval %s failed: %s", ex.id, exc)
            results.append({"id": ex.id, "question": ex.question, "error": str(exc)})

    n = len([r for r in results if "metrics" in r])
    elapsed = time.time() - start

    baseline = {
        "total_examples": len(examples),
        "successful": n,
        "elapsed_seconds": round(elapsed, 2),
        "avg_metrics": {k: round(v / max(n, 1), 4) for k, v in total_metrics.items()},
        "avg_judge": round(sum(judge_scores) / max(len(judge_scores), 1), 4),
        "judge_scores": judge_scores,
        "results": results,
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(baseline, indent=2), encoding="utf-8")
        logger.info("Baseline written to %s", output_path)
    else:
        default = Path("data/eval/baseline.json")
        default.parent.mkdir(parents=True, exist_ok=True)
        default.write_text(json.dumps(baseline, indent=2), encoding="utf-8")
        logger.info("Baseline written to %s", default)

    return baseline

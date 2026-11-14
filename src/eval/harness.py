"""Evaluation harness."""

from __future__ import annotations

import json
import logging
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any, cast

from src.config import ProviderName, settings
from src.eval.dataset import EvalExample, load_dataset
from src.eval.injection import summarize_injection
from src.eval.judge import evaluate_numeric_judge
from src.eval.metrics import (
    MetricsBackend,
    RagasUnavailableError,
    compute_metrics,
    compute_ragas_metrics,
    compute_retrieval_metrics,
    ensure_ragas_available,
    numeric_exact_match,
)
from src.observability.pricing import estimate_cost
from src.verification.abstention import ABSTAIN_MESSAGE

logger = logging.getLogger(__name__)

METRIC_NAMES = ("context_precision", "context_recall", "faithfulness", "answer_relevance")
RETRIEVAL_NAMES = ("recall_at_k", "mrr", "ndcg_at_k")


def summarize_abstention(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize abstention accuracy over examples that require abstention."""
    expected = [result for result in results if result.get("expected_abstention")]
    abstained = [result for result in expected if result.get("abstained")]
    correct_reason = [
        result
        for result in abstained
        if not result.get("expected_abstention_reason")
        or result.get("abstention_reason") == result["expected_abstention_reason"]
    ]
    return {
        "expected": len(expected),
        "abstained": len(abstained),
        "accuracy": round(len(abstained) / len(expected), 4) if expected else None,
        "reason_matches": len(correct_reason),
        "reason_accuracy": round(len(correct_reason) / len(expected), 4) if expected else None,
    }


def _percentile(values: list[float], percentile: float) -> float | None:
    """Linear-interpolated percentile (``percentile`` in ``[0, 100]``)."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 4)
    position = (len(ordered) - 1) * (percentile / 100.0)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return round(ordered[lower], 4)
    weight = position - lower
    return round(ordered[lower] * (1 - weight) + ordered[upper] * weight, 4)


def _latency_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "p50_ms": None, "p95_ms": None, "p99_ms": None, "mean_ms": None}
    return {
        "n": len(values),
        "p50_ms": _percentile(values, 50),
        "p95_ms": _percentile(values, 95),
        "p99_ms": _percentile(values, 99),
        "mean_ms": round(sum(values) / len(values), 4),
    }


def _mean_metrics(items: list[dict[str, Any]]) -> dict[str, float]:
    scored = [item for item in items if item.get("metrics")]
    return {
        name: round(sum(item["metrics"][name] for item in scored) / len(scored), 4) if scored else 0.0
        for name in METRIC_NAMES
    }


def _aggregate_group(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-tier/per-intent aggregate: quality, numeric accuracy, cost, latency."""
    scored = [item for item in items if item.get("metrics")]
    numeric = [item["numeric_accuracy"] for item in items if item.get("numeric_accuracy") is not None]
    judges = [item["judge"]["score"] for item in scored if item.get("judge")]
    latencies = [item["latency_ms"] for item in items if item.get("latency_ms") is not None]
    costs = [item["cost_usd"] for item in items if item.get("cost_usd") is not None]
    return {
        "n": len(items),
        "n_scored": len(scored),
        "avg_metrics": _mean_metrics(items),
        "numeric_accuracy": round(sum(numeric) / len(numeric), 4) if numeric else None,
        "n_numeric": len(numeric),
        "avg_judge": round(sum(judges) / len(judges), 4) if judges else None,
        "latency": _latency_summary(latencies),
        "cost_usd_total": round(sum(costs), 6),
        "cost_usd_mean": round(sum(costs) / len(costs), 6) if costs else None,
    }


def _retrieval_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [result["retrieval"] for result in results if result.get("retrieval")]
    summary: dict[str, Any] = {
        "n": len(scored),
        **{
            name: round(sum(item[name] for item in scored) / len(scored), 4) if scored else None
            for name in RETRIEVAL_NAMES
        },
    }
    summary["recall"] = summary["recall_at_k"]
    summary["ndcg"] = summary["ndcg_at_k"]
    return summary


def _judge_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    judged = [result for result in results if result.get("judge")]
    backends = Counter(result["judge"].get("backend", "unknown") for result in judged)
    heuristic_fallbacks = sum(1 for result in judged if result["judge"].get("used_heuristic_fallback"))
    judge_models = Counter(
        f"{result['judge'].get('provider') or 'unknown'}/{result['judge'].get('model') or 'unknown'}"
        for result in judged
        if result["judge"].get("backend") == "llm"
    )
    return {
        "n": len(judged),
        "backend_counts": dict(backends),
        "heuristic_fallback_runs": heuristic_fallbacks,
        "heuristic_fallback_rate": round(heuristic_fallbacks / len(judged), 4) if judged else None,
        "llm_models": dict(judge_models),
    }


def _dataset_summary(examples: list[EvalExample]) -> dict[str, Any]:
    return {
        "total": len(examples),
        "tiers": dict(Counter(example.complexity for example in examples)),
        "intents": dict(Counter(example.effective_intent for example in examples)),
        "categories": dict(Counter(example.category for example in examples)),
        "negatives": {
            "abstention": sum(1 for example in examples if example.expected_abstention),
            "prompt_injection": sum(1 for example in examples if example.injected_context),
        },
        "gold_chunk_labeled": sum(1 for example in examples if example.gold_chunk_ids),
    }


def _base_result(example: EvalExample, answer: str, state: dict[str, Any], latency_ms: float) -> dict[str, Any]:
    model = state.get("model") or ""
    tokens_in = int(state.get("tokens_in") or 0)
    tokens_out = int(state.get("tokens_out") or 0)
    return {
        "id": example.id,
        "question": example.question,
        "expected_answer": example.expected_answer,
        "answer": answer,
        "complexity": example.complexity,
        "intent": example.effective_intent,
        "category": example.category,
        "is_recommendation": example.is_recommendation,
        "gold_chunk_ids": example.gold_chunk_ids,
        "expected_abstention": example.expected_abstention,
        "expected_abstention_reason": example.expected_abstention_reason,
        "verification": state.get("verification"),
        "tier": state.get("tier"),
        "provider": state.get("provider"),
        "model": state.get("model"),
        "latency_ms": round(latency_ms, 4),
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cost_usd": round(estimate_cost(model, tokens_in, tokens_out), 6),
    }


def run_eval(
    limit: int | None = None,
    output_path: Path | None = None,
    metrics_backend: MetricsBackend = "heuristic",
) -> dict[str, Any]:
    """Run eval dataset through the pipeline and return aggregated results.

    ``metrics_backend="ragas"`` fails fast if RAGAS is unavailable. ``"auto"``
    tries RAGAS once and explicitly records a heuristic fallback instead of
    pretending the scores are RAGAS.

    The report includes per-tier and per-intent breakdowns, gold-chunk retrieval
    metrics (Recall@k/MRR/nDCG), unit-aware numeric accuracy, judge backend and
    heuristic-fallback usage, and p50/p95/p99 latency plus estimated cost.
    """
    if metrics_backend == "ragas":
        ensure_ragas_available()

    examples = load_dataset()
    if limit:
        examples = examples[:limit]

    from src.graph import ask

    results: list[dict[str, Any]] = []
    total_metrics: dict[str, float] = {name: 0.0 for name in METRIC_NAMES}
    judge_scores: list[float] = []
    active_backend: str = metrics_backend
    fallback_reason: str | None = None

    start = time.time()
    for ex in examples:
        logger.info("Eval %s: %r", ex.id, ex.question)
        try:
            started = time.perf_counter()
            state = ask(ex.question, k=settings.rerank_top_k, injected_context=ex.injected_context)
            latency_ms = (time.perf_counter() - started) * 1000
            answer = state.get("answer") or ""
            chunks = state.get("chunks", [])
            contexts = [c.get("content", "") for c in chunks]
            expected_citations = [c.model_dump() for c in ex.expected_citations]
            abstention_reason = state.get("abstention_reason")
            abstained = bool(abstention_reason) or answer.strip() == ABSTAIN_MESSAGE

            result = _base_result(ex, answer, state, latency_ms)
            result.update(
                {
                    "status": "abstained" if abstained else "completed",
                    "abstained": abstained,
                    "abstention_reason": abstention_reason,
                }
            )
            retrieval = compute_retrieval_metrics(chunks, ex.gold_chunk_ids, k=settings.rerank_top_k)
            if retrieval is not None:
                result["retrieval"] = retrieval.__dict__
            numeric = numeric_exact_match(answer, ex.expected_answer)
            result["numeric_accuracy"] = numeric

            if ex.forbidden_phrases:
                answer_lower = answer.lower()
                matched = [phrase for phrase in ex.forbidden_phrases if phrase.lower() in answer_lower]
                result["forbidden_phrases"] = ex.forbidden_phrases
                result["injection_followed"] = bool(matched)
                result["matched_phrases"] = matched

            if ex.expected_abstention:
                results.append(result)
                continue

            backend_for_example = active_backend
            if backend_for_example in ("ragas", "auto"):
                try:
                    metrics = compute_ragas_metrics(
                        ex.question,
                        answer,
                        contexts,
                        ex.expected_answer,
                        settings,
                    )
                    backend_for_example = "ragas"
                except RagasUnavailableError as exc:
                    if active_backend == "ragas":
                        raise
                    fallback_reason = str(exc)
                    logger.warning("RAGAS unavailable, using heuristic metrics: %s", exc)
                    active_backend = "heuristic"
                    backend_for_example = "heuristic"
                    metrics = compute_metrics(ex.question, answer, chunks, expected_citations, contexts)
            else:
                metrics = compute_metrics(ex.question, answer, chunks, expected_citations, contexts)

            judge = evaluate_numeric_judge(
                ex.question,
                answer,
                ex.expected_answer,
                contexts,
                active_settings=settings,
                generator_provider=cast(ProviderName | None, state.get("provider")),
                generator_model=state.get("model"),
            )

            total_metrics["context_precision"] += metrics.context_precision
            total_metrics["context_recall"] += metrics.context_recall
            total_metrics["faithfulness"] += metrics.faithfulness
            total_metrics["answer_relevance"] += metrics.answer_relevance
            judge_scores.append(judge.score)

            result.update(
                {
                    "metrics_backend": backend_for_example,
                    "metrics": metrics.__dict__,
                    "judge": {
                        "score": judge.score,
                        "reasoning": judge.reasoning,
                        "backend": judge.backend,
                        "provider": judge.provider,
                        "model": judge.model,
                        "used_heuristic_fallback": judge.used_heuristic_fallback,
                        "numeric_claims_checked": judge.numeric_claims_checked,
                        "numeric_claims_correct": judge.numeric_claims_correct,
                    },
                }
            )
            results.append(result)
        except Exception as exc:
            logger.exception("Eval %s failed: %s", ex.id, exc)
            results.append(
                {
                    "id": ex.id,
                    "question": ex.question,
                    "expected_abstention": ex.expected_abstention,
                    "expected_abstention_reason": ex.expected_abstention_reason,
                    "abstained": False,
                    "error": str(exc),
                }
            )

    scored = [result for result in results if "metrics" in result]
    n = len(scored)
    elapsed = time.time() - start
    latencies = [result["latency_ms"] for result in results if result.get("latency_ms") is not None]
    costs = [result["cost_usd"] for result in results if result.get("cost_usd") is not None]
    numeric_scores = [result["numeric_accuracy"] for result in results if result.get("numeric_accuracy") is not None]
    tiers = sorted({example.complexity for example in examples})
    intents = sorted({example.effective_intent for example in examples})

    report = {
        "total_examples": len(examples),
        "successful": n,
        "elapsed_seconds": round(elapsed, 2),
        "metrics_backend": active_backend,
        "metrics_fallback_reason": fallback_reason,
        "avg_metrics": {k: round(v / max(n, 1), 4) for k, v in total_metrics.items()},
        "avg_judge": round(sum(judge_scores) / max(len(judge_scores), 1), 4),
        "judge_scores": judge_scores,
        "judge": _judge_summary(results),
        "retrieval": _retrieval_summary(results),
        "numeric_accuracy": round(sum(numeric_scores) / len(numeric_scores), 4) if numeric_scores else None,
        "n_numeric": len(numeric_scores),
        "latency": _latency_summary(latencies),
        "cost": {
            "total_usd": round(sum(costs), 6),
            "mean_usd": round(sum(costs) / len(costs), 6) if costs else None,
            "tokens_in_total": sum(int(result.get("tokens_in") or 0) for result in results),
            "tokens_out_total": sum(int(result.get("tokens_out") or 0) for result in results),
        },
        "per_tier": {
            tier: _aggregate_group([result for result in results if result.get("complexity") == tier]) for tier in tiers
        },
        "per_intent": {
            intent: _aggregate_group([result for result in results if result.get("intent") == intent])
            for intent in intents
        },
        "abstention": summarize_abstention(results),
        "injection": summarize_injection(results),
        "dataset": _dataset_summary(examples),
        "results": results,
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info("Baseline written to %s", output_path)
    else:
        default = settings.data_dir / "eval" / "baseline.json"
        default.parent.mkdir(parents=True, exist_ok=True)
        default.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info("Baseline written to %s", default)

    return report

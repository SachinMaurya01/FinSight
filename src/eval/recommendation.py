"""Precision and recall for the investment-recommendation classifier.

The production gate is ``classify_recommendation``: the regex heuristic OR the
optional cheap-tier LLM classifier. This module scores that gate against a
labeled set so both error types are visible before the classifier is relied on.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from src.config import Settings
from src.config import settings as default_settings
from src.hitl.gate import classify_recommendation, is_recommendation

logger = logging.getLogger(__name__)

DEFAULT_RECOMMENDATION_DATASET = default_settings.data_dir / "eval" / "recommendation_dataset.json"

Classifier = Callable[[str, str], bool]


class RecommendationCase(BaseModel):
    """One labeled answer. ``source`` marks synthetic cases explicitly."""

    id: str
    query: str
    answer: str
    is_recommendation: bool
    source: str = "synthetic"


def load_recommendation_dataset(path: Path | None = None) -> list[RecommendationCase]:
    """Load and validate the labeled recommendation set."""
    target = path or DEFAULT_RECOMMENDATION_DATASET
    raw = json.loads(target.read_text(encoding="utf-8"))
    return [RecommendationCase.model_validate(item) for item in raw]


def score_predictions(predictions: list[bool], labels: list[bool]) -> dict[str, Any]:
    """Confusion-matrix metrics for the positive (recommendation) class."""
    if len(predictions) != len(labels):
        raise ValueError("predictions and labels must have the same length")
    tp = sum(1 for p, y in zip(predictions, labels, strict=True) if p and y)
    fp = sum(1 for p, y in zip(predictions, labels, strict=True) if p and not y)
    fn = sum(1 for p, y in zip(predictions, labels, strict=True) if not p and y)
    tn = sum(1 for p, y in zip(predictions, labels, strict=True) if not p and not y)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else None
    total = len(labels)
    return {
        "total": total,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": _round(precision),
        "recall": _round(recall),
        "f1": _round(f1),
        "accuracy": _round((tp + tn) / total if total else None),
    }


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


def run_recommendation_eval(
    classify: Classifier,
    path: Path | None = None,
) -> dict[str, Any]:
    """Score ``classify(answer, query)`` against the labeled set."""
    cases = load_recommendation_dataset(path)
    predictions = [bool(classify(case.answer, case.query)) for case in cases]
    labels = [case.is_recommendation for case in cases]
    misclassified = [
        {"id": case.id, "predicted": pred, "label": case.is_recommendation}
        for case, pred in zip(cases, predictions, strict=True)
        if pred != case.is_recommendation
    ]
    return {
        "metrics": score_predictions(predictions, labels),
        "misclassified": misclassified,
    }


def heuristic_classifier(answer: str, query: str) -> bool:
    """Regex-only gate (``RECOMMENDATION_CLASSIFIER_ENABLED=false``)."""
    return is_recommendation(answer, query)


def production_classifier(settings: Settings, llm: Any | None = None) -> Classifier:
    """Full gate: heuristic OR LLM classifier, as used by ``human_review_gate``."""
    enabled = settings.model_copy(update={"recommendation_classifier_enabled": True})

    def classify(answer: str, query: str) -> bool:
        return classify_recommendation(answer, query, enabled, llm=llm)

    return classify


def main(argv: list[str] | None = None) -> int:
    """``python -m src.eval.recommendation [--llm] [--output PATH]``."""
    parser = argparse.ArgumentParser(description="Score the recommendation classifier")
    parser.add_argument("--llm", action="store_true", help="include the LLM classifier (needs provider keys)")
    parser.add_argument("--output", type=Path, default=None, help="write the full report as JSON")
    args = parser.parse_args(argv)

    classify = production_classifier(default_settings) if args.llm else heuristic_classifier
    report = run_recommendation_eval(classify)
    report["classifier"] = "heuristic_or_llm" if args.llm else "heuristic"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"classifier": report["classifier"], **report["metrics"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

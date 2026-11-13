"""Recommendation-classifier eval: labeled set loads, metrics are correct, LLM path is OR-ed with the heuristic."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.eval.recommendation import (
    heuristic_classifier,
    load_recommendation_dataset,
    production_classifier,
    run_recommendation_eval,
    score_predictions,
)


class FakeLLM:
    """Classifier LLM stand-in that always answers with a fixed verdict."""

    def __init__(self, verdict: bool) -> None:
        self.verdict = verdict
        self.calls = 0

    def invoke(self, prompt: str) -> Any:
        self.calls += 1
        verdict = str(self.verdict).lower()
        return SimpleNamespace(content='{"is_recommendation": %s, "confidence": 0.9, "reason": "test"}' % verdict)


def test_dataset_is_labeled_and_balanced() -> None:
    cases = load_recommendation_dataset()
    labels = [case.is_recommendation for case in cases]
    assert len(cases) >= 20
    assert any(labels) and not all(labels)
    assert all(case.source == "synthetic" for case in cases)
    assert len({case.id for case in cases}) == len(cases)
    print("PASS test_dataset_is_labeled_and_balanced")


def test_score_predictions_math() -> None:
    metrics = score_predictions([True, True, False, False], [True, False, True, False])
    assert (metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"]) == (1, 1, 1, 1)
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["accuracy"] == 0.5
    empty = score_predictions([False], [False])
    assert empty["precision"] is None
    print("PASS test_score_predictions_math")


def test_heuristic_report_shape() -> None:
    report = run_recommendation_eval(heuristic_classifier)
    metrics = report["metrics"]
    assert metrics["total"] == len(load_recommendation_dataset())
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert all({"id", "predicted", "label"} <= set(item) for item in report["misclassified"])
    print("PASS test_heuristic_report_shape")


def test_llm_classifier_extends_heuristic_recall() -> None:
    cases = load_recommendation_dataset()
    baseline = run_recommendation_eval(heuristic_classifier)["metrics"]

    llm = FakeLLM(verdict=True)
    enabled_llm_report = run_recommendation_eval(production_classifier(settings, llm=llm))["metrics"]
    assert llm.calls > 0
    assert enabled_llm_report["recall"] >= baseline["recall"]
    assert enabled_llm_report["recall"] == 1.0
    assert enabled_llm_report["fp"] >= baseline["fp"]

    never = run_recommendation_eval(production_classifier(settings, llm=FakeLLM(verdict=False)))["metrics"]
    assert never["tp"] == baseline["tp"] and never["fp"] == baseline["fp"]
    assert len(cases) == never["total"]
    print("PASS test_llm_classifier_extends_heuristic_recall")


def main() -> int:
    test_dataset_is_labeled_and_balanced()
    test_score_predictions_math()
    test_heuristic_report_shape()
    test_llm_classifier_extends_heuristic_recall()
    return 0


if __name__ == "__main__":
    sys.exit(main())

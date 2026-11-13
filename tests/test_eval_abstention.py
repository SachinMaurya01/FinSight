"""Eval abstention coverage tests."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval.dataset import load_dataset
from src.eval.harness import summarize_abstention


def test_dataset_includes_out_of_corpus_abstention_cases() -> None:
    examples = load_dataset()
    abstention = [example for example in examples if example.expected_abstention]
    ids = {example.id for example in abstention}
    assert {"eval_031", "eval_032"} <= ids
    assert all(example.category == "abstention" for example in abstention)
    assert all(example.expected_abstention_reason == "no_evidence" for example in abstention)
    assert all(not example.expected_citations for example in abstention)
    print("PASS test_dataset_includes_out_of_corpus_abstention_cases")


def test_abstention_summary() -> None:
    results = [
        {
            "expected_abstention": True,
            "expected_abstention_reason": "no_evidence",
            "abstained": True,
            "abstention_reason": "no_evidence",
        },
        {
            "expected_abstention": True,
            "expected_abstention_reason": "no_evidence",
            "abstained": True,
            "abstention_reason": "low_retrieval_confidence",
        },
        {
            "expected_abstention": True,
            "expected_abstention_reason": "no_evidence",
            "abstained": False,
            "abstention_reason": None,
        },
        {"expected_abstention": False, "abstained": True},
    ]
    summary = summarize_abstention(results)
    assert summary == {
        "expected": 3,
        "abstained": 2,
        "accuracy": 0.6667,
        "reason_matches": 1,
        "reason_accuracy": 0.3333,
    }
    assert summarize_abstention([])["accuracy"] is None
    print("PASS test_abstention_summary")


def main() -> int:
    test_dataset_includes_out_of_corpus_abstention_cases()
    test_abstention_summary()
    return 0


if __name__ == "__main__":
    sys.exit(main())

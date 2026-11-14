"""Eval regression gate tests."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval.gate import compare_metrics, extract_metrics, main, run_gate

BASE = {
    "faithfulness": 0.90,
    "context_recall": 0.80,
    "context_precision": 0.70,
    "answer_relevance": 0.60,
}


def _write(payload: dict, directory: str, name: str) -> Path:
    path = Path(directory) / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_compare_within_tolerance() -> None:
    current = {**BASE, "faithfulness": 0.88, "answer_relevance": 0.56}
    assert compare_metrics(BASE, current) == []
    print("PASS test_compare_within_tolerance")


def test_compare_detects_regression() -> None:
    current = {**BASE, "faithfulness": 0.80}
    failures = compare_metrics(BASE, current)
    assert len(failures) == 1
    assert failures[0].startswith("faithfulness:")
    print("PASS test_compare_detects_regression")


def test_missing_metric_fails() -> None:
    current = {key: value for key, value in BASE.items() if key != "context_recall"}
    failures = compare_metrics(BASE, current)
    assert any("missing from new run" in failure for failure in failures)
    print("PASS test_missing_metric_fails")


def test_extract_shapes() -> None:
    assert extract_metrics({"avg_metrics": BASE}) == BASE
    assert extract_metrics({"aggregate": BASE}) == BASE
    retrieval = {
        "configs": {
            "hybrid": {"context_precision": 0.5, "mrr": 0.4, "n_scored": 10},
            "hybrid_rerank": {"context_precision": 0.7, "mrr": 0.9, "n_scored": 10},
        }
    }
    assert extract_metrics(retrieval) == {
        "context_precision": 0.7,
        "mrr": 0.9,
        "n_scored": 10.0,
    }
    assert extract_metrics(retrieval, section="hybrid") == {
        "context_precision": 0.5,
        "mrr": 0.4,
        "n_scored": 10.0,
    }
    print("PASS test_extract_shapes")


def test_extract_includes_expanded_metrics() -> None:
    payload = {
        "avg_metrics": BASE,
        "retrieval": {"n": 10, "recall_at_k": 0.8, "mrr": 0.7, "ndcg_at_k": 0.75},
        "numeric_accuracy": 0.9,
    }
    metrics = extract_metrics(payload)
    assert metrics["faithfulness"] == BASE["faithfulness"]
    assert metrics["recall_at_k"] == 0.8
    assert metrics["mrr"] == 0.7
    assert metrics["ndcg_at_k"] == 0.75
    assert metrics["numeric_accuracy"] == 0.9
    print("PASS test_extract_includes_expanded_metrics")


def test_run_gate_and_cli_exit_codes() -> None:
    with tempfile.TemporaryDirectory() as directory:
        baseline = _write({"avg_metrics": BASE}, directory, "baseline.json")
        good = _write({"avg_metrics": {**BASE, "faithfulness": 0.89}}, directory, "good.json")
        bad = _write({"avg_metrics": {**BASE, "faithfulness": 0.50}}, directory, "bad.json")

        passed, failures = run_gate(baseline, good)
        assert passed and failures == []
        passed, failures = run_gate(baseline, bad)
        assert not passed and failures
        assert main(["--baseline", str(baseline), "--new", str(good)]) == 0
        assert main(["--baseline", str(baseline), "--new", str(bad)]) == 1
        assert main(["--baseline", str(baseline), "--new", str(Path(directory) / "nope.json")]) == 2
    print("PASS test_run_gate_and_cli_exit_codes")


def main_test() -> int:
    test_compare_within_tolerance()
    test_compare_detects_regression()
    test_missing_metric_fails()
    test_extract_shapes()
    test_extract_includes_expanded_metrics()
    test_run_gate_and_cli_exit_codes()
    return 0


if __name__ == "__main__":
    sys.exit(main_test())

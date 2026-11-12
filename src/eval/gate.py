"""Eval regression gate.

Compares a fresh eval run against a checked-in baseline and exits non-zero when
any tracked metric drops by more than its tolerance. Supports both answer-level
runs (``avg_metrics``/``aggregate``) and offline retrieval ablation reports
(``configs.<name>``).

Usage:
    python -m src.eval.gate --baseline data/eval/baseline.json --new /tmp/eval.json

    python -m src.eval.gate \
        --baseline data/eval/smoke_baseline.json \
        --new /tmp/eval_smoke.json \
        --section hybrid_rerank
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

TOLERANCE: dict[str, float] = {
    "faithfulness": 0.03,
    "context_recall": 0.03,
    "context_precision": 0.03,
    "numeric_accuracy": 0.02,
    "answer_relevance": 0.05,
    "mrr": 0.03,
}
DEFAULT_TOLERANCE = 0.03
RETRIEVAL_SECTION = "hybrid_rerank"


class GateInputError(RuntimeError):
    """Raised when a baseline or new eval report has an unsupported shape."""


def load_report(path: str | Path) -> dict[str, Any]:
    """Load a JSON eval report, raising a clear error for missing/invalid files."""
    report_path = Path(path)
    if not report_path.exists():
        raise GateInputError(f"eval report not found: {report_path}")
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise GateInputError(f"invalid JSON in {report_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise GateInputError(f"eval report must be a JSON object: {report_path}")
    return payload


def extract_metrics(
    payload: dict[str, Any],
    *,
    section: str | None = None,
) -> dict[str, float]:
    """Return metric name -> score from an eval or ablation report."""
    for key in ("avg_metrics", "aggregate", "metrics"):
        candidate = payload.get(key)
        if isinstance(candidate, dict):
            return _numeric_metrics(candidate)

    configs = payload.get("configs")
    if isinstance(configs, dict):
        name = section or RETRIEVAL_SECTION
        candidate = configs.get(name)
        if not isinstance(candidate, dict):
            raise GateInputError(f"retrieval report has no config {name!r} (available: {', '.join(sorted(configs))})")
        return _numeric_metrics(candidate)

    raise GateInputError("unsupported eval report shape: expected 'avg_metrics', 'aggregate', or 'configs'")


def _numeric_metrics(values: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        metrics[name] = float(value)
    if not metrics:
        raise GateInputError("report contains no numeric metrics")
    return metrics


def compare_metrics(
    baseline: dict[str, float],
    current: dict[str, float],
    tolerances: dict[str, float] | None = None,
) -> list[str]:
    """Return human-readable regression failures (empty when the gate passes)."""
    tol = {**TOLERANCE, **(tolerances or {})}
    failures: list[str] = []
    for metric, base_value in sorted(baseline.items()):
        allowed_drop = tol.get(metric, DEFAULT_TOLERANCE)
        if metric not in current:
            failures.append(f"{metric}: missing from new run (baseline {base_value:.4f})")
            continue
        value = current[metric]
        if value < base_value - allowed_drop:
            failures.append(
                f"{metric}: {value:.4f} < {base_value - allowed_drop:.4f} "
                f"(baseline {base_value:.4f}, tolerance {allowed_drop:.3f})"
            )
    return failures


def run_gate(
    baseline_path: str | Path,
    new_path: str | Path,
    *,
    section: str | None = None,
    tolerances: dict[str, float] | None = None,
) -> tuple[bool, list[str]]:
    """Compare two reports; returns ``(passed, failures)``."""
    baseline_payload = load_report(baseline_path)
    current_payload = load_report(new_path)
    baseline = extract_metrics(baseline_payload, section=section)
    current = extract_metrics(current_payload, section=section)
    failures = compare_metrics(baseline, current, tolerances)
    return not failures, failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, help="checked-in baseline JSON")
    parser.add_argument("--new", required=True, help="fresh eval JSON to gate")
    parser.add_argument(
        "--section",
        default=None,
        help=f"retrieval config to compare (default: {RETRIEVAL_SECTION})",
    )
    args = parser.parse_args(argv)

    try:
        passed, failures = run_gate(args.baseline, args.new, section=args.section)
    except GateInputError as exc:
        print(f"Eval gate input error: {exc}", file=sys.stderr)
        return 2

    if passed:
        print(f"Eval gate passed: {args.new} is within tolerance of {args.baseline}")
        return 0

    print("Eval regression detected:", file=sys.stderr)
    for failure in failures:
        print(f"  - {failure}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())

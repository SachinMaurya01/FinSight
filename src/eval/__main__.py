"""Evaluation CLI: full pipeline eval or offline retrieval ablation."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from src.eval.ablation import format_markdown_table, run_retrieval_ablation
from src.eval.harness import run_eval
from src.eval.metrics import RagasUnavailableError


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="limit examples for quick run")
    parser.add_argument("--output", type=str, default=None, help="output JSON path")
    parser.add_argument(
        "--metrics",
        choices=("heuristic", "ragas", "auto"),
        default="heuristic",
        help="answer metric backend (default: heuristic; 'ragas' fails if unavailable)",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="run retrieval-only dense vs hybrid vs hybrid+rerank ablation (no LLM/DB)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    out = Path(args.output) if args.output else None

    if args.ablation:
        report = run_retrieval_ablation(limit=args.limit, output_path=out)
        print(format_markdown_table(report))
        print(
            f"\nScored {report['n_scored']}/{report['n_examples']} questions "
            f"(no-citation tool questions excluded), k={report['k']}, "
            f"chunks={report['chunk_count']}."
        )
        return 0

    try:
        baseline = run_eval(limit=args.limit, output_path=out, metrics_backend=args.metrics)
    except RagasUnavailableError as exc:
        print(f"RAGAS backend unavailable: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "metrics_backend": baseline["metrics_backend"],
                "avg_metrics": baseline["avg_metrics"],
                "avg_judge": baseline["avg_judge"],
                "successful": baseline["successful"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""CLI for."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from src.eval.harness import run_eval


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="limit examples for quick run")
    parser.add_argument("--output", type=str, default=None, help="output JSON path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    from pathlib import Path

    out = Path(args.output) if args.output else None
    baseline = run_eval(limit=args.limit, output_path=out)
    print(json.dumps({"avg_metrics": baseline["avg_metrics"], "avg_judge": baseline["avg_judge"], "successful": baseline["successful"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

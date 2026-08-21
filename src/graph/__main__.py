"""CLI for Phase 5 — ask a question through the LangGraph pipeline.

Usage:
    python -m src.graph ask "what are the main risks" --k 5
"""

from __future__ import annotations

import argparse
import logging
import sys

from src.config import settings
from src.graph import ask as run_graph
from src.llm.fallback import DegradedResponseError


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    ask_parser = subparsers.add_parser("ask", help="run a query through the LangGraph pipeline")
    ask_parser.add_argument("query", type=str)
    ask_parser.add_argument("--k", type=int, default=settings.rerank_top_k)

    args = parser.parse_args()

    if args.command == "ask":
        try:
            state = run_graph(args.query, k=args.k)
        except DegradedResponseError as exc:
            print("DEGRADED MODE: all LLM providers failed.", file=sys.stderr)
            print(str(exc), file=sys.stderr)
            return 2
        print(f"tier={state.get('tier')} provider={state.get('provider')} "
              f"tokens_in={state.get('tokens_in')}")
        print("=== ANSWER ===")
        print(state["answer"])
        print("\n=== SOURCES ===")
        for source in state["sources"]:
            print(f"  {source}")
        return 0

    return 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
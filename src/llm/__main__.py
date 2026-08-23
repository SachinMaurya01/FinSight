"""CLI for end-to-end answer (retrieve -> LLM).

Usage:
    python -m src.llm answer "what are the main risks" --k 5
"""

from __future__ import annotations

import argparse
import logging
import sys

from src.config import settings
from src.ingestion.chunker import Chunk
from src.llm.answer import answer_query
from src.retrieval.search import retrieve_dense

CONTEXT_CHARS = 260


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    answer_parser = subparsers.add_parser("answer", help="retrieve chunks and answer a question")
    answer_parser.add_argument("query", type=str)
    answer_parser.add_argument("--k", type=int, default=settings.rerank_top_k)

    args = parser.parse_args()

    if args.command == "answer":
        rows = retrieve_dense(args.query, settings, k=args.k)
        chunks = [Chunk(content=row["content"], metadata=row) for row in rows]
        print(f"Retrieved {len(chunks)} chunks for query: {args.query!r}")
        for rank, row in enumerate(rows, start=1):
            print(f"  #{rank} {row['ticker']} {row['filing_type']} {row['fiscal_period']} "
                  f"| {row['section']} (score={row['score']:.3f})")
            print(f"    {row['content'][:CONTEXT_CHARS].strip()}")
        print("\n=== ANSWER ===")
        answer = answer_query(args.query, chunks, settings)
        print(answer)
        return 0

    return 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())

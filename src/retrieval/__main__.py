"""CLI for Phase 3 — seed pgvector and run dense retrieval.

Usage:
    python -m src.retrieval seed
    python -m src.retrieval search "what are the main risks" --k 5
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.config import settings
from src.retrieval.search import retrieve_dense, seed_database
from src.retrieval.storage import get_engine, init_schema


def _print_result(row: dict, rank: int) -> None:
    print(f"\n#{rank}  score={row['score']:.4f}  "
          f"{row['ticker']} {row['filing_type']} {row['fiscal_period']} | {row['section']}")
    print(row["content"][:220].replace("\n", " "))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    seed_parser = subparsers.add_parser("seed", help="create schema and insert embedded chunks")
    seed_parser.add_argument(
        "--chunk-dir",
        type=Path,
        default=settings.chunk_store_dir,
        help="directory of *_chunks.json files (default: data/chunks)",
    )

    search_parser = subparsers.add_parser("search", help="dense similarity search")
    search_parser.add_argument("query", type=str)
    search_parser.add_argument("--k", type=int, default=settings.rerank_top_k)

    args = parser.parse_args()

    if args.command == "seed":
        engine = get_engine(settings)
        init_schema(engine, settings.embedding_dimensions)
        inserted = seed_database(settings, engine, chunk_dir=args.chunk_dir)
        print(f"Inserted {inserted} chunks into pgvector")
        return 0

    if args.command == "search":
        results = retrieve_dense(args.query, settings, k=args.k)
        print(f"Top {len(results)} results for query: {args.query!r}")
        for rank, row in enumerate(results, start=1):
            _print_result(row, rank)
        return 0

    return 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
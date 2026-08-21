"""CLI for Phase 3/6 — seed pgvector, run dense and hybrid retrieval.

Usage:
    python -m src.retrieval seed
    python -m src.retrieval search "what are the main risks" --k 5
    python -m src.retrieval hybrid "total net sales were 383 billion" --k 5
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.config import settings
from src.retrieval.hybrid import retrieve_hybrid
from src.retrieval.rerank import retrieve_reranked
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

    hybrid_parser = subparsers.add_parser("hybrid", help="dense + BM25 hybrid search")
    hybrid_parser.add_argument("query", type=str)
    hybrid_parser.add_argument("--k", type=int, default=settings.rerank_top_k)

    rerank_parser = subparsers.add_parser("rerank", help="hybrid search + cross-encoder rerank")
    rerank_parser.add_argument("query", type=str)
    rerank_parser.add_argument("--k", type=int, default=settings.rerank_top_k)

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

    if args.command == "hybrid":
        results = retrieve_hybrid(args.query, settings, k=args.k)
        print(f"Top {len(results)} hybrid results for query: {args.query!r}")
        for rank, row in enumerate(results, start=1):
            _print_result(row, rank)
        return 0

    if args.command == "rerank":
        results = retrieve_reranked(args.query, settings, k=args.k)
        print(f"Top {len(results)} reranked results for query: {args.query!r}")
        for rank, row in enumerate(results, start=1):
            score = row.get("rerank_score", row.get("score"))
            print(f"\n#{rank}  rerank_score={score:.4f}  "
                  f"{row['ticker']} {row['filing_type']} {row['fiscal_period']} | {row['section']}")
            print(row["content"][:220].replace("\n", " "))
        return 0

    return 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
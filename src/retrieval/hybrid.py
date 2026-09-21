"""Hybrid fusion (dense + BM25).

Why hybrid search?
------------------
Semantic (dense) retrieval finds chunks by meaning, so synonyms and
paraphrases match even with no shared keywords. But it can miss exact terms:
- tickers (e.g. "NVDA"), cusips, NaN-like codes, precise figures and
  rare/out-of-vocabulary jargon that embeddings rarely capture well.
- short queries where semantic similarity is noisy.

BM25 (lexical) retrieval is the opposite: it matches exact token overlap and
handles rare identifiers perfectly, but fails on paraphrases.

Combining both with Reciprocal Rank Fusion (RRF) covers both regimes:
- chunks that rank high lexically (exact hit on a ticker/figure) still rank
  well even if their semantic score is low;
- chunks that only match semantically (paraphrased concept) are not drowned
  out because lexical-only docs don't get a free ride on the dense list.

Net effect: higher recall (both signals surface) and higher precision
(consensus between the two lists ranks higher via additive RRF scores),
which is the core of the hybrid retrieval stage.
"""

from __future__ import annotations

import logging
from typing import Any

from src.config import Settings
from src.retrieval.bm25 import build_bm25_index
from src.retrieval.search import retrieve_dense

logger = logging.getLogger(__name__)


def _identity(row: dict[str, Any]) -> tuple[Any, Any]:
    return (row.get("source_doc"), row.get("chunk_index"))


def fuse_hybrid(
    dense_rows: list[dict[str, Any]],
    bm25_rows: list[dict[str, Any]],
    settings: Settings,
) -> list[dict[str, Any]]:
    """Merge two ranked lists via weighted reciprocal-rank fusion (RRF)."""
    entries: dict[tuple[Any, Any], dict[str, Any]] = {}

    def _accumulate(rows: list[dict[str, Any]], weight: float) -> None:
        for rank, row in enumerate(rows, start=1):
            entry = entries.setdefault(
                _identity(row), {"row": dict(row), "dense": 0.0, "lexical": 0.0}
            )
            contribution = weight / (settings.rrf_k + rank)
            if weight == settings.dense_weight:
                entry["dense"] += contribution
            else:
                entry["lexical"] += contribution

    _accumulate(dense_rows, settings.dense_weight)
    _accumulate(bm25_rows, settings.lexical_weight)

    fused: list[dict[str, Any]] = []
    for entry in entries.values():
        row = entry["row"]
        row["score"] = entry["dense"] + entry["lexical"]
        row["dense_score"] = entry["dense"]
        row["lexical_score"] = entry["lexical"]
        fused.append(row)
    fused.sort(key=lambda row: row["score"], reverse=True)
    return fused


def retrieve_hybrid(query: str, settings: Settings, k: int | None = None) -> list[dict[str, Any]]:
    """Run dense + BM25 retrieval and fuse into one top-``k`` ranked list."""
    k = k if k is not None else settings.fusion_top_k
    dense_rows = retrieve_dense(query, settings, k=settings.dense_top_k)
    bm25_rows = build_bm25_index(settings.chunk_store_dir).search(query, settings.bm25_top_k)
    fused = fuse_hybrid(dense_rows, bm25_rows, settings)
    logger.info("Hybrid retrieval: %d dense + %d bm25 -> %d fused (top %d)",
                len(dense_rows), len(bm25_rows), len(fused), k)
    return fused[:k]
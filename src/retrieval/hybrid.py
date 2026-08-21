"""Phase 6 — hybrid retrieval: weighted reciprocal-rank fusion of dense + BM25.

PRD §4.4 FR-10: fuse dense and BM25 results via a hybrid scoring strategy
(reciprocal rank fusion) with configurable weighting. Weights live in settings
(``dense_weight`` / ``lexical_weight``, RRF smoothing ``rrf_k``).
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
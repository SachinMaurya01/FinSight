"""Cross-encoder reranking."""

from __future__ import annotations

import logging
from typing import Any

from src.config import Settings
from src.retrieval.hybrid import retrieve_hybrid

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Thin wrapper over a sentence-transformers cross-encoder."""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name, device=device)

    def score(self, query: str, documents: list[str]) -> list[float]:
        """Return one relevance score per ``document`` against ``query``."""
        return [float(x) for x in self._model.predict([(query, doc) for doc in documents])]


def build_reranker(settings: Settings) -> CrossEncoderReranker | None:
    """Build the configured reranker, or ``None`` if disabled (empty model name)."""
    if not settings.reranker_model:
        return None
    logger.info("Loading cross-encoder reranker %s", settings.reranker_model)
    return CrossEncoderReranker(settings.reranker_model, device=settings.reranker_device)


def rerank_candidates(
    query: str,
    rows: list[dict[str, Any]],
    settings: Settings,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """Rerank ``rows`` by query relevance and return the top ``top_k``.

    Each row keeps its original fields plus a ``rerank_score``. If no reranker
    is configured, the input order is returned unchanged (truncated to top_k).
    """
    top_k = top_k if top_k is not None else settings.rerank_top_k
    reranker = build_reranker(settings)
    if reranker is None:
        logger.warning("No reranker configured; skipping rerank")
        return rows[:top_k]
    if not rows:
        return rows
    scores = reranker.score(query, [row["content"] for row in rows])
    for row, score in zip(rows, scores):
        row["rerank_score"] = score
    rows.sort(key=lambda row: row["rerank_score"], reverse=True)
    logger.info("Reranked %d candidates -> top %d", len(rows), top_k)
    return rows[:top_k]


def retrieve_reranked(
    query: str,
    settings: Settings,
    k: int | None = None,
) -> list[dict[str, Any]]:
    """Hybrid retrieval fused to ``fusion_top_k``, then reranked to top ``k``."""
    k = k if k is not None else settings.rerank_top_k
    fused = retrieve_hybrid(query, settings, k=settings.fusion_top_k)
    return rerank_candidates(query, fused, settings, top_k=k)
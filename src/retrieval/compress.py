"""Phase 8 — context compression: trim reranked chunks to query-relevant spans.

PRD §4.5 FR-12 / Phase 8: keep only the sentences in each chunk that are
relevant to the query, and enforce ``settings.context_token_budget`` before the
context reaches the LLM node. Relevance uses the configured embedder's
cosine similarity (no extra LLM call).
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any

from src.config import Settings
from src.retrieval.embeddings import Embedder, build_embedder
from src.tokens import count_tokens

logger = logging.getLogger(__name__)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_KEEP_SIM_RATIO = 0.6


def split_sentences(text: str) -> list[str]:
    """Split text into sentences on ``.``/``!``/``?`` boundaries."""
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def compress_chunks(
    query: str,
    rows: list[dict[str, Any]],
    settings: Settings,
    embedder: Embedder | None = None,
) -> list[dict[str, Any]]:
    """Compress each chunk to its query-relevant sentences within the token budget.

    Returns new rows (originals untouched) with ``content`` trimmed and a
    ``compressed`` flag set when sentences were dropped.
    """
    budget = settings.context_token_budget
    active_embedder = embedder or build_embedder(settings)
    query_embedding = active_embedder.embed_texts([query])[0]

    out: list[dict[str, Any]] = []
    used = 0
    for row in rows:
        sentences = split_sentences(row["content"])
        if not sentences:
            continue
        sentence_embeddings = active_embedder.embed_texts(sentences)
        similarities = [_cosine(query_embedding, emb) for emb in sentence_embeddings]
        best = max(similarities)
        kept = [
            sentence
            for sentence, sim in zip(sentences, similarities)
            if sim >= best * _KEEP_SIM_RATIO
        ]
        trimmed = " ".join(kept).strip()
        if not trimmed:
            trimmed = sentences[similarities.index(best)]

        # Fit the remaining budget: truncate with an ellipsis if needed.
        remaining = budget - used
        if count_tokens(trimmed) > remaining:
            tokens = 0
            kept_short: list[str] = []
            for sentence in kept:
                n = count_tokens(sentence)
                if tokens + n > remaining:
                    break
                kept_short.append(sentence)
                tokens += n
            trimmed = " ".join(kept_short)
            if not trimmed:
                trimmed = sentences[similarities.index(best)][: max(remaining * 4, 8)]

        used += count_tokens(trimmed)
        new_row = dict(row)
        new_row["content"] = trimmed
        new_row["compressed"] = trimmed != row["content"]
        out.append(new_row)
        if used >= budget:
            break

    total_before = sum(count_tokens(row["content"]) for row in rows)
    total_after = sum(count_tokens(row["content"]) for row in out)
    logger.info(
        "Compressed context: %d tokens -> %d tokens (budget=%d, chunks=%d)",
        total_before,
        total_after,
        budget,
        len(out),
    )
    return out
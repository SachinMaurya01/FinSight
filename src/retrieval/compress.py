"""Context compression."""

from __future__ import annotations

import logging
import math
import re
from typing import Any

from src.config import Settings
from src.retrieval.embeddings import Embedder, build_embedder
from src.tokens import count_tokens

logger = logging.getLogger(__name__)

# Splits chunks into sentences on ". "/"! "/"? " to keep only query-relevant ones.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_KEEP_SIM_RATIO = 0.6
_ENTITY_RE = re.compile(r"\b[A-Z][A-Za-z&.-]{2,}\b|\b\d[\d,]*\.?\d*\b")
_ENTITY_STOPWORDS = {
    "the",
    "and",
    "for",
    "from",
    "what",
    "when",
    "where",
    "which",
    "how",
    "was",
    "were",
    "did",
    "does",
}


def query_entities(query: str) -> list[str]:
    """Extract likely entity tokens (tickers, proper nouns, years, figures)."""
    entities: list[str] = []
    for match in _ENTITY_RE.finditer(query):
        token = match.group(0).strip()
        key = token.lower()
        if key in _ENTITY_STOPWORDS:
            continue
        if key not in {existing.lower() for existing in entities}:
            entities.append(token)
    return entities


def _sentence_has_entity(sentence: str, entities: list[str]) -> bool:
    lower = sentence.lower()
    return any(entity.lower() in lower for entity in entities)


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
    entities = query_entities(query)

    out: list[dict[str, Any]] = []
    used = 0
    for row in rows:
        if row.get("chunk_type") == "table":
            # Tables carry row/header relationships and must stay intact.
            if used + count_tokens(row["content"]) > budget:
                continue
            new_row = dict(row)
            new_row["compressed"] = False
            used += count_tokens(row["content"])
            out.append(new_row)
            continue
        sentences = split_sentences(row["content"])
        if not sentences:
            continue
        sentence_embeddings = active_embedder.embed_texts(sentences)
        similarities = [_cosine(query_embedding, emb) for emb in sentence_embeddings]
        best = max(similarities)
        relevant = {index for index, sim in enumerate(similarities) if sim >= best * _KEEP_SIM_RATIO}
        # Never drop a sentence that names a figure/year/entity from the question.
        entity_hits = {index for index, sentence in enumerate(sentences) if _sentence_has_entity(sentence, entities)}
        selected = relevant | entity_hits
        if not selected:
            selected = {similarities.index(best)}

        remaining = budget - used
        entity_tokens = sum(count_tokens(sentences[index]) for index in entity_hits)
        # Entity-bearing sentences are mandatory, even if they alone exceed budget.
        available = max(remaining - entity_tokens, 0)
        chosen: set[int] = set(entity_hits)
        extra_tokens = 0
        for index in sorted(selected - entity_hits, key=lambda i: similarities[i], reverse=True):
            n = count_tokens(sentences[index])
            if extra_tokens + n > available:
                continue
            chosen.add(index)
            extra_tokens += n
        if not chosen:
            chosen = {max(selected, key=lambda index: similarities[index])}
        trimmed = " ".join(sentences[index] for index in sorted(chosen)).strip()
        if not entity_hits and count_tokens(trimmed) > remaining:
            trimmed = trimmed[: max(remaining * 4, 8)].rstrip()
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

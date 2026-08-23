"""Retrieval API."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from sqlalchemy import Engine

from src.config import Settings
from src.ingestion.chunker import Chunk
from src.retrieval.embeddings import build_embedder, embed_chunks
from src.retrieval.storage import get_engine, insert_chunks, similarity_search

logger = logging.getLogger(__name__)


def load_chunk_files(chunk_store_dir: Path) -> list[Chunk]:
    """Load all chunks from ``data/chunks/*_chunks.json`` into ``Chunk`` objects."""
    chunks: list[Chunk] = []
    for path in sorted(chunk_store_dir.glob("*_chunks.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for item in payload.get("chunks", []):
            chunks.append(Chunk(content=item["content"], metadata=item.get("metadata", {})))
    logger.info("Loaded %d chunks from %s", len(chunks), chunk_store_dir)
    return chunks


def seed_database(
    settings: Settings,
    engine: Engine | None = None,
    chunk_dir: Path | None = None,
) -> int:
    """Embed all stored chunks and insert them into pgvector; returns count inserted."""
    engine = engine or get_engine(settings)
    chunks = load_chunk_files(chunk_dir or settings.chunk_store_dir)
    embedded = embed_chunks(chunks, settings)
    return insert_chunks(engine, embedded)


def retrieve_dense(query: str, settings: Settings, k: int | None = None) -> list[dict[str, Any]]:
    """Dense retrieval: embed ``query``, then top-``k`` cosine similarity search."""
    k = k if k is not None else settings.rerank_top_k
    engine = get_engine(settings)
    # Section
    use_cache = "test" not in settings.postgres_database_url
    if use_cache:
        try:
            from src.retrieval.cache import get_cached_embedding, set_cached_embedding

            cached = get_cached_embedding(settings.embedding_model, query, settings)
            if cached is not None:
                logger.info("Cache hit for query embedding")
                return similarity_search(engine, cached, k)
        except Exception:
            pass
    embedder = build_embedder(settings)
    query_embedding = embedder.embed_texts([query])[0]
    if use_cache:
        try:
            from src.retrieval.cache import set_cached_embedding

            set_cached_embedding(settings.embedding_model, query, query_embedding, settings)
        except Exception:
            pass
    return similarity_search(engine, query_embedding, k)
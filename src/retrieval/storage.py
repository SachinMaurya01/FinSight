"""Phase 3 — pgvector storage: schema, bulk insert, and similarity search.

Schema (PRD Phase 3): ``chunks(id, content, embedding vector(N), ticker,
filing_type, fiscal_period, section, source_doc, chunk_index, char_offset,
metadata jsonb)`` with an HNSW cosine index for approximate nearest-neighbor
search.

Connection uses ``settings.postgres_database_url`` (Postgres + pgvector must be
running locally).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from sqlalchemy import Engine, create_engine, text

from src.config import Settings
from src.retrieval.embeddings import ChunkEmbedding

logger = logging.getLogger(__name__)

_CREATE_EXTENSION_SQL = "CREATE EXTENSION IF NOT EXISTS vector;"

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS chunks (
    id BIGSERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding vector({dim}),
    ticker TEXT,
    filing_type TEXT,
    fiscal_period TEXT,
    section TEXT,
    source_doc TEXT,
    chunk_index INT,
    char_offset INT,
    metadata JSONB
);
"""

_CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);
"""

_INSERT_CHUNK_SQL = text("""
    INSERT INTO chunks
        (content, embedding, ticker, filing_type, fiscal_period, section,
         source_doc, chunk_index, char_offset, metadata)
    VALUES
        (:content, CAST(:embedding AS vector), :ticker, :filing_type,
         :fiscal_period, :section, :source_doc, :chunk_index, :char_offset,
         CAST(:metadata AS jsonb))
""")

_SIMILARITY_SQL = text("""
    SELECT content, ticker, filing_type, fiscal_period, section, source_doc,
           1 - (embedding <=> CAST(:query AS vector)) AS score
    FROM chunks
    ORDER BY embedding <=> CAST(:query AS vector)
    LIMIT :k
""")


def get_engine(settings: Settings) -> Engine:
    """Create a SQLAlchemy engine for the configured Postgres database."""
    return create_engine(settings.postgres_database_url, pool_pre_ping=True)


def _column_dimension(conn: Any) -> int | None:
    """Return the configured dimension of ``chunks.embedding``, or ``None`` if the table is absent."""
    row = conn.execute(
        text(
            """
            SELECT format_type(a.atttypid, a.atttypmod) AS coltype
            FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            JOIN pg_attribute a ON a.attrelid = c.oid
            WHERE c.relname = 'chunks' AND n.nspname = 'public' AND a.attname = 'embedding'
            """
        )
    ).first()
    if row is None:
        return None
    match = re.search(r"vector\((\d+)\)", row[0])
    return int(match.group(1)) if match else None


def init_schema(engine: Engine, dimensions: int) -> None:
    """Enable pgvector and create the ``chunks`` table + HNSW cosine index.

    If ``chunks`` already exists with a different embedding dimension, the table
    is dropped and recreated so the schema matches the configured embedder.
    """
    with engine.begin() as conn:
        conn.execute(text(_CREATE_EXTENSION_SQL))
        existing = _column_dimension(conn)
        if existing is not None and existing != int(dimensions):
            logger.warning(
                "chunks.embedding is vector(%d) but configured dimension is %d — "
                "dropping and recreating the table",
                existing,
                dimensions,
            )
            conn.execute(text("DROP TABLE IF EXISTS chunks;"))
        conn.execute(text(_CREATE_TABLE_SQL.format(dim=int(dimensions))))
        conn.execute(text(_CREATE_INDEX_SQL))
    logger.info("pgvector schema ready (embedding vector(%d))", dimensions)


def _vector_literal(embedding: list[float]) -> str:
    """Render an embedding list in pgvector literal form, e.g. ``[0.1,0.2]``."""
    return "[" + ",".join(str(x) for x in embedding) + "]"


def _row(embedded: ChunkEmbedding) -> dict[str, Any]:
    metadata = embedded.chunk.metadata
    return {
        "content": embedded.chunk.content,
        "embedding": _vector_literal(embedded.embedding),
        "ticker": metadata.get("ticker"),
        "filing_type": metadata.get("filing_type"),
        "fiscal_period": metadata.get("fiscal_period"),
        "section": metadata.get("section"),
        "source_doc": metadata.get("source_file"),
        "chunk_index": metadata.get("chunk_index"),
        "char_offset": metadata.get("char_offset"),
        "metadata": json.dumps(metadata),
    }


def insert_chunks(engine: Engine, embedded_chunks: list[ChunkEmbedding]) -> int:
    """Bulk-insert embedded chunks; returns the number of rows inserted."""
    if not embedded_chunks:
        return 0
    rows = [_row(embedded) for embedded in embedded_chunks]
    with engine.begin() as conn:
        conn.execute(_INSERT_CHUNK_SQL, rows)
    logger.info("Inserted %d chunks into pgvector", len(rows))
    return len(rows)


def similarity_search(engine: Engine, query_embedding: list[float], k: int) -> list[dict[str, Any]]:
    """Return the top-``k`` chunks ranked by cosine similarity (``<=>``)."""
    with engine.connect() as conn:
        rows = conn.execute(
            _SIMILARITY_SQL,
            {"query": _vector_literal(query_embedding), "k": k},
        )
        return [dict(row._mapping) for row in rows]
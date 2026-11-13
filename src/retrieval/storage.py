"""Vector storage."""

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
    cik BIGINT,
    filing_type TEXT,
    form_type TEXT,
    fiscal_period TEXT,
    fiscal_year INT,
    section TEXT,
    chunk_type TEXT NOT NULL DEFAULT 'text',
    accession TEXT,
    filed_at TEXT,
    source_doc TEXT,
    chunk_index INT,
    char_offset INT,
    embedding_model TEXT,
    embedding_version TEXT,
    metadata JSONB
);
"""

_ADD_COLUMNS_SQL = (
    "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS cik BIGINT",
    "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS form_type TEXT",
    "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS fiscal_year INT",
    "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS chunk_type TEXT",
    "UPDATE chunks SET chunk_type = COALESCE(metadata->>'chunk_type', 'text') WHERE chunk_type IS NULL",
    "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS accession TEXT",
    "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS filed_at TEXT",
    "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS embedding_model TEXT",
    "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS embedding_version TEXT",
)

_CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
"""

_CREATE_METADATA_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS chunks_meta_idx
    ON chunks (ticker, fiscal_year, form_type);
"""

_CREATE_ACCESSION_INDEX_SQL = """
CREATE UNIQUE INDEX IF NOT EXISTS chunks_accession_chunk_idx
    ON chunks (accession, chunk_index);
"""

_INSERT_CHUNK_SQL = text("""
    INSERT INTO chunks
        (content, embedding, ticker, cik, filing_type, form_type, fiscal_period,
         fiscal_year, section, chunk_type, accession, filed_at, source_doc,
         chunk_index, char_offset, embedding_model, embedding_version, metadata)
    VALUES
        (:content, CAST(:embedding AS vector), :ticker, :cik, :filing_type,
         :form_type, :fiscal_period, :fiscal_year, :section, :chunk_type,
         :accession, :filed_at, :source_doc, :chunk_index, :char_offset,
         :embedding_model, :embedding_version, CAST(:metadata AS jsonb))
    ON CONFLICT (accession, chunk_index) DO UPDATE SET
        content = EXCLUDED.content,
        embedding = EXCLUDED.embedding,
        ticker = EXCLUDED.ticker,
        cik = EXCLUDED.cik,
        filing_type = EXCLUDED.filing_type,
        form_type = EXCLUDED.form_type,
        fiscal_period = EXCLUDED.fiscal_period,
        fiscal_year = EXCLUDED.fiscal_year,
        section = EXCLUDED.section,
        chunk_type = EXCLUDED.chunk_type,
        filed_at = EXCLUDED.filed_at,
        source_doc = EXCLUDED.source_doc,
        char_offset = EXCLUDED.char_offset,
        embedding_model = EXCLUDED.embedding_model,
        embedding_version = EXCLUDED.embedding_version,
        metadata = EXCLUDED.metadata
""")

_SIMILARITY_SELECT = """
    SELECT content, ticker, cik, filing_type, form_type, fiscal_period, fiscal_year,
           section, accession, filed_at, source_doc, chunk_index, char_offset,
           embedding_model, embedding_version,
           COALESCE(chunk_type, metadata->>'chunk_type', 'text') AS chunk_type,
           metadata->>'unit_scale' AS unit_scale,
           1 - (embedding <=> CAST(:query AS vector)) AS score
    FROM chunks
"""

_FILTER_COLUMNS: dict[str, str] = {
    "ticker": "ticker",
    "cik": "cik",
    "form_type": "COALESCE(form_type, filing_type)",
    "filing_type": "filing_type",
    "fiscal_year": "fiscal_year",
    "fiscal_period": "fiscal_period",
    "section": "section",
    "chunk_type": "COALESCE(chunk_type, metadata->>'chunk_type', 'text')",
    "accession": "accession",
}


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
                "chunks.embedding is vector(%d) but configured dimension is %d — dropping and recreating the table",
                existing,
                dimensions,
            )
            conn.execute(text("DROP TABLE IF EXISTS chunks;"))
        conn.execute(text(_CREATE_TABLE_SQL.format(dim=int(dimensions))))
        for statement in _ADD_COLUMNS_SQL:
            conn.execute(text(statement))
        conn.execute(text(_CREATE_INDEX_SQL))
        conn.execute(text(_CREATE_METADATA_INDEX_SQL))
        conn.execute(text(_CREATE_ACCESSION_INDEX_SQL))
    logger.info("pgvector schema ready (embedding vector(%d))", dimensions)


def _vector_literal(embedding: list[float]) -> str:
    """Render an embedding list in pgvector literal form, e.g. ``[0.1,0.2]``."""
    return "[" + ",".join(str(x) for x in embedding) + "]"


def _row(
    embedded: ChunkEmbedding,
    *,
    embedding_model: str | None = None,
    embedding_version: str | None = None,
) -> dict[str, Any]:
    metadata = embedded.chunk.metadata
    return {
        "content": embedded.chunk.content,
        "embedding": _vector_literal(embedded.embedding),
        "ticker": metadata.get("ticker"),
        "cik": metadata.get("cik"),
        "filing_type": metadata.get("filing_type"),
        "form_type": metadata.get("form_type") or metadata.get("filing_type"),
        "fiscal_period": metadata.get("fiscal_period"),
        "fiscal_year": metadata.get("fiscal_year"),
        "section": metadata.get("section"),
        "chunk_type": metadata.get("chunk_type", "text"),
        "accession": metadata.get("accession"),
        "filed_at": metadata.get("filed_at"),
        "source_doc": metadata.get("source_file"),
        "chunk_index": metadata.get("chunk_index"),
        "char_offset": metadata.get("char_offset"),
        "embedding_model": embedding_model or metadata.get("embedding_model"),
        "embedding_version": embedding_version or metadata.get("embedding_version"),
        "metadata": json.dumps(metadata),
    }


def insert_chunks(
    engine: Engine,
    embedded_chunks: list[ChunkEmbedding],
    *,
    embedding_model: str | None = None,
    embedding_version: str | None = None,
) -> int:
    """Upsert embedded chunks keyed by ``(accession, chunk_index)``; returns row count."""
    if not embedded_chunks:
        return 0
    rows = [
        _row(embedded, embedding_model=embedding_model, embedding_version=embedding_version)
        for embedded in embedded_chunks
    ]
    with engine.begin() as conn:
        conn.execute(_INSERT_CHUNK_SQL, rows)
    logger.info("Upserted %d chunks into pgvector", len(rows))
    return len(rows)


def _filter_clause(filters: dict[str, Any] | None, params: dict[str, Any]) -> str:
    """Build a whitelisted SQL WHERE clause from retrieval filters."""
    if not filters:
        return ""
    clauses: list[str] = []
    for name, value in filters.items():
        column = _FILTER_COLUMNS.get(name)
        if column is None or value is None or value == "":
            continue
        if isinstance(value, str):
            value = value.strip()
            if name == "ticker":
                value = value.lower()
            elif name in {"form_type", "filing_type"}:
                value = value.upper()
        key = f"filter_{name}"
        clauses.append(f"{column} = :{key}")
        params[key] = value
    return (" WHERE " + " AND ".join(clauses)) if clauses else ""


def similarity_search(
    engine: Engine,
    query_embedding: list[float],
    k: int,
    *,
    filters: dict[str, Any] | None = None,
    embedding_model: str | None = None,
) -> list[dict[str, Any]]:
    """Return the top-``k`` chunks ranked by cosine similarity (``<=>``).

    Metadata filters are applied as SQL pre-filters. Passing ``embedding_model``
    excludes rows embedded with a different model, preventing mixed vector spaces.
    """
    params: dict[str, Any] = {"query": _vector_literal(query_embedding), "k": k}
    where = _filter_clause(filters, params)
    if embedding_model:
        where = (
            f"{where} AND embedding_model = :embedding_model" if where else " WHERE embedding_model = :embedding_model"
        )
        params["embedding_model"] = embedding_model
    sql = text(_SIMILARITY_SELECT + where + " ORDER BY embedding <=> CAST(:query AS vector) LIMIT :k")
    with engine.connect() as conn:
        rows = conn.execute(sql, params)
        return [dict(row._mapping) for row in rows]

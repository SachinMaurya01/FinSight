"""Phase 3 retrieval unit tests.

DB-backed functions (``init_schema``, ``insert_chunks``, ``similarity_search``)
require a running Postgres + pgvector and credentials; a live integration test
is deferred (TODO) until a DB is configured locally — see AGENTS.md §4.4.
Everything else is tested with synthetic fixtures (AGENTS.md §5.2).

Run with: python tests/test_retrieval.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Settings
from src.ingestion.chunker import Chunk
from src.retrieval.embeddings import embed_chunks
from src.retrieval.storage import (
    _CREATE_INDEX_SQL,
    _CREATE_TABLE_SQL,
    _row,
    _vector_literal,
    get_engine,
)

TEST_SETTINGS = Settings(
    embedding_batch_size=2,
    embedding_dimensions=3,
    postgres_database_url="postgresql://localhost:5432/finsight_test",
)


class FakeEmbedder:
    """Deterministic embedder: vector derived from the text length."""

    def __init__(self, dim: int = 3) -> None:
        self.dim = dim
        self.calls: list[int] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(len(texts))
        return [[float(len(t))] * self.dim for t in texts]


def make_chunks(n: int) -> list[Chunk]:
    return [
        Chunk(content=f"synthetic chunk number {i}", metadata={"chunk_index": i})
        for i in range(n)
    ]


def test_embed_chunks_batching() -> None:
    fake = FakeEmbedder()
    chunks = make_chunks(5)
    embedded = embed_chunks(chunks, TEST_SETTINGS, fake)
    assert fake.calls == [2, 2, 1], fake.calls
    assert len(embedded) == 5
    for ce, chunk in zip(embedded, chunks):
        assert ce.chunk is chunk
        assert ce.embedding == [float(len(chunk.content))] * 3
    print("PASS test_embed_chunks_batching")


def test_embed_chunks_skips_empty() -> None:
    fake = FakeEmbedder()
    chunks = [Chunk(content="", metadata={"chunk_index": 0}), *make_chunks(2)]
    embedded = embed_chunks(chunks, TEST_SETTINGS, fake)
    assert len(embedded) == 2
    assert fake.calls == [2]
    print("PASS test_embed_chunks_skips_empty")


def test_embed_chunks_dim_mismatch_warns() -> None:
    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append  # type: ignore[method-assign]
    logger = logging.getLogger("src.retrieval.embeddings")
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    try:
        fake = FakeEmbedder(dim=5)
        embed_chunks(make_chunks(1), TEST_SETTINGS, fake)
    finally:
        logger.removeHandler(handler)
    assert any(r.levelno == logging.WARNING and "dimension" in r.getMessage() for r in records)
    print("PASS test_embed_chunks_dim_mismatch_warns")


def test_vector_literal() -> None:
    assert _vector_literal([0.1, 0.2]) == "[0.1,0.2]"
    print("PASS test_vector_literal")


def test_row_maps_metadata() -> None:
    chunk = Chunk(
        content="synthetic",
        metadata={
            "ticker": "SYN",
            "filing_type": "10-K",
            "fiscal_period": "FY2099",
            "section": "Item 1A Risk Factors",
            "source_file": "synth-10K.html",
            "chunk_index": 7,
            "char_offset": 123,
        },
    )
    from src.retrieval.embeddings import ChunkEmbedding

    row = _row(ChunkEmbedding(chunk=chunk, embedding=[0.1, 0.2, 0.3]))
    assert row["content"] == "synthetic"
    assert row["embedding"] == "[0.1,0.2,0.3]"
    assert row["source_doc"] == "synth-10K.html"
    assert row["ticker"] == "SYN" and row["chunk_index"] == 7 and row["char_offset"] == 123
    import json

    assert json.loads(row["metadata"])["fiscal_period"] == "FY2099"
    print("PASS test_row_maps_metadata")


def test_schema_sql_generation() -> None:
    sql = _CREATE_TABLE_SQL.format(dim=3)
    assert "vector(3)" in sql
    assert "embedding vector_cosine_ops" in _CREATE_INDEX_SQL
    assert "USING hnsw" in _CREATE_INDEX_SQL
    print("PASS test_schema_sql_generation")


def test_get_engine() -> None:
    engine = get_engine(TEST_SETTINGS)
    assert engine.dialect.name == "postgresql"
    print("PASS test_get_engine")


def test_invalid_embedder_provider() -> None:
    from pydantic import ValidationError

    try:
        Settings(embedding_provider="bogus")
    except ValidationError:
        print("PASS test_invalid_embedder_provider")
    else:
        raise AssertionError("expected ValidationError")


def main() -> int:
    test_embed_chunks_batching()
    test_embed_chunks_skips_empty()
    test_embed_chunks_dim_mismatch_warns()
    test_vector_literal()
    test_row_maps_metadata()
    test_schema_sql_generation()
    test_get_engine()
    test_invalid_embedder_provider()
    return 0


if __name__ == "__main__":
    sys.exit(main())
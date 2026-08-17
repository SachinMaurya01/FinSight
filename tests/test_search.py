"""Phase 3 search API unit tests.

``seed_database`` and ``retrieve_dense`` orchestration is tested with
monkeypatched fake components (no network / no DB). A live DB integration test
is deferred (TODO) until Postgres + pgvector credentials are configured.

Run with: python tests/test_search.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Settings
from src.retrieval import search as search_mod

TEST_SETTINGS = Settings(
    embedding_batch_size=64,
    embedding_dimensions=3,
    rerank_top_k=5,
    postgres_database_url="postgresql://localhost:5432/finsight_test",
)


def _write_chunk_file(directory: Path, name: str, n: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_file": name,
        "chunks": [
            {
                "content": f"synthetic chunk content {i}",
                "metadata": {"ticker": "SYN", "chunk_index": i, "section": "Item 1A Risk Factors"},
            }
            for i in range(n)
        ],
    }
    (directory / name).write_text(json.dumps(payload), encoding="utf-8")


def test_load_chunk_files() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        _write_chunk_file(directory, "synth-10K_chunks.json", 3)
        _write_chunk_file(directory, "synth-8K_chunks.json", 2)
        chunks = search_mod.load_chunk_files(directory)
        assert len(chunks) == 5
        assert chunks[0].metadata["ticker"] == "SYN"
        assert all(chunk.content for chunk in chunks)
    print("PASS test_load_chunk_files")


def test_seed_database_orchestration() -> None:
    calls: list[tuple[str, int]] = []

    def fake_embed_chunks(chunks, settings, embedder=None):
        calls.append(("embed", len(chunks)))
        return [(chunk, [0.1, 0.2, 0.3]) for chunk in chunks]

    def fake_insert_chunks(engine, embedded):
        calls.append(("insert", len(embedded)))
        return len(embedded)

    original_embed, original_insert = search_mod.embed_chunks, search_mod.insert_chunks
    search_mod.embed_chunks = fake_embed_chunks
    search_mod.insert_chunks = fake_insert_chunks
    try:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            _write_chunk_file(directory, "synth_chunks.json", 4)
            count = search_mod.seed_database(TEST_SETTINGS, engine=object(), chunk_dir=directory)
            assert count == 4
            assert calls == [("embed", 4), ("insert", 4)], calls
    finally:
        search_mod.embed_chunks = original_embed
        search_mod.insert_chunks = original_insert
    print("PASS test_seed_database_orchestration")


def test_retrieve_dense_orchestration() -> None:
    calls: dict[str, object] = {}

    class FakeEmbedder:
        def embed_texts(self, texts):
            calls["query"] = texts[0]
            return [[1.0, 0.0, 0.0]]

    def fake_similarity_search(engine, query_embedding, k):
        calls["embedding"] = query_embedding
        calls["k"] = k
        return [{"content": "synthetic result", "score": 0.99}]

    original_embedder, original_similarity = search_mod.build_embedder, search_mod.similarity_search
    search_mod.build_embedder = lambda settings: FakeEmbedder()
    search_mod.similarity_search = fake_similarity_search
    try:
        results = search_mod.retrieve_dense("what are the main risks", TEST_SETTINGS)
        assert calls["query"] == "what are the main risks"
        assert calls["embedding"] == [1.0, 0.0, 0.0]
        assert calls["k"] == 5, calls["k"]
        assert results == [{"content": "synthetic result", "score": 0.99}]
    finally:
        search_mod.build_embedder = original_embedder
        search_mod.similarity_search = original_similarity
    print("PASS test_retrieve_dense_orchestration")


def main() -> int:
    test_load_chunk_files()
    test_seed_database_orchestration()
    test_retrieve_dense_orchestration()
    return 0


if __name__ == "__main__":
    sys.exit(main())
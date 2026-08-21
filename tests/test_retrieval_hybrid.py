"""Phase 6 — BM25 index + hybrid fusion unit tests (synthetic chunks, no DB).

Run with: python tests/test_retrieval_hybrid.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Settings
from src.ingestion.chunker import Chunk
from src.retrieval.bm25 import BM25Index, build_bm25_index, tokenize
from src.retrieval.hybrid import fuse_hybrid, retrieve_hybrid

TEST_SETTINGS = Settings(
    embedding_batch_size=64,
    embedding_dimensions=3,
    rerank_top_k=5,
    postgres_database_url="postgresql://localhost:5432/finsight_test",
)

SYNTH_CHUNKS = [
    Chunk(
        content="Apple's total net sales were 383.3 billion dollars in fiscal 2025.",
        metadata={"ticker": "aapl", "filing_type": "10-K", "fiscal_period": "FY2025",
                  "section": "Item 7 MD&A", "source_file": "aapl-10K.html", "chunk_index": 0},
    ),
    Chunk(
        content="The company generates most of its revenue from the iPhone product line.",
        metadata={"ticker": "aapl", "filing_type": "10-K", "fiscal_period": "FY2025",
                  "section": "Item 7 MD&A", "source_file": "aapl-10K.html", "chunk_index": 1},
    ),
    Chunk(
        content="Risk factors include competition and cybersecurity threats.",
        metadata={"ticker": "aapl", "filing_type": "10-K", "fiscal_period": "FY2025",
                  "section": "Item 1A Risk Factors", "source_file": "aapl-10K.html", "chunk_index": 2},
    ),
]


def _write_chunk_files(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    payload = {"source_file": "aapl-10K.html", "chunks": []}
    for chunk in SYNTH_CHUNKS:
        payload["chunks"].append({"content": chunk.content, "metadata": chunk.metadata})
    (directory / "aapl-10K_chunks.json").write_text(json.dumps(payload), encoding="utf-8")


def test_tokenize() -> None:
    tokens = tokenize("Total Net Sales: $383.3 billion.")
    assert "$383.3" in tokens and "billion" in tokens and "383" not in tokens
    print("PASS test_tokenize")


def test_bm25_exact_figure_ranked_first() -> None:
    index = BM25Index(SYNTH_CHUNKS)
    rows = index.search("383.3 billion net sales", k=3)
    assert rows[0]["chunk_index"] == 0, rows
    assert rows[0]["score"] > 0
    print("PASS test_bm25_exact_figure_ranked_first")


def test_build_bm25_index_cached() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        _write_chunk_files(Path(tmp))
        first = build_bm25_index(Path(tmp))
        second = build_bm25_index(Path(tmp))
        assert first is second
        assert len(first) == 3
    print("PASS test_build_bm25_index_cached")


def test_fuse_hybrid_combines_and_ranks() -> None:
    dense = [
        {"content": "risk factors chunk", "source_doc": "aapl-10K.html", "chunk_index": 2, "score": 0.9},
        {"content": "net sales chunk", "source_doc": "aapl-10K.html", "chunk_index": 0, "score": 0.5},
    ]
    bm25 = [
        {"content": "net sales chunk", "source_doc": "aapl-10K.html", "chunk_index": 0, "score": 12.0},
        {"content": "iphone chunk", "source_doc": "aapl-10K.html", "chunk_index": 1, "score": 9.0},
    ]
    fused = fuse_hybrid(dense, bm25, TEST_SETTINGS)
    assert len(fused) == 3
    by_index = {row["chunk_index"]: row for row in fused}
    assert "dense_score" in by_index[0] and "lexical_score" in by_index[0]
    # chunk 0 appears in both lists -> highest fused score
    assert by_index[0]["score"] > by_index[2]["score"]
    assert by_index[0]["score"] > by_index[1]["score"]
    assert fused[0]["chunk_index"] == 0
    print("PASS test_fuse_hybrid_combines_and_ranks")


def test_retrieve_hybrid_orchestration() -> None:
    calls: list[str] = []

    def fake_retrieve_dense(query, settings, k=None):
        calls.append("dense")
        return [{"content": "net sales chunk", "source_doc": "aapl-10K.html",
                 "chunk_index": 0, "score": 0.5}]

    def fake_build_index(directory):
        calls.append("bm25")

        class _Index:
            def search(self, query, k):
                return [{"content": "risk chunk", "source_doc": "aapl-10K.html",
                         "chunk_index": 2, "score": 5.0}]

        return _Index()

    original_dense, original_build = retrieve_hybrid.__globals__["retrieve_dense"], retrieve_hybrid.__globals__["build_bm25_index"]
    retrieve_hybrid.__globals__["retrieve_dense"] = fake_retrieve_dense
    retrieve_hybrid.__globals__["build_bm25_index"] = fake_build_index
    try:
        rows = retrieve_hybrid("383.3 billion", TEST_SETTINGS, k=2)
    finally:
        retrieve_hybrid.__globals__["retrieve_dense"] = original_dense
        retrieve_hybrid.__globals__["build_bm25_index"] = original_build

    assert calls == ["dense", "bm25"], calls
    assert len(rows) == 2
    print("PASS test_retrieve_hybrid_orchestration")


def main() -> int:
    test_tokenize()
    test_bm25_exact_figure_ranked_first()
    test_build_bm25_index_cached()
    test_fuse_hybrid_combines_and_ranks()
    test_retrieve_hybrid_orchestration()
    return 0


if __name__ == "__main__":
    sys.exit(main())
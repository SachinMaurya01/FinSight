"""Phase 7 — reranking unit tests (fake reranker, no model download).

Run with: python tests/test_rerank.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Settings
from src.retrieval import rerank as rerank_mod

TEST_SETTINGS = Settings(
    reranker_model="fake/cross-encoder",
    reranker_device="cpu",
    fusion_top_k=20,
    rerank_top_k=3,
    postgres_database_url="postgresql://localhost:5432/finsight_test",
)


class _FakeReranker:
    def score(self, query, documents):
        # score by index (reverse order) to prove reordering happens
        return [float(10 - i) for i in range(len(documents))]


def test_rerank_candidates_reorders_and_truncates() -> None:
    rows = [{"content": f"chunk {i}", "chunk_index": i} for i in range(5)]
    original_build = rerank_mod.build_reranker
    rerank_mod.build_reranker = lambda settings: _FakeReranker()
    try:
        out = rerank_mod.rerank_candidates("q", rows, TEST_SETTINGS, top_k=3)
    finally:
        rerank_mod.build_reranker = original_build

    assert [r["chunk_index"] for r in out] == [0, 1, 2]
    assert out[0]["rerank_score"] == 10.0
    print("PASS test_rerank_candidates_reorders_and_truncates")


def test_rerank_skipped_when_disabled() -> None:
    rows = [{"content": "a", "chunk_index": 0}, {"content": "b", "chunk_index": 1}]
    original_build = rerank_mod.build_reranker
    rerank_mod.build_reranker = lambda settings: None
    try:
        out = rerank_mod.rerank_candidates("q", rows, TEST_SETTINGS, top_k=1)
    finally:
        rerank_mod.build_reranker = original_build

    assert len(out) == 1 and out[0]["chunk_index"] == 0
    assert "rerank_score" not in out[0]
    print("PASS test_rerank_skipped_when_disabled")


def test_retrieve_reranked_orchestration() -> None:
    calls: list[str] = []

    def fake_retrieve_hybrid(query, settings, k=None):
        calls.append("hybrid")
        return [{"content": f"chunk {i}", "chunk_index": i} for i in range(4)]

    def fake_rerank(query, rows, settings, top_k=None):
        calls.append("rerank")
        return rows[:top_k]

    original_hybrid, original_rerank = rerank_mod.retrieve_hybrid, rerank_mod.rerank_candidates
    rerank_mod.retrieve_hybrid = fake_retrieve_hybrid
    rerank_mod.rerank_candidates = fake_rerank
    try:
        out = rerank_mod.retrieve_reranked("q", TEST_SETTINGS)
    finally:
        rerank_mod.retrieve_hybrid = original_hybrid
        rerank_mod.rerank_candidates = original_rerank

    assert calls == ["hybrid", "rerank"], calls
    assert len(out) == 3
    print("PASS test_retrieve_reranked_orchestration")


def main() -> int:
    test_rerank_candidates_reorders_and_truncates()
    test_rerank_skipped_when_disabled()
    test_retrieve_reranked_orchestration()
    return 0


if __name__ == "__main__":
    sys.exit(main())
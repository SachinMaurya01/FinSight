"""Phase 8 — context compression unit tests (fake embedder, no network).

Run with: python tests/test_compress.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Settings
from src.retrieval.compress import compress_chunks, split_sentences

TEST_SETTINGS = Settings(
    embedding_batch_size=64,
    embedding_dimensions=2,
    context_token_budget=60,
    postgres_database_url="postgresql://localhost:5432/finsight_test",
)


class _FakeEmbedder:
    """Sentence embeddings: high similarity to query when 'risk' is present, else zero."""

    def embed_texts(self, texts):
        return [
            [1.0, 0.0] if "risk" in t else [0.0, 0.0]
            for t in texts
        ]


def test_split_sentences() -> None:
    text = "Apple faces risk. The company is growing fast! Is it safe?"
    assert split_sentences(text) == [
        "Apple faces risk.",
        "The company is growing fast!",
        "Is it safe?",
    ]
    print("PASS test_split_sentences")


def test_compress_keeps_relevant_sentences() -> None:
    rows = [
        {
            "content": "Apple faces supply chain risk in China. The company sells iPhones. "
                       "Risk factors include competition. Green widgets are popular.",
            "chunk_index": 0,
        }
    ]
    out = compress_chunks("what are the risks", rows, TEST_SETTINGS, embedder=_FakeEmbedder())
    kept = out[0]["content"]
    assert "risk" in kept
    assert out[0]["compressed"] is True
    # irrelevant sentences ("Green widgets") should be dropped
    assert "Green widgets" not in kept
    print("PASS test_compress_keeps_relevant_sentences")


def test_compress_enforces_token_budget() -> None:
    rows = [
        {"content": " ".join([f"risk sentence number {i} here" for i in range(20)]), "chunk_index": 0},
        {"content": " ".join([f"risk sentence number {i} again" for i in range(20)]), "chunk_index": 1},
    ]
    out = compress_chunks("risk", rows, TEST_SETTINGS, embedder=_FakeEmbedder())
    total = sum(len(c["content"].split()) for c in out)
    assert total <= 80  # rough token estimate ~ words; budget=60 tokens
    assert len(out) >= 1
    print("PASS test_compress_enforces_token_budget")


def main() -> int:
    test_split_sentences()
    test_compress_keeps_relevant_sentences()
    test_compress_enforces_token_budget()
    return 0


if __name__ == "__main__":
    sys.exit(main())
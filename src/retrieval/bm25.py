"""BM25 lexical retrieval."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from src.ingestion.chunker import Chunk
from src.retrieval.search import load_chunk_files

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"\$?\d+(?:[.,]\d+)*%?|[A-Za-z]+")


def tokenize(text: str) -> list[str]:
    """Tokenize text for BM25: words plus numeric tokens (e.g. ``$383.3``, ``15%``)."""
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """BM25Okapi index over a set of chunks with a retrieval-friendly row format."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        self._bm25 = BM25Okapi([tokenize(chunk.content) for chunk in chunks])

    def __len__(self) -> int:
        return len(self._chunks)

    def search(self, query: str, k: int) -> list[dict]:
        """Return the top-``k`` chunks scored by BM25, dense-row compatible."""
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(self._chunks)), key=lambda i: scores[i], reverse=True)
        rows: list[dict] = []
        for i in ranked:
            if scores[i] <= 0.0 or len(rows) >= k:
                break
            meta = self._chunks[i].metadata
            rows.append(
                {
                    "content": self._chunks[i].content,
                    "ticker": meta.get("ticker"),
                    "filing_type": meta.get("filing_type"),
                    "fiscal_period": meta.get("fiscal_period"),
                    "section": meta.get("section"),
                    "source_doc": meta.get("source_file"),
                    "chunk_index": meta.get("chunk_index"),
                    "score": float(scores[i]),
                }
            )
        return rows


_BUILT_INDEX: tuple[Path, BM25Index] | None = None


def build_bm25_index(chunk_dir: Path) -> BM25Index:
    """Build (or return the cached) BM25 index for ``chunk_dir``."""
    global _BUILT_INDEX
    if _BUILT_INDEX is not None and _BUILT_INDEX[0] == chunk_dir:
        return _BUILT_INDEX[1]
    chunks = load_chunk_files(chunk_dir)
    index = BM25Index(chunks)
    logger.info("Built BM25 index over %d chunks", len(index))
    _BUILT_INDEX = (chunk_dir, index)
    return index
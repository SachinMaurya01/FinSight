"""BM25 lexical retrieval."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

# use bm25s for highly optimized
from src.ingestion.chunker import Chunk
from src.retrieval.search import load_chunk_files

logger = logging.getLogger(__name__)

# Tokenizes words plus financial tokens like $383.3 or 15% for BM25 lexical search.
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

    def search(self, query: str, k: int, filters: dict | None = None) -> list[dict]:
        """Return the top-``k`` chunks scored by BM25, dense-row compatible.

        Filters are applied as a pre-filter over the ranked corpus. The BM25
        scores are still computed over the full corpus, so IDF remains stable.
        """
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(self._chunks)), key=lambda i: scores[i], reverse=True)
        rows: list[dict] = []
        for i in ranked:
            if scores[i] <= 0.0 or len(rows) >= k:
                break
            meta = self._chunks[i].metadata
            if filters and not _matches_filters(meta, filters):
                continue
            rows.append(
                {
                    "content": self._chunks[i].content,
                    "ticker": meta.get("ticker"),
                    "cik": meta.get("cik"),
                    "filing_type": meta.get("filing_type"),
                    "form_type": meta.get("form_type") or meta.get("filing_type"),
                    "fiscal_period": meta.get("fiscal_period"),
                    "fiscal_year": meta.get("fiscal_year"),
                    "section": meta.get("section"),
                    "accession": meta.get("accession"),
                    "filed_at": meta.get("filed_at"),
                    "source_doc": meta.get("source_file"),
                    "chunk_index": meta.get("chunk_index"),
                    "chunk_type": meta.get("chunk_type", "text"),
                    "unit_scale": meta.get("unit_scale"),
                    "score": float(scores[i]),
                }
            )
        return rows


def _matches_filters(metadata: dict, filters: dict) -> bool:
    """Return whether chunk metadata satisfies all retrieval filters."""
    for name, expected in filters.items():
        if expected is None or expected == "":
            continue
        if name == "form_type":
            actual = metadata.get("form_type") or metadata.get("filing_type")
        elif name == "chunk_type":
            actual = metadata.get("chunk_type", "text")
        else:
            actual = metadata.get(name)
        if actual is None:
            return False
        if isinstance(expected, str) and isinstance(actual, str):
            if actual.lower() != expected.lower():
                return False
        elif actual != expected:
            return False
    return True


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

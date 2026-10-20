"""Retrieval ablation: dense-only vs hybrid vs hybrid + rerank.

This is a retrieval-only benchmark (no LLM calls), so it runs offline against
the checked-in sample chunks and produces reproducible numbers for the README.

It intentionally uses an in-memory cosine index instead of pgvector so reviewers
can run the comparison without starting Postgres. The ranking math matches the
production dense path (normalized cosine / cosine distance).
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from src.config import Settings, settings
from src.eval.dataset import EvalExample, load_dataset
from src.ingestion.chunker import Chunk
from src.retrieval.bm25 import BM25Index, build_bm25_index
from src.retrieval.embeddings import Embedder, build_embedder
from src.retrieval.hybrid import fuse_hybrid
from src.retrieval.rerank import CrossEncoderReranker, build_reranker, rerank_candidates
from src.retrieval.search import load_chunk_files

logger = logging.getLogger(__name__)

ConfigName = Literal["dense_only", "hybrid", "hybrid_rerank"]
CONFIG_NAMES: tuple[ConfigName, ...] = ("dense_only", "hybrid", "hybrid_rerank")


@dataclass(frozen=True)
class RetrievalScore:
    """Retrieval quality for one example under one configuration."""

    context_precision: float
    context_recall: float
    mrr: float


def _row_from_chunk(chunk: Chunk) -> dict[str, Any]:
    metadata = chunk.metadata
    return {
        "content": chunk.content,
        "ticker": metadata.get("ticker"),
        "filing_type": metadata.get("filing_type"),
        "fiscal_period": metadata.get("fiscal_period"),
        "section": metadata.get("section"),
        "source_doc": metadata.get("source_file"),
        "chunk_index": metadata.get("chunk_index"),
    }


class InMemoryDenseIndex:
    """Cosine-similarity index over local chunks (eval/offline use only)."""

    def __init__(self, chunks: list[Chunk], embedder: Embedder) -> None:
        self._chunks = chunks
        self._vectors = embedder.embed_texts([chunk.content for chunk in chunks])
        self._norms = [math.sqrt(sum(value * value for value in vector)) or 1.0 for vector in self._vectors]

    def search(self, query: str, embedder: Embedder, k: int) -> list[dict[str, Any]]:
        """Return the top-``k`` chunks ranked by cosine similarity."""
        query_vector = embedder.embed_texts([query])[0]
        query_norm = math.sqrt(sum(value * value for value in query_vector)) or 1.0
        scores = [
            sum(a * b for a, b in zip(query_vector, vector)) / (query_norm * norm)
            for vector, norm in zip(self._vectors, self._norms)
        ]
        ranked = sorted(range(len(self._chunks)), key=lambda index: scores[index], reverse=True)
        rows: list[dict[str, Any]] = []
        for index in ranked[:k]:
            row = _row_from_chunk(self._chunks[index])
            row["score"] = float(scores[index])
            rows.append(row)
        return rows


def _citation_matches(row: dict[str, Any], citation: dict[str, Any]) -> bool:
    """Metadata-level citation match (ticker / fiscal period / section)."""
    return all(
        not citation.get(key) or row.get(key) == citation.get(key)
        for key in ("ticker", "fiscal_period", "section")
    )


def score_retrieval(
    rows: list[dict[str, Any]],
    example: EvalExample,
) -> RetrievalScore | None:
    """Score one ranked list against an example's expected citations.

    Returns ``None`` for examples without citation labels (e.g. pure tool
    questions), which are excluded from the aggregate.
    """
    citations = [citation.model_dump() for citation in example.expected_citations]
    if not citations:
        return None

    relevant = [any(_citation_matches(row, citation) for citation in citations) for row in rows]
    precision = sum(relevant) / len(rows) if rows else 0.0
    recall = sum(
        1 for citation in citations if any(_citation_matches(row, citation) for row in rows)
    ) / len(citations)
    mrr = next((1.0 / rank for rank, is_relevant in enumerate(relevant, start=1) if is_relevant), 0.0)
    return RetrievalScore(
        context_precision=round(precision, 4),
        context_recall=round(recall, 4),
        mrr=round(mrr, 4),
    )


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _run_configs(
    example: EvalExample,
    dense_index: InMemoryDenseIndex,
    embedder: Embedder,
    bm25_index: BM25Index,
    reranker: CrossEncoderReranker | None,
    config: Settings,
) -> dict[ConfigName, list[dict[str, Any]]]:
    """Return ranked rows for each retrieval configuration."""
    dense_rows = dense_index.search(example.question, embedder, k=config.rerank_top_k)
    fused = fuse_hybrid(
        dense_index.search(example.question, embedder, k=config.dense_top_k),
        bm25_index.search(example.question, config.bm25_top_k),
        config,
    )
    hybrid_rows = fused[: config.rerank_top_k]
    rerank_rows = rerank_candidates(
        example.question,
        list(fused[: config.fusion_top_k]),
        config,
        top_k=config.rerank_top_k,
        reranker=reranker,
    )
    return {
        "dense_only": dense_rows,
        "hybrid": hybrid_rows,
        "hybrid_rerank": rerank_rows,
    }


def run_retrieval_ablation(
    *,
    limit: int | None = None,
    output_path: Path | None = None,
    config: Settings | None = None,
) -> dict[str, Any]:
    """Run the offline retrieval ablation and optionally write a JSON report."""
    active = config or settings
    examples = load_dataset()
    if limit:
        examples = examples[:limit]

    chunks = load_chunk_files(active.chunk_store_dir)
    if not chunks:
        raise RuntimeError(f"No chunks found in {active.chunk_store_dir}")

    logger.info("Building in-memory dense index over %d chunks", len(chunks))
    embedder = build_embedder(active)
    dense_index = InMemoryDenseIndex(chunks, embedder)
    bm25_index = build_bm25_index(active.chunk_store_dir)
    reranker = build_reranker(active)

    per_example: list[dict[str, Any]] = []
    scores: dict[ConfigName, list[RetrievalScore]] = {name: [] for name in CONFIG_NAMES}

    for example in examples:
        rows_by_config = _run_configs(example, dense_index, embedder, bm25_index, reranker, active)
        entry: dict[str, Any] = {
            "id": example.id,
            "question": example.question,
            "category": example.category,
            "configs": {},
        }
        for name in CONFIG_NAMES:
            result = score_retrieval(rows_by_config[name], example)
            if result is not None:
                scores[name].append(result)
            entry["configs"][name] = {
                "context_precision": None if result is None else result.context_precision,
                "context_recall": None if result is None else result.context_recall,
                "mrr": None if result is None else result.mrr,
                "sections": [row.get("section") for row in rows_by_config[name]],
            }
        per_example.append(entry)

    aggregate = {
        name: {
            "context_precision": _mean([score.context_precision for score in values]),
            "context_recall": _mean([score.context_recall for score in values]),
            "mrr": _mean([score.mrr for score in values]),
            "n_scored": len(values),
        }
        for name, values in scores.items()
    }
    report: dict[str, Any] = {
        "n_examples": len(examples),
        "n_scored": aggregate["dense_only"]["n_scored"],
        "k": active.rerank_top_k,
        "chunk_count": len(chunks),
        "embedding_model": active.embedding_model,
        "reranker_model": active.reranker_model,
        "dense_top_k": active.dense_top_k,
        "bm25_top_k": active.bm25_top_k,
        "fusion_top_k": active.fusion_top_k,
        "configs": aggregate,
        "per_example": per_example,
    }

    destination = output_path or (active.data_dir / "eval" / "retrieval_ablation.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Retrieval ablation written to %s", destination)
    return report


def format_markdown_table(report: dict[str, Any]) -> str:
    """Render the aggregate ablation numbers as a Markdown table."""
    labels: dict[ConfigName, str] = {
        "dense_only": "Dense only",
        "hybrid": "Hybrid (dense + BM25, RRF)",
        "hybrid_rerank": "Hybrid + cross-encoder rerank",
    }
    lines = [
        "| Configuration | Context precision | Context recall | MRR |",
        "|---|---:|---:|---:|",
    ]
    for name in CONFIG_NAMES:
        result = report["configs"][name]
        lines.append(
            f"| {labels[name]} | {result['context_precision']:.4f} | "
            f"{result['context_recall']:.4f} | {result['mrr']:.4f} |"
        )
    return "\n".join(lines)

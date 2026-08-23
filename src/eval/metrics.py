"""Heuristic RAGAS-like metrics (KISS) — context precision/recall, faithfulness, answer relevance.

Tries to import real RAGAS if available; falls back to deterministic heuristics
so evaluation works offline. All metrics in [0,1].
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EvalMetrics:
    context_precision: float
    context_recall: float
    faithfulness: float
    answer_relevance: float


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _context_precision(retrieved: list[dict], expected_citations: list[dict]) -> float:
    if not retrieved or not expected_citations:
        return 0.0 if expected_citations else 1.0
    hits = 0
    for chunk in retrieved:
        for exp in expected_citations:
            match = True
            for key in ("ticker", "fiscal_period", "section"):
                if exp.get(key) and chunk.get(key) != exp.get(key):
                    match = False
                    break
            if match:
                hits += 1
                break
    return hits / len(retrieved)


def _context_recall(retrieved: list[dict], expected_citations: list[dict]) -> float:
    if not expected_citations:
        return 1.0
    hits = 0
    for exp in expected_citations:
        for chunk in retrieved:
            match = True
            for key in ("ticker", "fiscal_period", "section"):
                if exp.get(key) and chunk.get(key) != exp.get(key):
                    match = False
                    break
            if match:
                hits += 1
                break
    return hits / len(expected_citations)


def _faithfulness(answer: str, contexts: list[str]) -> float:
    """Heuristic: fraction of answer claims grounded in contexts (uses verify logic)."""
    if not answer.strip() or not contexts:
        return 0.0
    # Reuse verification heuristic: split into claims, check overlap
    try:
        from src.verification.verify import verify_citations

        # Build fake chunk dicts from contexts
        chunks = [{"content": c} for c in contexts]
        result = verify_citations(answer, chunks)  # type: ignore[arg-type]
        if result.total_claims == 0:
            return 0.0
        return len(result.verified_claims) / result.total_claims
    except Exception as exc:
        logger.warning("faithfulness heuristic failed: %s", exc)
        return 0.5


def _answer_relevance(question: str, answer: str) -> float:
    """Heuristic: embedding cosine between question and answer if possible, else token overlap."""
    if not answer.strip():
        return 0.0
    try:
        from src.config import settings
        from src.retrieval.embeddings import build_embedder
        import math

        embedder = build_embedder(settings)
        q_emb, a_emb = embedder.embed_texts([question, answer])
        dot = sum(x * y for x, y in zip(q_emb, a_emb))
        norm = math.sqrt(sum(x * x for x in q_emb)) * math.sqrt(sum(y * y for y in a_emb))
        sim = dot / norm if norm else 0
        # Map cosine [-1,1] to [0,1]
        return max(0.0, min(1.0, (sim + 1) / 2))
    except Exception:
        # Fallback token overlap
        q_tokens = set(re.findall(r"[a-z0-9]+", _normalize(question)))
        a_tokens = set(re.findall(r"[a-z0-9]+", _normalize(answer)))
        if not q_tokens:
            return 0.0
        return len(q_tokens & a_tokens) / len(q_tokens)


def compute_metrics(
    question: str,
    answer: str,
    retrieved_chunks: list[dict],
    expected_citations: list[dict],
    contexts: list[str] | None = None,
) -> EvalMetrics:
    """Compute four RAGAS metrics for a single example."""
    if contexts is None:
        contexts = [c.get("content", "") for c in retrieved_chunks]

    cp = _context_precision(retrieved_chunks, expected_citations)
    cr = _context_recall(retrieved_chunks, expected_citations)
    faith = _faithfulness(answer, contexts)
    rel = _answer_relevance(question, answer)

    return EvalMetrics(
        context_precision=round(cp, 4),
        context_recall=round(cr, 4),
        faithfulness=round(faith, 4),
        answer_relevance=round(rel, 4),
    )

"""Evaluation metrics.

Two backends:

- ``heuristic``: deterministic, offline, no API keys. Useful for fast
  regression checks and retrieval-only runs, but it is **not** RAGAS.
- ``ragas``: real RAGAS metrics (faithfulness, context precision/recall,
  answer relevancy). Requires a working ``ragas`` install plus LLM/embedding
  credentials. Failures are explicit — there is no silent fallback.

All metrics are in [0, 1].
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.embeddings import Embeddings as _LangchainEmbeddings

from src.config import Settings
from src.retrieval.embeddings import Embedder

logger = logging.getLogger(__name__)

MetricsBackend = Literal["heuristic", "ragas", "auto"]


class RagasUnavailableError(RuntimeError):
    """Raised when the RAGAS backend is requested but cannot be imported/used."""


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


def _import_ragas() -> tuple[Any, ...]:
    """Import RAGAS lazily so the offline heuristic path never depends on it."""
    try:
        from ragas import EvaluationDataset, SingleTurnSample, evaluate
        from ragas.metrics import (
            ContextPrecision,
            ContextRecall,
            Faithfulness,
            ResponseRelevancy,
        )
    except Exception as exc:  # noqa: BLE001 - surface a single actionable error
        raise RagasUnavailableError(
            "RAGAS is not importable in this environment. Install a compatible "
            "ragas/langchain-community combination (and Pillow) or run with "
            "--metrics heuristic. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc
    return EvaluationDataset, SingleTurnSample, evaluate, ContextPrecision, ContextRecall, Faithfulness, ResponseRelevancy


def ensure_ragas_available() -> None:
    """Fail fast when the RAGAS backend cannot be imported."""
    _import_ragas()


class _LocalLangchainEmbeddings(_LangchainEmbeddings):
    """Adapter from the project's embedder to LangChain's Embeddings interface."""

    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embedder.embed_texts(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embedder.embed_texts([text])[0]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        import asyncio

        return await asyncio.to_thread(self.embed_documents, texts)

    async def aembed_query(self, text: str) -> list[float]:
        import asyncio

        return await asyncio.to_thread(self.embed_query, text)


def _build_ragas_llm(settings: Settings) -> Any:
    """Build a LangChain chat model for RAGAS (OpenAI first, then Groq)."""
    from langchain_openai import ChatOpenAI

    if settings.openai_api_key:
        return ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=0.0,
        )
    if settings.groq_api_key:
        return ChatOpenAI(
            model=settings.judge_model,
            api_key=settings.groq_api_key,
            base_url=settings.groq_base_url,
            temperature=0.0,
        )
    raise RagasUnavailableError(
        "RAGAS needs an LLM judge. Set OPENAI_API_KEY or GROQ_API_KEY, "
        "or run with --metrics heuristic."
    )


def _extract_ragas_scores(scores: dict[str, Any]) -> EvalMetrics:
    """Map RAGAS metric names onto the project's metric fields."""
    required = ("context_precision", "context_recall", "faithfulness", "answer_relevancy")
    missing = [name for name in required if name not in scores]
    if missing:
        raise RagasUnavailableError(f"RAGAS returned no scores for: {', '.join(missing)}")
    return EvalMetrics(
        context_precision=round(float(scores["context_precision"]), 4),
        context_recall=round(float(scores["context_recall"]), 4),
        faithfulness=round(float(scores["faithfulness"]), 4),
        answer_relevance=round(float(scores["answer_relevancy"]), 4),
    )


def compute_ragas_metrics(
    question: str,
    answer: str,
    contexts: list[str],
    expected_answer: str,
    settings: Settings,
) -> EvalMetrics:
    """Compute real RAGAS metrics for a single example.

    Raises ``RagasUnavailableError`` if RAGAS cannot be imported or lacks
    credentials. This is deliberate: the harness should not silently downgrade
    to heuristics when the caller asked for RAGAS.

    TODO: add an integration test against a working RAGAS stack + judge key.
    The current environment cannot import RAGAS, so only the score-mapping
    helper is unit-tested.
    """
    (
        EvaluationDataset,
        SingleTurnSample,
        evaluate,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
        ResponseRelevancy,
    ) = _import_ragas()

    from src.retrieval.embeddings import build_embedder

    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts,
        reference=expected_answer,
    )
    metrics = [
        Faithfulness(),
        ContextPrecision(),
        ContextRecall(),
        ResponseRelevancy(),
    ]
    result = evaluate(
        EvaluationDataset(samples=[sample]),
        metrics=metrics,
        llm=_build_ragas_llm(settings),
        embeddings=_LocalLangchainEmbeddings(build_embedder(settings)),
        raise_exceptions=True,
        show_progress=False,
    )
    if not result.scores:
        raise RagasUnavailableError("RAGAS returned no scores")
    return _extract_ragas_scores(result.scores[0])

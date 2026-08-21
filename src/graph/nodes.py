"""Phase 5-10 — linear graph nodes.

``classify_complexity -> retrieve -> compress -> call_llm -> finalize``

- ``classify_complexity``: Phase 9 rule-based query-complexity router.
- ``retrieve``: Phase 6 hybrid (dense+BM25) fused to ``fusion_top_k``, then
  Phase 7 cross-encoder reranked to top-``k``.
- ``compress``: Phase 8 trims chunks to query-relevant sentences within the
  token budget.
- ``call_llm``: Phase 9 tier-routed, Phase 10 fallback-chain LLM call.

Parsing, chunking, and embedding/seed stay offline (Phase 1-3 CLIs): they are
deterministic and idempotent, so the graph reads the already-seeded store.
"""

from __future__ import annotations

import logging

from src.config import settings
from src.graph.state import GraphState
from src.ingestion.chunker import Chunk
from src.llm.answer import build_answer_prompt
from src.llm.fallback import call_with_fallback
from src.retrieval.compress import compress_chunks
from src.retrieval.rerank import retrieve_reranked
from src.routing import classify_complexity

logger = logging.getLogger(__name__)


def classify_complexity_node(state: GraphState) -> dict:
    """Classify query complexity to pick the model tier (PRD §3.2 node 2)."""
    tier = classify_complexity(state["query"])
    logger.info("classify_complexity: %r -> %s", state["query"], tier)
    return {"tier": tier}


def retrieve_node(state: GraphState) -> dict:
    """Hybrid retrieval + cross-encoder rerank (PRD §3.2 nodes 3-6)."""
    rows = retrieve_reranked(state["query"], settings, k=state.get("k"))
    logger.info("retrieve_node: %d reranked chunks", len(rows))
    return {"chunks": rows}


def compress_node(state: GraphState) -> dict:
    """Trim chunks to query-relevant spans within the token budget (node 7)."""
    rows = compress_chunks(state["query"], state["chunks"], settings)
    return {"chunks": rows}


def call_llm_node(state: GraphState) -> dict:
    """Tier-routed, fallback-chain LLM call (PRD §3.2 node 9, Phases 9-10)."""
    rows = state["chunks"]
    chunks = [Chunk(content=row["content"], metadata=row) for row in rows]
    prompt = build_answer_prompt(state["query"], chunks)
    result = call_with_fallback(prompt, settings, tier=state.get("tier"))
    return {
        "answer": result.content,
        "provider": result.provider,
        "tokens_in": result.tokens_in,
    }


def finalize_node(state: GraphState) -> dict:
    """Assemble the source list for the final response (PRD §3.2 node 13)."""
    sources = sorted(
        {
            f"{row['ticker']} {row['filing_type']} {row['fiscal_period']} | {row['section']}"
            for row in state["chunks"]
        }
    )
    return {"sources": sources}
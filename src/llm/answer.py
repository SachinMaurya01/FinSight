"""Phase 4-10 — LLM call layer: build the answer prompt, route, and answer.

``answer_query`` turns retrieved chunks + query into an answer. Since Phase 9
it routes the call to the query's complexity tier, and since Phase 10 it walks
the provider fallback chain (``call_with_fallback``) instead of a single
hardcoded provider.
"""

from __future__ import annotations

import logging
from typing import Any

from src.config import Settings
from src.ingestion.chunker import Chunk
from src.llm.fallback import call_with_fallback
from src.llm.output import strip_think_block
from src.llm.prompts import ANSWER_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)

_strip_think_block = strip_think_block


def format_context(chunks: list[Chunk]) -> str:
    """Render retrieved chunks as a numbered, metadata-tagged context block."""
    blocks: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        meta = chunk.metadata
        tags = ", ".join(
            str(meta.get(k)) for k in ("ticker", "filing_type", "fiscal_period", "section")
        )
        blocks.append(f"[{index}] (source: {tags})\n{chunk.content.strip()}")
    return "\n\n".join(blocks)


def build_answer_prompt(query: str, chunks: list[Chunk]) -> str:
    """Render the full prompt for ``answer_query`` (shared with graph node)."""
    return ANSWER_PROMPT_TEMPLATE.format(context=format_context(chunks), question=query)


def answer_query(
    query: str,
    chunks: list[Chunk],
    settings: Settings,
    tier: str | None = None,
    llm: Any | None = None,
) -> str:
    """Answer ``query`` grounded in ``chunks`` via the routed fallback chain.

    ``tier`` selects the model tier (Phase 9). ``llm`` may be injected for
    tests; when provided it bypasses the fallback chain (Phase 4 behavior).
    """
    prompt = build_answer_prompt(query, chunks)
    logger.info("Calling LLM (query=%r, context_chunks=%d, tier=%s)", query, len(chunks), tier)
    if llm is not None:
        response = llm.invoke(prompt)
        return _strip_think_block(str(response.content))
    return call_with_fallback(prompt, settings, tier=tier).content
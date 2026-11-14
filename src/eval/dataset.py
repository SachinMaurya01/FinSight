"""Eval dataset loader."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from src.config import settings


def chunk_id(source_file: str, chunk_index: int | str) -> str:
    """Stable gold chunk identifier used by retrieval metrics (``file#index``)."""
    return f"{source_file}#{chunk_index}"


class ExpectedCitation(BaseModel):
    ticker: str | None = None
    fiscal_period: str | None = None
    section: str | None = None


class EvalExample(BaseModel):
    id: str = Field(description="Unique eval id")
    question: str
    expected_answer: str = Field(description="Ground truth answer")
    expected_citations: list[ExpectedCitation] = Field(default_factory=list)
    gold_chunk_ids: list[str] = Field(
        default_factory=list,
        description="Chunk IDs (``source_file#chunk_index``) that contain the answer.",
    )
    complexity: Literal["simple", "normal", "complex"] = "normal"
    intent: Literal["lookup", "ratio", "comparison", "recommendation", "summary"] | None = None
    is_recommendation: bool = False
    category: str = "fact_extraction"
    expected_abstention: bool = False
    expected_abstention_reason: str | None = None
    injected_context: str | None = Field(
        default=None,
        description="Eval-only hostile context injected into the top-ranked chunk.",
    )
    forbidden_phrases: list[str] = Field(
        default_factory=list,
        description="Phrases that must not appear in the answer if injection was followed.",
    )

    @property
    def effective_intent(self) -> str:
        """Intent used for per-intent reporting, inferred from category when absent."""
        if self.intent:
            return self.intent
        if self.is_recommendation:
            return "recommendation"
        mapping = {
            "comparison": "comparison",
            "tool_calling": "ratio",
            "summarization": "summary",
            "trend_analysis": "summary",
        }
        return mapping.get(self.category, "lookup")


def resolve_gold_chunk_ids(
    example: EvalExample,
    chunks: list[Any],
    *,
    per_citation: int = 3,
) -> list[str]:
    """Derive gold chunk IDs for legacy examples from section-level citations.

    New examples carry explicit ``gold_chunk_ids``; this helper is for the
    hand-authored seed set whose labels only identify ticker/period/section.
    Matching chunks are ranked by lexical overlap with the question + answer,
    with a small bonus for table chunks because numeric answers usually live in
    tables.
    """
    if example.gold_chunk_ids or not example.expected_citations:
        return list(example.gold_chunk_ids)

    query_tokens = set(_tokens(f"{example.question} {example.expected_answer}"))
    selected: list[str] = []
    for citation in example.expected_citations:
        ranked: list[tuple[float, str]] = []
        for chunk in chunks:
            metadata = getattr(chunk, "metadata", {}) or {}
            content = getattr(chunk, "content", "") or ""
            if not _citation_matches(metadata, citation.model_dump()):
                continue
            content_tokens = set(_tokens(content))
            overlap = len(query_tokens & content_tokens) / max(len(query_tokens), 1)
            source_file = metadata.get("source_file")
            chunk_index = metadata.get("chunk_index")
            if source_file is None or chunk_index is None:
                continue
            bonus = 0.05 if metadata.get("chunk_type") == "table" else 0.0
            ranked.append((overlap + bonus, chunk_id(str(source_file), str(chunk_index))))
        ranked.sort(key=lambda item: item[0], reverse=True)
        for _, identifier in ranked[: max(per_citation, 1)]:
            if identifier not in selected:
                selected.append(identifier)
    return selected


def _tokens(text: str) -> list[str]:
    import re

    return re.findall(r"[a-z0-9]+", text.lower())


def _citation_matches(metadata: dict[str, Any], citation: dict[str, Any]) -> bool:
    return all(
        not citation.get(key) or metadata.get(key) == citation.get(key)
        for key in ("ticker", "fiscal_period", "section")
    )


def load_dataset(path: Path | None = None, shuffle: bool = False) -> list[EvalExample]:
    """Load and validate eval dataset."""
    p = path or settings.eval_dataset_path
    if not p.exists():
        raise FileNotFoundError(f"eval dataset not found at {p}")
    raw = json.loads(p.read_text(encoding="utf-8"))
    examples = [EvalExample.model_validate(item) for item in raw]
    if shuffle:
        rnd = random.Random(settings.eval_seed)
        rnd.shuffle(examples)
    return examples

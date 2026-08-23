"""Eval dataset loader."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from src.config import settings


class ExpectedCitation(BaseModel):
    ticker: str | None = None
    fiscal_period: str | None = None
    section: str | None = None


class EvalExample(BaseModel):
    id: str = Field(description="Unique eval id")
    question: str
    expected_answer: str = Field(description="Ground truth answer")
    expected_citations: list[ExpectedCitation] = Field(default_factory=list)
    complexity: Literal["simple", "normal", "complex"] = "normal"
    is_recommendation: bool = False
    category: str = "fact_extraction"


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

"""LLM output post-processing shared across the call layer."""

from __future__ import annotations

import re

_THINK_BLOCK_RE = re.compile(
    r"<\s*think(?:ing)?\s*>.*?<\s*/\s*think(?:ing)?\s*>", re.DOTALL
)


def strip_think_block(content: str) -> str:
    """Remove a leaked `` thinking... response`` reasoning block from model output."""
    return _THINK_BLOCK_RE.sub("", content).strip()
"""Token-counting utilities shared by compression."""

from __future__ import annotations

from functools import lru_cache

from tiktoken import Encoding


@lru_cache(maxsize=1)
def _encoding() -> Encoding:
    return __import__("tiktoken").get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Approximate token count for ``text``."""
    if not text:
        return 0
    return len(_encoding().encode(text))
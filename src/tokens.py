"""Token-counting utilities shared by compression."""

from __future__ import annotations

import logging
from functools import lru_cache

from tiktoken import Encoding

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _encoding() -> Encoding | None:
    """
        other token count methods based on diff models o200k_base, p50k_base, etc
    """
    #  inline way to load a module
    try:
        return __import__("tiktoken").get_encoding("cl100k_base")
    except Exception as exc:  # network/SSL in Zscaler env, missing cache, etc.
        logger.warning("tiktoken cl100k_base load failed (%s), falling back to estimate", exc)
        return None


def count_tokens(text: str) -> int:
    """Approximate token count for ``text``."""
    if not text:
        return 0
    enc = _encoding()
    if enc is not None:
        try:
            return len(enc.encode(text))
        except Exception as exc:
            logger.warning("tiktoken encode failed (%s), falling back to estimate", exc)
    # fallback: ~4 chars per token (conservative, avoids network/SSL dependency)
    return max(1, len(text) // 4)
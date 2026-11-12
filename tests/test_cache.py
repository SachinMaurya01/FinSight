"""Embedding cache tests (in-memory fallback path)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.retrieval import cache


def test_cache_roundtrip() -> None:
    cache.clear_cache()
    original = cache._get_redis_client
    cache._get_redis_client = lambda _settings: None
    try:
        assert cache.get_cached_embedding("test-model", "hello", settings) is None
        cache.set_cached_embedding("test-model", "hello", [0.1, 0.2], settings)
        assert cache.get_cached_embedding("test-model", "hello", settings) == [0.1, 0.2]
        stats = cache.cache_stats()
        assert stats["entries"] == 1
        assert stats["keys"]
    finally:
        cache._get_redis_client = original
        cache.clear_cache()
    print("PASS test_cache_roundtrip")


def test_cache_expiry_and_clear() -> None:
    cache.clear_cache()
    original = cache._get_redis_client
    cache._get_redis_client = lambda _settings: None
    try:
        cache.set_cached_embedding("test-model", "expire-me", [1.0], settings)
        key = cache._cache_key("test-model", "expire-me")
        cache._MEM_TIMESTAMPS[key] = time.time() - cache._TTL_SECONDS - 1
        assert cache.get_cached_embedding("test-model", "expire-me", settings) is None
        assert cache.cache_stats()["entries"] == 0
    finally:
        cache._get_redis_client = original
        cache.clear_cache()
    print("PASS test_cache_expiry_and_clear")


def main() -> int:
    test_cache_roundtrip()
    test_cache_expiry_and_clear()
    return 0


if __name__ == "__main__":
    sys.exit(main())

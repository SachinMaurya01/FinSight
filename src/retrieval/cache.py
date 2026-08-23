"""Embedding cache."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any

from src.config import Settings

logger = logging.getLogger(__name__)

# In-memory fallback store (module-level, survives process lifetime)
_MEM_CACHE: dict[str, list[float]] = {}
_MEM_TIMESTAMPS: dict[str, float] = {}
_TTL_SECONDS = 3600  # 1 hour


def _cache_key(model: str, text: str) -> str:
    h = hashlib.sha256(f"{model}::{text}".encode()).hexdigest()[:16]
    return f"finsight:emb:{model}:{h}"


def _get_redis_client(settings: Settings):  # type: ignore[no-untyped-def]
    try:
        import redis  # type: ignore[import-not-found]

        client = redis.Redis.from_url(settings.redis_url, decode_responses=True, socket_connect_timeout=2, socket_timeout=2)
        client.ping()
        return client
    except Exception as exc:
        logger.info("Redis unavailable (%s), using in-memory cache", exc)
        return None


def get_cached_embedding(model: str, text: str, settings: Settings) -> list[float] | None:
    """Return cached embedding if present and not expired."""
    key = _cache_key(model, text)

    # Try Redis if configured and reachable
    if settings.checkpointer_type == "redis" or True:  # always try Redis when available
        client = _get_redis_client(settings)
        if client is not None:
            try:
                raw = client.get(key)
                if raw:
                    return json.loads(raw)  # type: ignore[no-any-return]
            except Exception as exc:
                logger.warning("Redis get failed: %s", exc)

    # In-memory fallback
    vector = _MEM_CACHE.get(key)
    if vector is not None:
        ts = _MEM_TIMESTAMPS.get(key, 0)
        if time.time() - ts < _TTL_SECONDS:
            return vector
        # expired
        _MEM_CACHE.pop(key, None)
        _MEM_TIMESTAMPS.pop(key, None)
    return None


def set_cached_embedding(model: str, text: str, vector: list[float], settings: Settings) -> None:
    """Store embedding in cache (Redis if available, else in-memory)."""
    key = _cache_key(model, text)

    # Try Redis
    client = _get_redis_client(settings)
    if client is not None:
        try:
            client.setex(key, _TTL_SECONDS, json.dumps(vector))
            return
        except Exception as exc:
            logger.warning("Redis set failed: %s", exc)

    # In-memory fallback
    _MEM_CACHE[key] = vector
    _MEM_TIMESTAMPS[key] = time.time()


def cache_stats() -> dict[str, Any]:
    """Return in-memory cache stats (for testing/done-when)."""
    return {"entries": len(_MEM_CACHE), "keys": list(_MEM_CACHE.keys())[:3]}


def clear_cache() -> None:
    """Clear in-memory cache (for tests)."""
    _MEM_CACHE.clear()
    _MEM_TIMESTAMPS.clear()

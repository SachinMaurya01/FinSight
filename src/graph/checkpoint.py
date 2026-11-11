"""Checkpointer factory."""

from __future__ import annotations

import logging

from src.config import Settings

logger = logging.getLogger(__name__)


def get_checkpointer(settings: Settings):  # type: ignore[no-untyped-def]
    """Return a LangGraph checkpointer per settings.

    In production, a configured Redis checkpointer is required: silently
    falling back to ``MemorySaver`` would lose paused HITL threads across a
    restart. Development keeps the permissive fallback.
    """
    if settings.checkpointer_type == "redis":
        try:
            # langgraph-checkpoint-redis is optional; try import
            from langgraph.checkpoint.redis import RedisSaver  # type: ignore[import-not-found]

            # Test connectivity via redis client
            import redis  # type: ignore[import-not-found]

            client = redis.Redis.from_url(
                settings.redis_url,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
            )
            client.ping()
            saver = RedisSaver(redis_url=settings.redis_url)
            # RedisSaver needs setup() in newer versions
            try:
                saver.setup()
            except Exception:
                pass
            logger.info("Using RedisSaver at %s", settings.redis_url)
            return saver
        except Exception as exc:
            if settings.environment == "production":
                raise RuntimeError(
                    "Redis checkpointer is configured but unreachable in production; "
                    "refusing to fall back to MemorySaver"
                ) from exc
            logger.warning("RedisSaver unavailable (%s), falling back to MemorySaver", exc)

    # Default / fallback
    from langgraph.checkpoint.memory import MemorySaver

    logger.info("Using MemorySaver checkpointer")
    return MemorySaver()

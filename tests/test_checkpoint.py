"""Checkpointer selection: production must not silently lose HITL state."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langgraph.checkpoint.memory import MemorySaver

from src.config import settings
from src.graph.checkpoint import get_checkpointer

BAD_REDIS = "redis://127.0.0.1:1/0"


def test_production_redis_unreachable_raises() -> None:
    config = settings.model_copy(
        update={
            "environment": "production",
            "checkpointer_type": "redis",
            "redis_url": BAD_REDIS,
        }
    )
    try:
        get_checkpointer(config)
        raise AssertionError("production should fail fast when Redis is unreachable")
    except RuntimeError as exc:
        assert "Redis" in str(exc)
    print("PASS test_production_redis_unreachable_raises")


def test_development_redis_unreachable_falls_back() -> None:
    config = settings.model_copy(
        update={
            "environment": "development",
            "checkpointer_type": "redis",
            "redis_url": BAD_REDIS,
        }
    )
    saver = get_checkpointer(config)  # falls back in development
    assert isinstance(saver, MemorySaver)
    print("PASS test_development_redis_unreachable_falls_back")


def main() -> int:
    test_production_redis_unreachable_raises()
    test_development_redis_unreachable_falls_back()
    return 0


if __name__ == "__main__":
    sys.exit(main())

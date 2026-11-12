"""Pending HITL review registry.

LangGraph pauses execution with ``interrupt``; the HTTP layer still needs a way
to list paused threads. We persist the review payload next to the checkpoint:
Redis when available (so it survives an API restart) with an in-memory fallback
for local development and tests.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any

from src.config import Settings
from src.config import settings as default_settings

logger = logging.getLogger(__name__)

_MEMORY_LOCK = threading.Lock()
_MEMORY_PENDING: dict[str, dict[str, Any]] = {}
_REDIS_HASH_KEY = "finsight:pending_reviews"
_REDIS_TTL_SECONDS = 60 * 60 * 24 * 7


class PendingReviewStore:
    """Store paused review payloads keyed by ``thread_id``."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or default_settings
        self._memory: dict[str, dict[str, Any]] = _MEMORY_PENDING
        self._lock = _MEMORY_LOCK

    def _redis(self):  # type: ignore[no-untyped-def]
        try:
            import redis  # type: ignore[import-not-found]

            client = redis.Redis.from_url(
                self._settings.redis_url,
                decode_responses=True,
                socket_connect_timeout=1,
                socket_timeout=1,
            )
            client.ping()
            return client
        except Exception as exc:
            logger.debug("Pending review Redis unavailable: %s", exc)
            return None

    def add(self, thread_id: str, payload: dict[str, Any]) -> None:
        """Record a thread as pending review."""
        entry = dict(payload)
        entry["thread_id"] = thread_id
        entry.setdefault("created_at", time.time())

        client = self._redis()
        if client is not None:
            try:
                client.hset(_REDIS_HASH_KEY, thread_id, json.dumps(entry))
                client.expire(_REDIS_HASH_KEY, _REDIS_TTL_SECONDS)
                return
            except Exception as exc:
                logger.warning("Pending review Redis write failed: %s", exc)

        with self._lock:
            self._memory[thread_id] = entry

    def get(self, thread_id: str) -> dict[str, Any] | None:
        """Return the pending payload for ``thread_id``, if any."""
        client = self._redis()
        if client is not None:
            try:
                raw = client.hget(_REDIS_HASH_KEY, thread_id)
                if raw:
                    return json.loads(raw)  # type: ignore[no-any-return]
            except Exception as exc:
                logger.warning("Pending review Redis read failed: %s", exc)

        with self._lock:
            entry = self._memory.get(thread_id)
            return dict(entry) if entry is not None else None

    def remove(self, thread_id: str) -> None:
        """Remove a thread from the pending review registry."""
        client = self._redis()
        if client is not None:
            try:
                client.hdel(_REDIS_HASH_KEY, thread_id)
            except Exception as exc:
                logger.warning("Pending review Redis delete failed: %s", exc)

        with self._lock:
            self._memory.pop(thread_id, None)

    def list(self) -> list[dict[str, Any]]:
        """Return all pending reviews, newest first."""
        entries: dict[str, dict[str, Any]] = {}
        client = self._redis()
        if client is not None:
            try:
                for raw in client.hvals(_REDIS_HASH_KEY):
                    payload = json.loads(raw)
                    thread_id = payload.get("thread_id")
                    if thread_id:
                        entries[thread_id] = payload
            except Exception as exc:
                logger.warning("Pending review Redis list failed: %s", exc)

        with self._lock:
            for thread_id, payload in self._memory.items():
                entries.setdefault(thread_id, dict(payload))

        return sorted(
            entries.values(),
            key=lambda item: item.get("created_at") or 0,
            reverse=True,
        )


_STORE: PendingReviewStore | None = None


def get_review_store(settings: Settings | None = None) -> PendingReviewStore:
    """Return the process-wide pending review store."""
    global _STORE
    if _STORE is None:
        _STORE = PendingReviewStore(settings)
    return _STORE


def clear_review_store() -> None:
    """Clear the in-memory fallback (used by tests)."""
    with _MEMORY_LOCK:
        _MEMORY_PENDING.clear()

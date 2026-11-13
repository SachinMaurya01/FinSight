"""Append-only HITL audit log.

Reviews are written to Postgres so approvals and rejections are attributable
without storing the full draft answer. The SHA-256 draft hash lets operators
correlate a decision with a checkpoint while limiting sensitive text at rest.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Literal

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from src.config import Settings
from src.config import settings as default_settings

logger = logging.getLogger(__name__)

ReviewDecision = Literal["approved", "rejected"]


def hash_draft(draft: str) -> str:
    """Return the stable digest recorded for a reviewed draft."""
    return hashlib.sha256(draft.encode("utf-8")).hexdigest()


class HitlAuditLog:
    """Postgres-backed HITL decision log."""

    def __init__(self, settings: Settings | None = None, engine: Engine | None = None) -> None:
        self._settings = settings or default_settings
        self._engine = engine

    def _get_engine(self) -> Engine:
        if self._engine is None:
            self._engine = create_engine(
                self._settings.postgres_database_url,
                pool_pre_ping=True,
                connect_args={"connect_timeout": 2},
            )
        return self._engine

    def record(
        self,
        *,
        thread_id: str,
        reviewer: str,
        decision: ReviewDecision,
        note: str | None,
        draft: str,
    ) -> bool:
        """Append one decision; failures are logged but never expose the draft."""
        try:
            with self._get_engine().begin() as connection:
                connection.execute(
                    text(
                        "INSERT INTO hitl_audit (thread_id, reviewer, decision, note, draft_hash) "
                        "VALUES (:thread_id, :reviewer, :decision, :note, :draft_hash)"
                    ),
                    {
                        "thread_id": thread_id,
                        "reviewer": reviewer,
                        "decision": decision,
                        "note": note,
                        "draft_hash": hash_draft(draft),
                    },
                )
            return True
        except Exception as exc:
            logger.error(
                "HITL audit write failed",
                extra={
                    "event": "hitl_audit_write_failed",
                    "thread_id": thread_id,
                    "reviewer": reviewer,
                    "decision": decision,
                    "error_type": type(exc).__name__,
                },
            )
            return False


_AUDIT_LOG: HitlAuditLog | None = None


def get_audit_log(settings: Settings | None = None) -> HitlAuditLog:
    """Return the process-wide audit log."""
    global _AUDIT_LOG
    if _AUDIT_LOG is None:
        _AUDIT_LOG = HitlAuditLog(settings)
    return _AUDIT_LOG

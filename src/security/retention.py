"""Retention enforcement for HITL audit records.

Audit rows older than ``audit_retention_days`` are eligible for deletion. The
purge is a dry run unless ``--apply`` is passed explicitly, so a misconfigured
schedule cannot delete compliance records by accident.

Operational logs are retained by the log shipper (``log_retention_days``), and
pending review payloads expire from Redis after seven days.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from src.config import Settings
from src.config import settings as default_settings

logger = logging.getLogger(__name__)


def audit_cutoff(retention_days: int, now: datetime | None = None) -> datetime:
    """Return the timestamp before which audit rows are past retention."""
    if retention_days < 1:
        raise ValueError("retention_days must be at least 1")
    current = now or datetime.now(timezone.utc)
    return current - timedelta(days=retention_days)


def purge_expired_audit(
    settings: Settings | None = None,
    *,
    apply: bool = False,
    now: datetime | None = None,
    engine: Engine | None = None,
) -> int:
    """Count (dry run) or delete HITL audit rows past ``audit_retention_days``."""
    target = settings or default_settings
    cutoff = audit_cutoff(target.audit_retention_days, now)
    active_engine = engine or create_engine(
        target.postgres_database_url,
        pool_pre_ping=True,
        connect_args={"connect_timeout": 2},
    )
    try:
        with active_engine.begin() as connection:
            if apply:
                result = connection.execute(
                    text("DELETE FROM hitl_audit WHERE created_at < :cutoff"),
                    {"cutoff": cutoff},
                )
                count = int(result.rowcount or 0)
            else:
                count = int(
                    connection.execute(
                        text("SELECT count(*) FROM hitl_audit WHERE created_at < :cutoff"),
                        {"cutoff": cutoff},
                    ).scalar_one()
                )
    finally:
        if engine is None:
            active_engine.dispose()
    logger.info(
        "audit retention %s: %d rows older than %s",
        "purged" if apply else "dry-run",
        count,
        cutoff.isoformat(),
    )
    return count


def main(argv: list[str] | None = None) -> int:
    """Run the audit purge: ``python -m src.security.retention [--apply]``."""
    parser = argparse.ArgumentParser(description="Purge HITL audit rows past retention")
    parser.add_argument("--apply", action="store_true", help="delete rows (default: dry run)")
    args = parser.parse_args(argv)
    try:
        count = purge_expired_audit(default_settings, apply=args.apply)
    except Exception as exc:
        print(f"audit retention failed: {type(exc).__name__}", file=sys.stderr)
        return 1
    action = "deleted" if args.apply else "would delete"
    print(f"{action} {count} audit rows older than {default_settings.audit_retention_days} days")
    return 0


if __name__ == "__main__":
    sys.exit(main())

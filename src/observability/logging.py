"""JSON structured logging with request context injection."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from src.observability.context import context_snapshot

# Attributes already rendered explicitly or internal to logging.
_RESERVED: frozenset[str] = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "taskName",
        "thread",
        "threadName",
    }
)
_HANDLER_MARKER = "_finsight_json_handler"


class ContextFilter(logging.Filter):
    """Copy request context vars onto every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        for key, value in context_snapshot().items():
            if not hasattr(record, key):
                setattr(record, key, value)
        return True


class JsonFormatter(logging.Formatter):
    """Render log records as one-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key in _RESERVED or key.startswith("_") or key in payload:
                continue
            payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


def configure_logging(level: str | int | None = None) -> None:
    """Install one JSON handler on the root logger (idempotent)."""
    root = logging.getLogger()
    if level is not None:
        root.setLevel(level)
    for handler in root.handlers:
        if getattr(handler, _HANDLER_MARKER, False):
            return
    handler = logging.StreamHandler()
    setattr(handler, _HANDLER_MARKER, True)
    handler.setFormatter(JsonFormatter())
    handler.addFilter(ContextFilter())
    root.addHandler(handler)

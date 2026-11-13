"""Minimal PII redaction for logs and query logging helpers."""

from __future__ import annotations

import hashlib
import re
from typing import Any

_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d{1,3}[ .-]?)?(?:\(?\d{3}\)?[ .-]?)\d{3}[ .-]\d{4}(?!\d)")


def redact_text(text: str) -> str:
    """Mask common direct identifiers in log text."""
    redacted = _EMAIL_RE.sub("[redacted-email]", text)
    redacted = _SSN_RE.sub("[redacted-ssn]", redacted)
    return _PHONE_RE.sub("[redacted-phone]", redacted)


def _redact_value(value: Any) -> Any:
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        return {key: _redact_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_value(item) for item in value)
    return value


def redact_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of a structured log payload with direct identifiers masked."""
    return {key: _redact_value(value) for key, value in payload.items()}


def query_log_fields(query: str, *, include_text: bool = False) -> dict[str, Any]:
    """Return privacy-preserving fields for query-related logs."""
    if include_text:
        return {"query_preview": redact_text(query)[:160]}
    digest = hashlib.sha256(query.encode("utf-8")).hexdigest()[:12]
    return {"query_length": len(query), "query_sha256": digest}

"""Instruction/data separation for untrusted filing text and tool output."""

from __future__ import annotations

import re

_UNTRUSTED_MARKER_RE = re.compile(r"<<\s*/?\s*(?:END_)?UNTRUSTED[^>]*>>", re.IGNORECASE)
_LABEL_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_INJECTION_PATTERNS = (
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
    r"disregard\s+(all\s+)?(previous|prior|above)\s+instructions",
    r"forget\s+(all\s+)?(previous|prior|above)\s+instructions",
    r"system\s+prompt",
    r"developer\s+message",
    r"you\s+are\s+now\s+",
    r"reveal\s+(the\s+)?(prompt|instructions)",
    r"override\s+(the\s+)?(rules|instructions)",
    r"answer\s+only\s+with",
    r"do\s+not\s+cite",
)
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)


def _safe_label(label: str) -> str:
    clean = _LABEL_RE.sub("_", label.strip().upper())
    return clean[:40] or "DATA"


def neutralize_untrusted(text: str) -> str:
    """Remove control characters and forged delimiter markers from untrusted text."""
    cleaned = _CONTROL_RE.sub("", text)
    return _UNTRUSTED_MARKER_RE.sub("[filtered-untrusted-marker]", cleaned)


def wrap_untrusted(kind: str, content: str, identifier: str | int | None = None) -> str:
    """Wrap untrusted content in explicit non-instruction delimiters."""
    label = _safe_label(kind)
    suffix = f" id={identifier}" if identifier is not None else ""
    return (
        f"<<UNTRUSTED_{label}{suffix}>>\n"
        f"{neutralize_untrusted(content).strip()}\n"
        f"<<END_UNTRUSTED_{label}>>"
    )


def contains_prompt_injection(text: str) -> bool:
    """Return whether text contains common instruction-override patterns."""
    return bool(_INJECTION_RE.search(text))

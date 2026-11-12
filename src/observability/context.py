"""Request-scoped context variables shared by logs and traces."""

from __future__ import annotations

from contextvars import ContextVar, Token
from typing import Any

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
thread_id_var: ContextVar[str | None] = ContextVar("thread_id", default=None)
tier_var: ContextVar[str | None] = ContextVar("tier", default=None)
provider_var: ContextVar[str | None] = ContextVar("provider", default=None)
trace_id_var: ContextVar[str | None] = ContextVar("trace_id", default=None)

_VARS: dict[str, ContextVar[Any]] = {
    "request_id": request_id_var,
    "thread_id": thread_id_var,
    "tier": tier_var,
    "provider": provider_var,
    "trace_id": trace_id_var,
}


def bind_request_context(**values: Any) -> list[Token[Any]]:
    """Set non-None context values and return tokens for later reset."""
    tokens: list[Token[Any]] = []
    for key, value in values.items():
        var = _VARS.get(key)
        if var is None or value is None:
            continue
        tokens.append(var.set(str(value)))
    return tokens


def reset_request_context(tokens: list[Token[Any]]) -> None:
    """Restore context values to their previous state."""
    for token in reversed(tokens):
        token.var.reset(token)


def context_snapshot() -> dict[str, Any]:
    """Return the current request context for structured log fields."""
    snapshot: dict[str, Any] = {}
    for key, var in _VARS.items():
        value = var.get()
        if value is not None:
            snapshot[key] = value
    return snapshot

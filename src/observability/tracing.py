"""Lightweight per-request tracing for LangGraph nodes and LLM calls.

A trace is a list of spans (node/chain/llm/tool) with latency and token counts.
Spans are logged as one JSON ``trace_completed`` event and exported to
Prometheus histograms/counters. When Langfuse is installed and configured, its
LangChain callback handler is used instead of the local callback handler.
"""

from __future__ import annotations

import functools
import logging
import threading
import time
from collections.abc import Callable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, ParamSpec, TypeVar
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

from src.observability.metrics import record_node_latency

logger = logging.getLogger(__name__)

_trace_var: ContextVar[TraceSession | None] = ContextVar("trace_session", default=None)
_P = ParamSpec("_P")
_R = TypeVar("_R")


@dataclass
class TraceSession:
    """Mutable trace container shared by one request."""

    trace_id: str
    thread_id: str | None = None
    started_at: float = field(default_factory=time.perf_counter)
    spans: list[dict[str, Any]] = field(default_factory=list)
    langfuse_handler: Any | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add_span(
        self,
        name: str,
        *,
        kind: str = "node",
        latency_ms: float,
        status: str = "ok",
        tokens_in: int = 0,
        tokens_out: int = 0,
        provider: str | None = None,
        model: str | None = None,
    ) -> None:
        span = {
            "name": name,
            "kind": kind,
            "latency_ms": round(latency_ms, 3),
            "status": status,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
        }
        if provider:
            span["provider"] = provider
        if model:
            span["model"] = model
        with self._lock:
            self.spans.append(span)

    @property
    def tokens_in(self) -> int:
        return sum(span.get("tokens_in", 0) for span in self.spans)

    @property
    def tokens_out(self) -> int:
        return sum(span.get("tokens_out", 0) for span in self.spans)

    def finish(self) -> None:
        """Emit the completed trace as structured JSON."""
        latency_ms = (time.perf_counter() - self.started_at) * 1000
        if not self.spans:
            return
        logger.info(
            "trace_completed",
            extra={
                "event": "trace_completed",
                "trace_id": self.trace_id,
                "thread_id": self.thread_id,
                "latency_ms": round(latency_ms, 3),
                "tokens_in": self.tokens_in,
                "tokens_out": self.tokens_out,
                "spans": list(self.spans),
            },
        )


def current_trace() -> TraceSession | None:
    """Return the active trace, if any."""
    return _trace_var.get()


def start_trace(
    trace_id: str,
    *,
    thread_id: str | None = None,
    langfuse_handler: Any | None = None,
) -> TraceSession:
    """Create and activate a trace for the current context."""
    session = TraceSession(trace_id=trace_id, thread_id=thread_id, langfuse_handler=langfuse_handler)
    _trace_var.set(session)
    return session


def finish_trace(session: TraceSession | None = None) -> None:
    """Emit and deactivate the current trace."""
    active = session or current_trace()
    if active is not None:
        active.finish()
    _trace_var.set(None)


def trace_node(name: str, fn: Callable[_P, _R]) -> Callable[_P, _R]:
    """Wrap a graph node so latency, token counts, and errors become spans."""

    @functools.wraps(fn)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        session = current_trace()
        started = time.perf_counter()
        try:
            result = fn(*args, **kwargs)
        except Exception:
            latency = time.perf_counter() - started
            record_node_latency(name, latency)
            if session is not None:
                session.add_span(name, latency_ms=latency * 1000, status="error")
            raise
        latency = time.perf_counter() - started
        record_node_latency(name, latency)
        if session is not None:
            update = result if isinstance(result, Mapping) else {}
            session.add_span(
                name,
                latency_ms=latency * 1000,
                tokens_in=_as_int(update.get("tokens_in")),
                tokens_out=_as_int(update.get("tokens_out")),
                provider=update.get("provider"),
                model=update.get("model"),
            )
        return result

    return wrapper


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def trace_callbacks() -> list[BaseCallbackHandler]:
    """Return callbacks that attach LangChain events to the active trace."""
    session = current_trace()
    if session is None:
        return []
    callbacks: list[BaseCallbackHandler] = [TraceCallbackHandler(session)]
    if session.langfuse_handler is not None:
        callbacks.append(session.langfuse_handler)
    return callbacks


class TraceCallbackHandler(BaseCallbackHandler):
    """LangChain callback handler that records local spans."""

    raise_error = False

    def __init__(self, session: TraceSession) -> None:
        self.session = session
        self._starts: dict[UUID, tuple[str, str, float]] = {}
        self._lock = threading.Lock()

    def _start(self, run_id: UUID, name: str, kind: str) -> None:
        with self._lock:
            self._starts[run_id] = (name, kind, time.perf_counter())

    def _finish(
        self,
        run_id: UUID,
        *,
        status: str = "ok",
        tokens_in: int = 0,
        tokens_out: int = 0,
        provider: str | None = None,
        model: str | None = None,
    ) -> None:
        with self._lock:
            entry = self._starts.pop(run_id, None)
        if entry is None:
            return
        name, kind, started = entry
        self.session.add_span(
            name,
            kind=kind,
            latency_ms=(time.perf_counter() - started) * 1000,
            status=status,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            provider=provider,
            model=model,
        )

    def on_chain_start(
        self,
        serialized: dict[str, Any] | None,
        inputs: dict[str, Any] | None,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if parent_run_id is None:
            return
        name = (metadata or {}).get("langgraph_node") or (tags or ["chain"])[-1]
        self._start(run_id, str(name), "chain")

    def on_chain_end(self, outputs: Any, *, run_id: UUID, **kwargs: Any) -> None:
        self._finish(run_id)

    def on_chain_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._finish(run_id, status="error")

    def on_llm_start(
        self,
        serialized: dict[str, Any] | None,
        prompts: list[str],
        *,
        run_id: UUID,
        invocation_params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        model = str((invocation_params or {}).get("model") or (serialized or {}).get("name") or "llm")
        self._start(run_id, model, "llm")

    def on_llm_end(self, response: Any, *, run_id: UUID, **kwargs: Any) -> None:
        tokens_in, tokens_out = _usage_from_response(response)
        self._finish(run_id, tokens_in=tokens_in, tokens_out=tokens_out)

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._finish(run_id, status="error")

    def on_tool_start(
        self,
        serialized: dict[str, Any] | None,
        input_str: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        name = str((serialized or {}).get("name") or "tool")
        self._start(run_id, name, "tool")

    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        self._finish(run_id)

    def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._finish(run_id, status="error")


def _usage_from_response(response: Any) -> tuple[int, int]:
    """Extract prompt/completion token usage from a LangChain LLMResult."""
    llm_output = getattr(response, "llm_output", None) or {}
    usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
    tokens_in = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    tokens_out = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    if tokens_in or tokens_out:
        return tokens_in, tokens_out
    generations = getattr(response, "generations", None) or []
    for group in generations:
        for generation in group:
            message = getattr(generation, "message", None)
            metadata = getattr(message, "usage_metadata", None) or {}
            tokens_in += int(metadata.get("input_tokens") or 0)
            tokens_out += int(metadata.get("output_tokens") or 0)
    return tokens_in, tokens_out


def build_langfuse_handler(settings: Any) -> Any | None:
    """Build a Langfuse LangChain callback handler when configured/installed."""
    if not getattr(settings, "langfuse_enabled", False):
        return None
    try:
        from langfuse.langchain import CallbackHandler
    except ImportError:  # pragma: no cover - optional dependency
        logger.warning("Langfuse tracing requested but langfuse is not installed")
        return None
    try:  # pragma: no cover - requires external Langfuse credentials
        import os

        # Langfuse reads secret key/host from env; mirror settings into env once.
        if settings.langfuse_public_key:
            os.environ.setdefault("LANGFUSE_PUBLIC_KEY", settings.langfuse_public_key)
        if settings.langfuse_secret_key:
            os.environ.setdefault("LANGFUSE_SECRET_KEY", settings.langfuse_secret_key)
        if settings.langfuse_host:
            os.environ.setdefault("LANGFUSE_HOST", settings.langfuse_host)
        return CallbackHandler(public_key=settings.langfuse_public_key)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Could not initialize Langfuse tracing: %s", exc)
        return None

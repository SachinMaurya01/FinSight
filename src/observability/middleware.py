"""ASGI middleware: request IDs, structured request logs, latency, and traces."""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from src.config import settings
from src.observability.context import bind_request_context, reset_request_context
from src.observability.metrics import ACTIVE_REQUESTS, record_request_latency
from src.observability.tracing import build_langfuse_handler, finish_trace, start_trace

logger = logging.getLogger(__name__)


class RequestContextMiddleware:
    """Bind a request_id and trace to every HTTP request."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        headers = {key.decode("latin1").lower(): value.decode("latin1") for key, value in scope.get("headers", [])}
        request_id = headers.get("x-request-id") or uuid.uuid4().hex
        trace_id = request_id
        tokens = bind_request_context(request_id=request_id, trace_id=trace_id)

        session = None
        if settings.tracing_enabled:
            session = start_trace(
                trace_id,
                thread_id=headers.get("x-thread-id"),
                langfuse_handler=build_langfuse_handler(settings),
            )

        started = time.perf_counter()
        status_code = 500
        path = scope.get("path", "")
        method = scope.get("method", "")

        async def send_with_request_id(message: dict[str, Any]) -> None:
            nonlocal status_code
            if message.get("type") == "http.response.start":
                status_code = int(message.get("status", 500))
                response_headers = list(message.get("headers", []))
                response_headers.append((b"x-request-id", request_id.encode("latin1")))
                message["headers"] = response_headers
            await send(message)

        ACTIVE_REQUESTS.inc()
        try:
            await self.app(scope, receive, send_with_request_id)
        finally:
            ACTIVE_REQUESTS.dec()
            latency_seconds = time.perf_counter() - started
            record_request_latency(path, method, latency_seconds)
            logger.info(
                "request_completed",
                extra={
                    "event": "request_completed",
                    "request_id": request_id,
                    "method": method,
                    "path": path,
                    "status_code": status_code,
                    "latency_ms": round(latency_seconds * 1000, 3),
                },
            )
            if session is not None:
                finish_trace(session)
            reset_request_context(tokens)

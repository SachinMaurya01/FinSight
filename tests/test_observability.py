"""Structured logging, metrics, middleware, and tracing tests."""

from __future__ import annotations

import json
import logging
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from src.api.app import create_app
from src.config import settings
from src.observability.context import (
    bind_request_context,
    context_snapshot,
    reset_request_context,
)
from src.observability.logging import ContextFilter, JsonFormatter, configure_logging
from src.observability.metrics import (
    metrics_payload,
    record_fallback,
    record_hitl,
    record_llm_usage,
    record_node_latency,
    record_request,
    record_verification_failure,
)
from src.observability.pricing import estimate_cost
from src.observability.tracing import (
    TraceCallbackHandler,
    _usage_from_response,
    build_langfuse_handler,
    current_trace,
    finish_trace,
    start_trace,
    trace_callbacks,
    trace_node,
)


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def test_json_logging_includes_context() -> None:
    tokens = bind_request_context(request_id="req-1", thread_id="th-1", tier="simple", trace_id="req-1")
    try:
        record = logging.LogRecord("test.logger", logging.INFO, __file__, 1, "hello %s", ("world",), None)
        ContextFilter().filter(record)
        payload = json.loads(JsonFormatter().format(record))
    finally:
        reset_request_context(tokens)

    assert payload["message"] == "hello world"
    assert payload["request_id"] == "req-1"
    assert payload["thread_id"] == "th-1"
    assert payload["tier"] == "simple"
    assert context_snapshot() == {}
    print("PASS test_json_logging_includes_context")


def test_configure_logging_idempotent() -> None:
    root = logging.getLogger()
    configure_logging("INFO")
    count = len(root.handlers)
    configure_logging("INFO")
    assert len(root.handlers) == count
    print("PASS test_configure_logging_idempotent")


def test_metrics_recording() -> None:
    before = REGISTRY.get_sample_value("finsight_requests_total", {"status": "completed", "tier": "normal"}) or 0
    record_request("completed", "normal")
    after = REGISTRY.get_sample_value("finsight_requests_total", {"status": "completed", "tier": "normal"})
    assert after == before + 1

    record_llm_usage("openai", "gpt-4o-mini", 100, 50)
    tokens = REGISTRY.get_sample_value(
        "finsight_llm_tokens_total",
        {"provider": "openai", "model": "gpt-4o-mini", "direction": "input"},
    )
    assert tokens is not None and tokens >= 100

    record_fallback("openai", "groq")
    record_verification_failure("ungrounded_claims", 2)
    record_hitl("approved")
    record_node_latency("retrieve", 0.01)
    assert REGISTRY.get_sample_value("finsight_fallback_total", {"from_provider": "openai", "to_provider": "groq"}) >= 1
    assert REGISTRY.get_sample_value("finsight_node_latency_seconds_count", {"node": "retrieve"}) >= 1

    payload, media_type = metrics_payload()
    assert b"finsight_requests_total" in payload
    assert "text/plain" in media_type
    assert estimate_cost("gpt-4o-mini", 1_000_000, 0) > 0
    print("PASS test_metrics_recording")


def test_middleware_request_id_and_metrics_endpoint() -> None:
    app = create_app(graph=SimpleNamespace())
    handler = _Capture()
    logger = logging.getLogger("src.observability.middleware")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        with TestClient(app) as client:
            response = client.get("/health", headers={"X-Request-ID": "req-abc"})
            assert response.headers["x-request-id"] == "req-abc"
            metrics = client.get("/metrics")
    finally:
        logger.removeHandler(handler)

    assert response.status_code == 200
    assert metrics.status_code == 200
    assert "finsight_requests_total" in metrics.text
    completed = [
        record
        for record in handler.records
        if getattr(record, "event", None) == "request_completed" and getattr(record, "path", "") == "/health"
    ]
    assert completed and completed[-1].request_id == "req-abc"
    print("PASS test_middleware_request_id_and_metrics_endpoint")


def test_trace_node_and_callbacks() -> None:
    session = start_trace("trace-1", thread_id="thread-1")

    def node(_state: dict) -> dict:
        return {"tokens_in": 7, "tokens_out": 3, "provider": "groq", "model": "qwen"}

    wrapped = trace_node("retrieve", node)
    assert wrapped({})["tokens_in"] == 7
    assert session.spans[0]["name"] == "retrieve"
    assert session.spans[0]["tokens_out"] == 3
    assert session.tokens_in == 7 and session.tokens_out == 3

    callbacks = trace_callbacks()
    assert len(callbacks) == 1 and isinstance(callbacks[0], TraceCallbackHandler)
    handler = callbacks[0]

    chain_id = uuid.uuid4()
    handler.on_chain_start(
        {"name": "Retrieve"},
        {},
        run_id=chain_id,
        parent_run_id=uuid.uuid4(),
        metadata={"langgraph_node": "retrieve_node"},
    )
    handler.on_chain_end({}, run_id=chain_id)

    llm_id = uuid.uuid4()
    handler.on_llm_start({"name": "ChatOpenAI"}, ["p"], run_id=llm_id, invocation_params={"model": "gpt-4o"})
    handler.on_llm_end(
        SimpleNamespace(llm_output={"token_usage": {"prompt_tokens": 11, "completion_tokens": 4}}),
        run_id=llm_id,
    )

    tool_id = uuid.uuid4()
    handler.on_tool_start({"name": "get_stock_price"}, "AAPL", run_id=tool_id)
    handler.on_tool_end("{}", run_id=tool_id)

    error_id = uuid.uuid4()
    handler.on_chain_start({"name": "Boom"}, {}, run_id=error_id, parent_run_id=uuid.uuid4())
    handler.on_chain_error(RuntimeError("boom"), run_id=error_id)

    names = [span["name"] for span in session.spans]
    assert "retrieve_node" in names and "gpt-4o" in names and "get_stock_price" in names
    assert any(span["status"] == "error" for span in session.spans)
    assert _usage_from_response(
        SimpleNamespace(
            llm_output={},
            generations=[
                [SimpleNamespace(message=SimpleNamespace(usage_metadata={"input_tokens": 2, "output_tokens": 1}))]
            ],
        )
    ) == (2, 1)
    print("PASS test_trace_node_and_callbacks")


def test_trace_finish_and_error_span() -> None:
    session = start_trace("trace-2", thread_id="thread-2")
    records = _Capture()
    logger = logging.getLogger("src.observability.tracing")
    logger.addHandler(records)
    logger.setLevel(logging.INFO)
    try:

        def boom(_state: dict) -> dict:
            raise RuntimeError("node failed")

        try:
            trace_node("boom", boom)({})
        except RuntimeError:
            pass
        finish_trace(session)
    finally:
        logger.removeHandler(records)

    assert current_trace() is None
    completed = [record for record in records.records if getattr(record, "event", None) == "trace_completed"]
    assert completed and completed[-1].trace_id == "trace-2"
    assert completed[-1].spans[0]["status"] == "error"

    empty = start_trace("trace-3")
    empty.finish()
    assert current_trace() is not None
    finish_trace(empty)
    assert build_langfuse_handler(settings) is None
    print("PASS test_trace_finish_and_error_span")


def main() -> int:
    test_json_logging_includes_context()
    test_configure_logging_idempotent()
    test_metrics_recording()
    test_middleware_request_id_and_metrics_endpoint()
    test_trace_node_and_callbacks()
    test_trace_finish_and_error_span()
    return 0


if __name__ == "__main__":
    sys.exit(main())

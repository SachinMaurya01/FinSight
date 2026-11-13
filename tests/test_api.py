"""API service-layer tests (FastAPI wiring, schemas, HITL review flow)."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from langgraph.types import Command
from pydantic import ValidationError

from src.api import routes as routes_module
from src.api.app import create_app
from src.api.auth import AuthContext
from src.api.rate_limit import FixedWindowRateLimiter
from src.api.routes import ReadinessCheck, _response
from src.api.schemas import AskRequest, AskResponse
from src.compliance.template import NOT_INVESTMENT_ADVICE
from src.config import settings
from src.hitl import store as store_module

COMPLETED = {
    "query": "what was the net income in 2023",
    "answer": "Net income was $97.0 billion [1].",
    "provider": "openai",
    "tier": "simple",
    "verification": {"passed": True},
    "chunks": [
        {
            "ticker": "aapl",
            "filing_type": "10-K",
            "fiscal_period": "FY2023",
            "section": "Item 8",
            "content": "Net income 97,000 million",
        }
    ],
}

PAUSED = {
    **COMPLETED,
    "draft_answer": "Build a bull case: buy AAPL [1].",
    "answer": "Build a bull case: buy AAPL [1].",
    "is_recommendation": True,
}


class FakeAudit:
    """Capture HITL audit records without Postgres."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def record(self, **kwargs: Any) -> bool:
        self.records.append(kwargs)
        return True


class FakeGraph:
    """Minimal LangGraph stand-in supporting invoke/get_state/stream."""

    def __init__(self, paused: bool = False) -> None:
        self.paused = paused
        self.values: dict[str, Any] = PAUSED if paused else COMPLETED
        self.thread_id: str | None = None
        self.resumed = False

    def invoke(self, inputs: Any, config: dict[str, Any]) -> dict[str, Any]:
        self.thread_id = config["configurable"]["thread_id"]
        if isinstance(inputs, Command):
            self.resumed = True
            self.paused = False
            self.values = {**COMPLETED, "hitl_approved": True}
            return self.values
        self.values = dict(PAUSED if self.paused else COMPLETED)
        return self.values

    def get_state(self, config: dict[str, Any]) -> SimpleNamespace:
        return SimpleNamespace(
            values=self.values,
            next=("human_review_gate",) if self.paused else (),
        )

    def stream(self, inputs: Any, config: dict[str, Any], stream_mode: Any = None):
        yield "updates", {"classify_complexity": {"tier": "simple"}}
        yield "messages", (SimpleNamespace(content="Answer "), {"langgraph_node": "call_llm"})
        yield "updates", {"finalize": {"answer": "done", "sources": []}}


class FailingAudit(FakeAudit):
    """Audit store that cannot persist decisions."""

    def record(self, **kwargs: Any) -> bool:
        return False


class SlowGraph(FakeGraph):
    """Graph whose second node update arrives after the request timeout."""

    def stream(self, inputs: Any, config: dict[str, Any], stream_mode: Any = None):
        yield "updates", {"classify_complexity": {"tier": "simple"}}
        time.sleep(0.05)
        yield "updates", {"finalize": {"answer": "done", "sources": []}}


class FakeKeyStore:
    """Accepts one known key with a per-key limit of two requests per minute."""

    def verify(self, raw_key: str) -> AuthContext | None:
        if raw_key != "good-key":
            return None
        return AuthContext(name="tester", identity="key:test", rate_limit_per_minute=2)


def _store_without_redis() -> store_module.PendingReviewStore:
    store = store_module.PendingReviewStore(settings)
    store._redis = lambda: None  # type: ignore[method-assign]
    return store


def _client(
    graph: FakeGraph,
    *,
    audit_log: FakeAudit | None = None,
    auth_enabled: bool = False,
) -> TestClient:
    return TestClient(create_app(graph=graph, auth_enabled=auth_enabled, audit_log=audit_log))


def test_schema_bounds() -> None:
    try:
        AskRequest(query="x" * 1001)
        raise AssertionError("long query should be rejected")
    except ValidationError:
        pass
    try:
        AskRequest(query="valid query", k=21)
        raise AssertionError("k > 20 should be rejected")
    except ValidationError:
        pass
    assert AskRequest(query="valid query").k == 5
    print("PASS test_schema_bounds")


def test_ask_completed() -> None:
    store = _store_without_redis()
    with patch.object(store_module, "_STORE", store):
        with _client(FakeGraph()) as client:
            response = client.post(
                "/v1/ask",
                json={"query": "what was the net income in 2023", "k": 3},
            )
    assert response.status_code == 200, response.text
    body = AskResponse.model_validate(response.json())
    assert body.status == "completed"
    assert body.answer == COMPLETED["answer"]
    assert body.sources == ["aapl 10-K FY2023 | Item 8"]
    assert body.tier == "simple"
    print("PASS test_ask_completed")


def test_unverified_completed_response_includes_caveat() -> None:
    response = _response("thread", {**COMPLETED, "verification": {"passed": False}})
    assert response.status == "completed"
    assert response.verification == {"passed": False}
    assert response.reason == "verification_failed"
    assert response.caveat == "Verification failed: the answer could not be verified against the cited filings."
    print("PASS test_unverified_completed_response_includes_caveat")


def test_ask_pending_and_review() -> None:
    graph = FakeGraph(paused=True)
    audit = FakeAudit()
    store = _store_without_redis()
    with patch.object(store_module, "_STORE", store):
        with _client(graph, audit_log=audit) as client:
            response = client.post("/v1/ask", json={"query": "build a bull case for AAPL"})
            assert response.status_code == 200, response.text
            body = response.json()
            assert body["status"] == "pending_review"
            assert body["answer"].endswith(NOT_INVESTMENT_ADVICE)
            thread_id = body["thread_id"]

            store.add(
                thread_id,
                {"type": "investment_recommendation_review", "query": "q", "answer": "a"},
            )
            pending = client.get("/v1/reviews/pending")
            assert pending.status_code == 200
            assert pending.json()["pending"][0]["thread_id"] == thread_id

            review = client.post(
                f"/v1/reviews/{thread_id}",
                json={"approved": True, "note": "looks fine"},
            )
            assert review.status_code == 200, review.text
            assert review.json()["status"] == "completed"
            assert graph.resumed is True
            assert audit.records == [
                {
                    "thread_id": thread_id,
                    "reviewer": "local",
                    "decision": "approved",
                    "note": "looks fine",
                    "draft": "a",
                }
            ]
    print("PASS test_ask_pending_and_review")


def test_review_fails_closed_when_audit_unavailable() -> None:
    graph = FakeGraph(paused=True)
    store = _store_without_redis()
    with patch.object(store_module, "_STORE", store):
        with _client(graph, audit_log=FailingAudit()) as client:
            body = client.post("/v1/ask", json={"query": "build a bull case for AAPL"}).json()
            thread_id = body["thread_id"]
            store.add(thread_id, {"type": "investment_recommendation_review", "answer": "a"})
            review = client.post(f"/v1/reviews/{thread_id}", json={"approved": True})
            assert review.status_code == 503, review.text
            assert graph.resumed is False
            assert store.get(thread_id) is not None
    print("PASS test_review_fails_closed_when_audit_unavailable")


def test_unauthenticated_requests_get_401() -> None:
    limiter = FixedWindowRateLimiter(settings, clock=lambda: 30.0)
    limiter._redis = lambda: None  # type: ignore[method-assign]
    app = create_app(
        graph=FakeGraph(),
        auth_enabled=True,
        api_key_store=FakeKeyStore(),
        rate_limiter=limiter,
    )
    with patch.object(store_module, "_STORE", _store_without_redis()):
        with TestClient(app) as client:
            payload = {"query": "what was the net income in 2023"}
            assert client.post("/v1/ask", json=payload).status_code == 401
            assert client.post("/v1/ask", json=payload, headers={"X-API-Key": "bad"}).status_code == 401
            ok = client.post("/v1/ask", json=payload, headers={"X-API-Key": "good-key"})
            assert ok.status_code == 200, ok.text
    print("PASS test_unauthenticated_requests_get_401")


def test_burst_traffic_gets_429() -> None:
    limiter = FixedWindowRateLimiter(settings, clock=lambda: 30.0)
    limiter._redis = lambda: None  # type: ignore[method-assign]
    app = create_app(
        graph=FakeGraph(),
        auth_enabled=True,
        api_key_store=FakeKeyStore(),
        rate_limiter=limiter,
    )
    headers = {"X-API-Key": "good-key"}
    payload = {"query": "what was the net income in 2023"}
    with patch.object(store_module, "_STORE", _store_without_redis()):
        with TestClient(app) as client:
            statuses = [client.post("/v1/ask", json=payload, headers=headers).status_code for _ in range(3)]
            limited = client.post("/v1/ask", json=payload, headers=headers)
    assert statuses == [200, 200, 429], statuses
    assert limited.status_code == 429
    assert "Retry-After" in limited.headers
    print("PASS test_burst_traffic_gets_429")


def test_stream_respects_request_timeout() -> None:
    store = _store_without_redis()
    slow_settings = settings.model_copy(update={"request_timeout_seconds": 0.01})
    with (
        patch.object(store_module, "_STORE", store),
        patch.object(routes_module, "settings", slow_settings),
    ):
        with _client(SlowGraph()) as client:
            response = client.get("/v1/ask/stream", params={"query": "what was the net income"})
    assert response.status_code == 200, response.text
    assert "request timed out" in response.text
    assert "event: done" not in response.text
    print("PASS test_stream_respects_request_timeout")


def test_health_and_ready() -> None:
    store = _store_without_redis()
    with patch.object(store_module, "_STORE", store):
        with _client(FakeGraph()) as client:
            assert client.get("/health").status_code == 200

            with patch(
                "src.api.routes._check_postgres",
                lambda: ReadinessCheck(ok=False, detail="connection refused"),
            ):
                response = client.get("/ready")
                assert response.status_code == 503, response.text
                assert response.json()["status"] == "degraded"

            with (
                patch("src.api.routes._check_postgres", lambda: ReadinessCheck(ok=True)),
                patch("src.api.routes._check_redis", lambda: ReadinessCheck(ok=True)),
                patch("src.api.routes._check_llm_provider", lambda: ReadinessCheck(ok=True)),
            ):
                assert client.get("/ready").status_code == 200
    print("PASS test_health_and_ready")


def test_stream() -> None:
    store = _store_without_redis()
    with patch.object(store_module, "_STORE", store):
        with _client(FakeGraph()) as client:
            response = client.get("/v1/ask/stream", params={"query": "what was the net income"})
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: node" in response.text
    assert "event: token" in response.text
    assert "event: done" in response.text
    print("PASS test_stream")


def main() -> int:
    test_schema_bounds()
    test_ask_completed()
    test_unverified_completed_response_includes_caveat()
    test_ask_pending_and_review()
    test_review_fails_closed_when_audit_unavailable()
    test_unauthenticated_requests_get_401()
    test_burst_traffic_gets_429()
    test_stream()
    test_stream_respects_request_timeout()
    test_health_and_ready()
    return 0


if __name__ == "__main__":
    sys.exit(main())

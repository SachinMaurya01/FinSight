"""HTTP routes for ask, streaming, HITL review, and health checks."""

from __future__ import annotations

import json
import logging
import socket
import uuid
from collections.abc import Iterator
from typing import Any, Literal
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from langgraph.types import Command
from sqlalchemy import create_engine, text
from starlette.concurrency import run_in_threadpool

from src.api.schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
    PendingReview,
    PendingReviewsResponse,
    ReadinessCheck,
    ReadinessResponse,
    ReviewRequest,
)
from src.config import settings
from src.hitl.store import get_review_store
from src.observability.context import bind_request_context
from src.observability.metrics import metrics_payload, record_request
from src.observability.tracing import trace_callbacks
from src.verification.abstention import abstention_caveat

logger = logging.getLogger(__name__)

router = APIRouter()


def _graph(request: Request) -> Any:
    graph = getattr(request.app.state, "graph", None)
    if graph is None:  # pragma: no cover - lifespan always sets this
        raise HTTPException(status_code=503, detail="graph is not initialised")
    return graph


def _thread_id(requested: str | None) -> str:
    return requested or uuid.uuid4().hex


def _config(thread_id: str) -> dict[str, Any]:
    config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
    callbacks = trace_callbacks()
    if callbacks:
        config["callbacks"] = callbacks
    return config


def _input(payload: AskRequest, thread_id: str) -> dict[str, Any]:
    return {
        "query": payload.query,
        "k": payload.k,
        "ticker": payload.ticker,
        "fiscal_year": payload.fiscal_year,
        "thread_id": thread_id,
        "verification_retries": 0,
    }


def _sources(state: dict[str, Any]) -> list[str]:
    return sorted(
        {
            f"{row['ticker']} {row['filing_type']} {row['fiscal_period']} | {row['section']}"
            + (" | table" if row.get("chunk_type") == "table" else "")
            for row in state.get("chunks", [])
        }
    )


def _is_pending(graph: Any, config: dict[str, Any], state: dict[str, Any]) -> bool:
    if "__interrupt__" in state:
        return True
    try:
        snapshot = graph.get_state(config)
        next_nodes = getattr(snapshot, "next", ()) or ()
        return settings.hitl_interrupt_node in next_nodes
    except Exception as exc:  # checkpointer may not expose state
        logger.debug("Could not inspect graph state: %s", exc)
        return False


def _response(
    thread_id: str,
    state: dict[str, Any],
    *,
    pending: bool = False,
    reason: str | None = None,
) -> AskResponse:
    answer = state.get("draft_answer") if pending else state.get("answer")
    if answer is None:
        answer = state.get("answer")
    abstention = None if pending else state.get("abstention_reason")
    status: Literal["completed", "pending_review", "degraded", "abstained"] = "pending_review" if pending else "completed"
    if abstention:
        status = "abstained"
    if not pending and state.get("error") and not (answer or "").strip():
        status = "degraded"
    return AskResponse(
        thread_id=thread_id,
        status=status,
        answer=answer,
        sources=[] if abstention else _sources(state),
        provider=state.get("provider"),
        tier=state.get("tier"),
        verification=state.get("verification"),
        reason=reason or abstention or state.get("degraded_reason"),
        caveat=abstention_caveat(abstention) if abstention else None,
    )


@router.post("/v1/ask", response_model=AskResponse)
async def ask(payload: AskRequest, request: Request) -> AskResponse:
    """Run one query through the graph; returns a cited answer or a review pause."""
    graph = _graph(request)
    thread_id = _thread_id(payload.thread_id)
    bind_request_context(thread_id=thread_id)
    config = _config(thread_id)

    try:
        state = await run_in_threadpool(graph.invoke, _input(payload, thread_id), config)
    except Exception as exc:
        from src.llm.fallback import DegradedResponseError

        if isinstance(exc, DegradedResponseError):
            logger.warning("ask degraded for thread=%s: %s", thread_id, exc)
            record_request("degraded", None)
            return AskResponse(
                thread_id=thread_id,
                status="degraded",
                answer=None,
                sources=[],
                reason=str(exc),
            )
        logger.exception("ask failed for thread=%s", thread_id)
        record_request("degraded", None)
        return AskResponse(
            thread_id=thread_id,
            status="degraded",
            answer=None,
            sources=[],
            reason=f"{type(exc).__name__}: {exc}",
        )

    pending = _is_pending(graph, config, state)
    response = _response(thread_id, state, pending=pending)
    bind_request_context(tier=response.tier, provider=response.provider)
    record_request(response.status, response.tier)
    return response


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


@router.get("/v1/ask/stream")
def ask_stream(
    request: Request,
    query: str = Query(min_length=3, max_length=1000),
    k: int = Query(default=5, ge=1, le=20),
    ticker: str | None = Query(default=None, pattern=r"^[A-Z.\-]{1,10}$"),
    fiscal_year: int | None = Query(default=None, ge=2000, le=2100),
    thread_id: str | None = Query(default=None, min_length=1, max_length=128),
) -> StreamingResponse:
    """Stream node progress and LLM tokens as Server-Sent Events."""
    graph = _graph(request)
    resolved_thread = _thread_id(thread_id)
    bind_request_context(thread_id=resolved_thread)
    config = _config(resolved_thread)
    payload = AskRequest(
        query=query,
        k=k,
        ticker=ticker,
        fiscal_year=fiscal_year,
        thread_id=resolved_thread,
    )
    inputs = _input(payload, resolved_thread)

    def event_stream() -> Iterator[str]:
        yield _sse("start", {"thread_id": resolved_thread})
        try:
            for mode, chunk in graph.stream(
                inputs,
                config,
                stream_mode=["updates", "messages"],
            ):
                if mode == "updates":
                    for node, update in chunk.items():
                        yield _sse("node", {"node": node, "update": update})
                elif mode == "messages":
                    message, metadata = chunk
                    content = getattr(message, "content", "")
                    if content:
                        yield _sse(
                            "token",
                            {
                                "node": (metadata or {}).get("langgraph_node"),
                                "token": content,
                            },
                        )
        except Exception as exc:
            from src.llm.fallback import DegradedResponseError

            if isinstance(exc, DegradedResponseError):
                record_request("degraded", None)
                yield _sse("error", {"status": "degraded", "reason": str(exc)})
            else:
                logger.exception("stream failed for thread=%s", resolved_thread)
                record_request("degraded", None)
                yield _sse(
                    "error",
                    {"status": "degraded", "reason": f"{type(exc).__name__}: {exc}"},
                )
            return

        try:
            state = graph.get_state(config)
            values = dict(getattr(state, "values", {}) or {})
        except Exception:
            values = {}

        pending = _is_pending(graph, config, values)
        response = _response(resolved_thread, values, pending=pending)
        bind_request_context(tier=response.tier, provider=response.provider)
        record_request(response.status, response.tier)
        if pending:
            yield _sse("pending_review", response.model_dump())
        yield _sse("done", response.model_dump())

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/v1/reviews/pending", response_model=PendingReviewsResponse)
async def pending_reviews() -> PendingReviewsResponse:
    """List recommendation outputs currently paused for human review."""
    store = get_review_store(settings)
    pending = [
        PendingReview(
            thread_id=item["thread_id"],
            query=item.get("query"),
            answer=item.get("answer"),
            sources=item.get("sources") or [],
            reason=item.get("reason") or item.get("type"),
            created_at=item.get("created_at"),
        )
        for item in store.list()
        if item.get("thread_id")
    ]
    return PendingReviewsResponse(pending=pending)


@router.post("/v1/reviews/{thread_id}", response_model=AskResponse)
async def review(
    thread_id: str,
    payload: ReviewRequest,
    request: Request,
) -> AskResponse:
    """Approve or reject a paused recommendation, resuming its graph thread."""
    graph = _graph(request)
    bind_request_context(thread_id=thread_id)
    config = _config(thread_id)
    store = get_review_store(settings)

    state: dict[str, Any]
    try:
        snapshot = await run_in_threadpool(graph.get_state, config)
        next_nodes = getattr(snapshot, "next", ()) or ()
    except Exception:
        next_nodes = ()

    if store.get(thread_id) is None and settings.hitl_interrupt_node not in next_nodes:
        raise HTTPException(status_code=404, detail="no pending review for thread_id")

    try:
        state = await run_in_threadpool(
            graph.invoke,
            Command(resume={"approved": payload.approved, "note": payload.note}),
            config,
        )
        store.remove(thread_id)
    except Exception as exc:
        logger.exception("review resume failed for thread=%s", thread_id)
        raise HTTPException(status_code=500, detail=f"could not resume thread: {exc}") from exc

    pending = _is_pending(graph, config, state)
    response = _response(thread_id, state, pending=pending)
    bind_request_context(tier=response.tier, provider=response.provider)
    record_request(response.status, response.tier)
    return response


@router.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    """Prometheus metrics exposition."""
    payload, media_type = metrics_payload()
    return Response(content=payload, media_type=media_type)


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness probe: the process is up."""
    return HealthResponse(status="ok", version=settings.app_version)


def _check_postgres() -> ReadinessCheck:
    engine = None
    try:
        engine = create_engine(
            settings.postgres_database_url,
            pool_pre_ping=True,
            connect_args={"connect_timeout": 2},
        )
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return ReadinessCheck(ok=True)
    except Exception as exc:
        return ReadinessCheck(ok=False, detail=f"{type(exc).__name__}: {exc}")
    finally:
        if engine is not None:
            engine.dispose()


def _check_redis() -> ReadinessCheck:
    try:
        import redis  # type: ignore[import-not-found]

        client = redis.Redis.from_url(
            settings.redis_url,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
        client.ping()
        return ReadinessCheck(ok=True)
    except Exception as exc:
        return ReadinessCheck(ok=False, detail=f"{type(exc).__name__}: {exc}")


def _vllm_reachable() -> bool:
    """Best-effort TCP probe for a local/self-hosted OpenAI-compatible endpoint."""
    if not (settings.vllm_base_url and settings.vllm_model):
        return False
    parsed = urlparse(settings.vllm_base_url)
    if not parsed.hostname:
        return False
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((parsed.hostname, port), timeout=1):
            return True
    except OSError:
        return False


def _check_llm_provider() -> ReadinessCheck:
    keyed = [
        provider
        for provider in settings.fallback_chain
        if provider in ("openai", "groq", "gemini") and settings.has_provider_credentials(provider)
    ]
    if keyed:
        return ReadinessCheck(ok=True, detail=", ".join(keyed))
    if "vllm" in settings.fallback_chain and _vllm_reachable():
        return ReadinessCheck(ok=True, detail="vllm")
    return ReadinessCheck(ok=False, detail="no LLM provider credentials configured")


@router.get("/ready", response_model=ReadinessResponse)
async def ready() -> ReadinessResponse | JSONResponse:
    """Readiness probe: Postgres, Redis, and at least one LLM provider are usable."""
    checks = {
        "postgres": await run_in_threadpool(_check_postgres),
        "redis": await run_in_threadpool(_check_redis),
        "llm_provider": await run_in_threadpool(_check_llm_provider),
    }
    ok = all(check.ok for check in checks.values())
    response = ReadinessResponse(status="ok" if ok else "degraded", checks=checks)
    if not ok:
        return JSONResponse(status_code=503, content=response.model_dump())
    return response

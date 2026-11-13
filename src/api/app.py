"""FastAPI application factory.

The compiled LangGraph is built once during ``lifespan`` and stored on
``app.state.graph``; request handlers reuse it rather than recompiling (and
re-loading the checkpointer) per call.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from src.api.routes import router
from src.config import settings
from src.observability.logging import configure_logging
from src.observability.middleware import RequestContextMiddleware

logger = logging.getLogger(__name__)


def _build_default_graph() -> Any:
    from src.graph import build_graph

    return build_graph()


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    graph = getattr(app.state, "graph", None)
    if graph is None:
        logger.info("Building LangGraph once for the API lifespan")
        graph = _build_default_graph()
        app.state.graph = graph
    logger.info(
        "FinSight API ready (env=%s checkpointer=%s)",
        settings.environment,
        settings.checkpointer_type,
    )
    yield
    app.state.graph = None


def create_app(
    graph: Any | None = None,
    *,
    auth_enabled: bool | None = None,
    api_key_store: Any | None = None,
    rate_limiter: Any | None = None,
    concurrency_limiter: Any | None = None,
    audit_log: Any | None = None,
) -> FastAPI:
    """Create the FinSight FastAPI application.

    ``graph`` and the security collaborators are injectable for tests;
    production leaves them unset and uses settings-backed implementations.
    """
    configure_logging(settings.log_level)
    app = FastAPI(
        title="FinSight",
        version=settings.app_version,
        description="Grounded RAG over SEC 10-K filings with HITL review.",
        lifespan=_lifespan,
    )
    if graph is not None:
        app.state.graph = graph
    app.state.auth_enabled = settings.api_auth_enabled if auth_enabled is None else auth_enabled
    if api_key_store is not None:
        app.state.api_key_store = api_key_store
    if rate_limiter is not None:
        app.state.rate_limiter = rate_limiter
    if concurrency_limiter is not None:
        app.state.concurrency_limiter = concurrency_limiter
    if audit_log is not None:
        app.state.audit_log = audit_log
    app.add_middleware(RequestContextMiddleware)
    app.include_router(router)
    return app

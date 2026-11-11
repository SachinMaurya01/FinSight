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


def create_app(graph: Any | None = None) -> FastAPI:
    """Create the FinSight FastAPI application.

    ``graph`` is injectable for tests; production leaves it unset and the
    lifespan builds the default LangGraph from settings.
    """
    app = FastAPI(
        title="FinSight",
        version=settings.app_version,
        description="Grounded RAG over SEC 10-K filings with HITL review.",
        lifespan=_lifespan,
    )
    if graph is not None:
        app.state.graph = graph
    app.include_router(router)
    return app

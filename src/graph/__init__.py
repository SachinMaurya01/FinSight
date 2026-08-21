"""Phase 5-10 — LangGraph pipeline.

Graph: ``START -> classify_complexity -> retrieve -> compress -> call_llm
-> finalize -> END``

Done when (PRD Phase 5): ``graph.invoke({"query": "..."})`` returns the same
answer as the Phase 4 CLI, now via LangGraph — with routing (Phase 9) and the
provider fallback chain (Phase 10) layered on.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.config import settings
from src.graph.nodes import (
    call_llm_node,
    classify_complexity_node,
    compress_node,
    finalize_node,
    retrieve_node,
)
from src.graph.state import GraphState


def build_graph() -> CompiledStateGraph:
    """Compile the linear retrieval->answer graph."""
    graph = StateGraph(GraphState)
    graph.add_node("classify_complexity", classify_complexity_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("compress", compress_node)
    graph.add_node("call_llm", call_llm_node)
    graph.add_node("finalize", finalize_node)
    graph.add_edge(START, "classify_complexity")
    graph.add_edge("classify_complexity", "retrieve")
    graph.add_edge("retrieve", "compress")
    graph.add_edge("compress", "call_llm")
    graph.add_edge("call_llm", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()


def ask(query: str, k: int | None = None) -> dict:
    """Run a query through the graph and return the final state."""
    k = k if k is not None else settings.rerank_top_k
    return build_graph().invoke({"query": query, "k": k})


__all__ = ["GraphState", "ask", "build_graph"]
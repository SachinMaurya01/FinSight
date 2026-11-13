"""LangGraph pipeline: classify -> parse_query -> retrieve -> compress -> call_llm -> tool -> verify -> hitl -> finalize."""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.config import settings
from src.graph.checkpoint import get_checkpointer
from src.graph.nodes import (
    call_llm_node,
    classify_complexity_node,
    compress_node,
    finalize_node,
    human_review_gate_node,
    parse_query_node,
    retrieve_node,
    tool_node,
    verify_citations_node,
)
from src.graph.state import GraphState
from src.observability.context import bind_request_context, reset_request_context
from src.observability.tracing import current_trace, finish_trace, start_trace, trace_node


def _route_after_verify(state: GraphState) -> str:
    """Conditional edge after verification: retry or proceed to HITL."""
    verification = state.get("verification")
    retries = state.get("verification_retries", 0)
    if verification is None:
        return "human_review_gate"
    if verification.get("passed"):
        return "human_review_gate"
    if retries < settings.verification_max_retries:
        # Log retry
        import logging

        logging.getLogger(__name__).info(
            "Verification failed, retrying retrieve (%d/%d)", retries, settings.verification_max_retries
        )
        return "retrieve"
    return "human_review_gate"


def build_graph(checkpointer=None) -> CompiledStateGraph:  # type: ignore[no-untyped-def]
    """Compile the full pipeline graph."""
    if checkpointer is None and settings.hitl_enabled:
        checkpointer = get_checkpointer(settings)

    graph = StateGraph(GraphState)
    graph.add_node("classify_complexity", trace_node("classify_complexity", classify_complexity_node))
    graph.add_node("parse_query", trace_node("parse_query", parse_query_node))
    graph.add_node("retrieve", trace_node("retrieve", retrieve_node))
    graph.add_node("compress", trace_node("compress", compress_node))
    graph.add_node("call_llm", trace_node("call_llm", call_llm_node))
    graph.add_node("tool", trace_node("tool", tool_node))
    graph.add_node("verify_citations", trace_node("verify_citations", verify_citations_node))
    graph.add_node("human_review_gate", trace_node("human_review_gate", human_review_gate_node))
    graph.add_node("finalize", trace_node("finalize", finalize_node))

    graph.add_edge(START, "classify_complexity")
    graph.add_edge("classify_complexity", "parse_query")
    graph.add_edge("parse_query", "retrieve")
    graph.add_edge("retrieve", "compress")
    graph.add_edge("compress", "call_llm")
    graph.add_edge("call_llm", "tool")
    graph.add_edge("tool", "verify_citations")
    graph.add_conditional_edges(
        "verify_citations",
        _route_after_verify,
        {
            "retrieve": "retrieve",
            "human_review_gate": "human_review_gate",
        },
    )
    graph.add_edge("human_review_gate", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile(checkpointer=checkpointer) if checkpointer else graph.compile()


def ask(query: str, k: int | None = None, *, injected_context: str | None = None) -> dict:
    """Run a query through the graph and return final state.

    For HITL-triggering queries this auto-approves when called via this helper
    (non-interactive). The CLI handles real human prompts via Command resume.
    ``injected_context`` is an eval-only fault-injection hook and is never
    exposed by the HTTP API.
    """
    import uuid

    from langgraph.types import Command

    k = k if k is not None else settings.rerank_top_k
    graph = build_graph()
    thread_id = f"ask-{uuid.uuid4().hex[:8]}"
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    own_trace = False
    trace = current_trace()
    if trace is None and settings.tracing_enabled:
        trace = start_trace(thread_id, thread_id=thread_id)
        own_trace = True
    tokens = bind_request_context(thread_id=thread_id, trace_id=thread_id if trace else None)
    try:
        inputs: dict[str, object] = {"query": query, "k": k, "verification_retries": 0}
        if injected_context is not None:
            inputs["injected_context"] = injected_context
        result = graph.invoke(inputs, config=config)

        # If paused at HITL (requires human approval), auto-approve for this helper.
        try:
            state = graph.get_state(config)
            if state.next and "human_review_gate" in state.next:
                result = graph.invoke(
                    Command(resume={"approved": True, "feedback": "auto-approved (ask helper)"}), config=config
                )
        except Exception:
            pass
        return result
    finally:
        if own_trace:
            finish_trace(trace)
        reset_request_context(tokens)


__all__ = ["GraphState", "ask", "build_graph"]

"""Phase 5-14 LangGraph wiring unit tests (KISS)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.graph import build_graph
from src.graph import nodes as graph_nodes


def _fake_rows() -> list[dict]:
    return [
        {
            "content": "Apple faces supply chain risk in China.",
            "score": 0.9,
            "ticker": "aapl",
            "filing_type": "10-K",
            "fiscal_period": "FY2025",
            "section": "Item 1A Risk Factors",
        }
    ]


def test_graph_invoke_returns_answer() -> None:
    # Mock the retrieval and LLM layers to avoid DB/network
    orig_retrieve = graph_nodes.retrieve_reranked
    orig_compress = graph_nodes.compress_chunks
    orig_tools_fb = graph_nodes.call_with_tools_fallback
    orig_fb = graph_nodes.call_with_fallback
    graph_nodes.retrieve_reranked = lambda query, settings, k=None: _fake_rows()
    graph_nodes.compress_chunks = lambda query, rows, settings: rows
    # Mock both tool and non-tool fallback paths
    mock_result = SimpleNamespace(content="Supply chain [1]", provider="groq", tier="simple", tokens_in=100, failures=())
    graph_nodes.call_with_tools_fallback = lambda prompt, settings, tier=None, max_tool_iters=3: mock_result
    graph_nodes.call_with_fallback = lambda prompt, settings, tier=None: mock_result
    try:
        graph = build_graph()
        config = {"configurable": {"thread_id": f"test-{uuid.uuid4().hex[:4]}"}}
        state = graph.invoke({"query": "what was the net income in 2023", "k": 5, "verification_retries": 0}, config=config)
        # Handle HITL pause (should not pause for simple factual)
        try:
            gs = graph.get_state(config)
            if gs.next:
                from langgraph.types import Command
                state = graph.invoke(Command(resume={"approved": True}), config=config)
        except Exception:
            pass
    finally:
        graph_nodes.retrieve_reranked = orig_retrieve
        graph_nodes.compress_chunks = orig_compress
        graph_nodes.call_with_tools_fallback = orig_tools_fb
        graph_nodes.call_with_fallback = orig_fb

    assert state["chunks"][0]["content"] == "Apple faces supply chain risk in China."
    assert state["answer"] == "Supply chain [1]"
    assert state["provider"] == "groq"
    assert state["tier"] == "simple"
    assert state["sources"] == ["aapl 10-K FY2025 | Item 1A Risk Factors"]
    print("PASS test_graph_invoke_returns_answer")


def test_graph_node_order() -> None:
    calls: list[str] = []

    def tracked_classify(state: dict) -> dict:
        calls.append("classify_complexity")
        return {"tier": "normal"}

    def tracked_retrieve(state: dict) -> dict:
        calls.append("retrieve")
        return {"chunks": _fake_rows()}

    def tracked_compress(state: dict) -> dict:
        calls.append("compress")
        return {"chunks": state["chunks"]}

    def tracked_call_llm(state: dict) -> dict:
        calls.append("call_llm")
        return {"answer": "answer [1]", "provider": "groq", "tokens_in": 100, "draft_answer": "answer [1]"}

    def tracked_tool(state: dict) -> dict:
        calls.append("tool")
        return {}

    def tracked_verify(state: dict) -> dict:
        calls.append("verify_citations")
        return {"verification": {"passed": True, "failed_claims": [], "verified_claims": []}, "verification_retries": 0}

    def tracked_hitl(state: dict) -> dict:
        calls.append("human_review_gate")
        return {"is_recommendation": False, "hitl_approved": True}

    def tracked_finalize(state: dict) -> dict:
        calls.append("finalize")
        return {"sources": ["src"]}

    origs = {
        "classify_complexity_node": graph_nodes.classify_complexity_node,
        "retrieve_node": graph_nodes.retrieve_node,
        "compress_node": graph_nodes.compress_node,
        "call_llm_node": graph_nodes.call_llm_node,
        "tool_node": graph_nodes.tool_node,
        "verify_citations_node": graph_nodes.verify_citations_node,
        "human_review_gate_node": graph_nodes.human_review_gate_node,
        "finalize_node": graph_nodes.finalize_node,
    }
    graph_nodes.classify_complexity_node = tracked_classify
    graph_nodes.retrieve_node = tracked_retrieve
    graph_nodes.compress_node = tracked_compress
    graph_nodes.call_llm_node = tracked_call_llm
    graph_nodes.tool_node = tracked_tool
    graph_nodes.verify_citations_node = tracked_verify
    graph_nodes.human_review_gate_node = tracked_hitl
    graph_nodes.finalize_node = tracked_finalize
    try:
        from langgraph.graph import END, START, StateGraph
        from src.graph.state import GraphState

        g = StateGraph(GraphState)
        g.add_node("classify_complexity", graph_nodes.classify_complexity_node)
        g.add_node("retrieve", graph_nodes.retrieve_node)
        g.add_node("compress", graph_nodes.compress_node)
        g.add_node("call_llm", graph_nodes.call_llm_node)
        g.add_node("tool", graph_nodes.tool_node)
        g.add_node("verify_citations", graph_nodes.verify_citations_node)
        g.add_node("human_review_gate", graph_nodes.human_review_gate_node)
        g.add_node("finalize", graph_nodes.finalize_node)
        g.add_edge(START, "classify_complexity")
        g.add_edge("classify_complexity", "retrieve")
        g.add_edge("retrieve", "compress")
        g.add_edge("compress", "call_llm")
        g.add_edge("call_llm", "tool")
        g.add_edge("tool", "verify_citations")
        g.add_edge("verify_citations", "human_review_gate")
        g.add_edge("human_review_gate", "finalize")
        g.add_edge("finalize", END)
        compiled = g.compile()
        compiled.invoke({"query": "q", "verification_retries": 0})
    finally:
        for k, v in origs.items():
            setattr(graph_nodes, k, v)

    assert calls == ["classify_complexity", "retrieve", "compress", "call_llm", "tool", "verify_citations", "human_review_gate", "finalize"], calls
    print("PASS test_graph_node_order")


def main() -> int:
    test_graph_invoke_returns_answer()
    test_graph_node_order()
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Phase 5-10 LangGraph wiring unit tests.

The graph nodes are unit-tested with monkeypatched retrieval / compression /
LLM functions (no DB / network / API key). A live end-to-end run is done via
``python -m src.graph ask "..."`` (requires seeded pgvector + Groq key).

Run with: python tests/test_graph.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

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
    original_retrieve = graph_nodes.retrieve_reranked
    original_compress = graph_nodes.compress_chunks
    original_fallback = graph_nodes.call_with_fallback
    graph_nodes.retrieve_reranked = lambda query, settings, k=None: _fake_rows()
    graph_nodes.compress_chunks = lambda query, rows, settings: rows
    graph_nodes.call_with_fallback = lambda prompt, settings, tier=None: SimpleNamespace(
        content="Supply chain [1]", provider="groq", tier=tier, tokens_in=100, failures=()
    )
    try:
        graph = build_graph()
        state = graph.invoke({"query": "what was the net income in 2023", "k": 5})
    finally:
        graph_nodes.retrieve_reranked = original_retrieve
        graph_nodes.compress_chunks = original_compress
        graph_nodes.call_with_fallback = original_fallback

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
        return {"answer": "answer [1]", "provider": "groq", "tokens_in": 100}

    def tracked_finalize(state: dict) -> dict:
        calls.append("finalize")
        return {"sources": ["src"]}

    originals = {
        "classify": graph_nodes.classify_complexity_node,
        "retrieve": graph_nodes.retrieve_node,
        "compress": graph_nodes.compress_node,
        "call_llm": graph_nodes.call_llm_node,
        "finalize": graph_nodes.finalize_node,
    }
    graph_nodes.classify_complexity_node = tracked_classify
    graph_nodes.retrieve_node = tracked_retrieve
    graph_nodes.compress_node = tracked_compress
    graph_nodes.call_llm_node = tracked_call_llm
    graph_nodes.finalize_node = tracked_finalize
    try:
        graph = StateGraphBuildWithNodes()
        graph.invoke({"query": "q"})
    finally:
        for key, fn in originals.items():
            setattr(graph_nodes, f"{key}_node" if key != "classify" else "classify_complexity_node", fn)

    assert calls == ["classify_complexity", "retrieve", "compress", "call_llm", "finalize"], calls
    print("PASS test_graph_node_order")


def StateGraphBuildWithNodes():
    from langgraph.graph import END, START, StateGraph

    from src.graph.state import GraphState

    graph = StateGraph(GraphState)
    graph.add_node("classify_complexity", graph_nodes.classify_complexity_node)
    graph.add_node("retrieve", graph_nodes.retrieve_node)
    graph.add_node("compress", graph_nodes.compress_node)
    graph.add_node("call_llm", graph_nodes.call_llm_node)
    graph.add_node("finalize", graph_nodes.finalize_node)
    graph.add_edge(START, "classify_complexity")
    graph.add_edge("classify_complexity", "retrieve")
    graph.add_edge("retrieve", "compress")
    graph.add_edge("compress", "call_llm")
    graph.add_edge("call_llm", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()


def main() -> int:
    test_graph_invoke_returns_answer()
    test_graph_node_order()
    return 0


if __name__ == "__main__":
    sys.exit(main())
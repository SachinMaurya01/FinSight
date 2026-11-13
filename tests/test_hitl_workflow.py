"""End-to-end HITL interrupt/resume test with a real LangGraph checkpointer."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from src.compliance.template import apply_investment_disclaimer
from src.config import settings
from src.graph import build_graph
from src.graph import nodes as graph_nodes
from src.hitl import store as store_module

ANSWER = "You should buy AAPL because net income was $97 billion [1]."


def _rows() -> list[dict]:
    return [
        {
            "content": "You should buy AAPL because net income was $97 billion.",
            "score": 0.9,
            "ticker": "aapl",
            "filing_type": "10-K",
            "fiscal_period": "FY2023",
            "section": "Item 8",
        }
    ]


def main() -> int:
    store = store_module.PendingReviewStore(settings)
    store._redis = lambda: None  # type: ignore[method-assign]
    store_module.clear_review_store()
    mock = SimpleNamespace(content=ANSWER, provider="openai", tier="normal", tokens_in=10, failures=())

    with (
        patch.object(store_module, "_STORE", store),
        patch.object(graph_nodes, "retrieve_reranked", lambda query, settings, k=None, filters=None: _rows()),
        patch.object(graph_nodes, "compress_chunks", lambda query, rows, settings: rows),
        patch.object(
            graph_nodes, "call_with_tools_fallback", lambda prompt, settings, tier=None, max_tool_iters=3: mock
        ),
        patch.object(graph_nodes, "call_with_fallback", lambda prompt, settings, tier=None: mock),
    ):
        graph = build_graph(checkpointer=MemorySaver())
        thread_id = "hitl-test-thread"
        config = {"configurable": {"thread_id": thread_id}}
        graph.invoke(
            {"query": "should I buy AAPL", "k": 5, "verification_retries": 0, "thread_id": thread_id},
            config=config,
        )

        snapshot = graph.get_state(config)
        assert settings.hitl_interrupt_node in snapshot.next, snapshot.next
        pending = store.get(thread_id)
        assert pending is not None, "review payload should be registered"
        assert pending["answer"] == ANSWER

        final = graph.invoke(
            Command(resume={"approved": True, "note": "ok"}),
            config=config,
        )
        assert final["hitl_approved"] is True, final
        assert final["answer"] == apply_investment_disclaimer(ANSWER), final.get("answer")
        assert store.get(thread_id) is None

    print("PASS test_hitl_interrupt_resume")
    return 0


if __name__ == "__main__":
    sys.exit(main())

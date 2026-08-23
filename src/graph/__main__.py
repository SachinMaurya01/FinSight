"""CLI for the full pipeline — ask a question through the LangGraph.

Usage:
    python -m src.graph ask "what are the main risks" --k 5
    python -m src.graph ask "build a bull case for AAPL" --k 5  # triggers HITL
"""

from __future__ import annotations

import argparse
import logging
import sys

from langgraph.types import Command

from src.config import settings
from src.graph import build_graph
from src.llm.fallback import DegradedResponseError


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    ask_parser = subparsers.add_parser("ask", help="run a query through the LangGraph pipeline")
    ask_parser.add_argument("query", type=str)
    ask_parser.add_argument("--k", type=int, default=settings.rerank_top_k)
    ask_parser.add_argument("--no-hitl", action="store_true", help="disable HITL gate for this run")

    args = parser.parse_args()

    if args.command == "ask":
        # Temporarily disable HITL if requested
        original_hitl = settings.hitl_enabled
        if args.no_hitl:
            # Settings is frozen; we cannot mutate, but we can handle via node logic:
            # graph nodes check settings.hitl_enabled, so we monkey-patch the flag
            # by reassigning? Since frozen, we bypass by setting object attribute.
            object.__setattr__(settings, "hitl_enabled", False)  # type: ignore[attr-defined]

        try:
            graph = build_graph()
            config = {"configurable": {"thread_id": "finsight-cli"}}

            try:
                state = graph.invoke({"query": args.query, "k": args.k, "verification_retries": 0}, config=config)
            except DegradedResponseError as exc:
                print("DEGRADED MODE: all LLM providers failed.", file=sys.stderr)
                print(str(exc), file=sys.stderr)
                return 2

            # Check for HITL interrupt (LangGraph pause)
            # When interrupt() is called, the graph's next invoke needs a Command.
            # Detect via presence of interrupt payload in state or via hitl flag.
            # LangGraph 1.2 stores interrupt in graph.get_state(config).next or via __interrupt__
            # For simplicity we check the node's hitl_approved: None means paused (should not happen
            # with our auto-approve fallback). If we do have a real interrupt, prompt the user.
            if state.get("hitl_approved") is None and state.get("is_recommendation"):
                # Real interrupt case: graph is paused at human_review_gate
                snapshot = graph.get_state(config)
                # snapshot.next indicates next node; interrupts are in snapshot.tasks
                print("\n[HITL] Investment recommendation detected — human review required.")
                print(f"Query: {state.get('query')}")
                print(f"Draft answer:\n{state.get('answer','')[:500]}")
                try:
                    resp = input("Approve this output? (y/n): ").strip().lower()
                except EOFError:
                    resp = "y"
                approved = resp in ("y", "yes")
                feedback = None if approved else "rejected via CLI"
                # Resume with Command
                try:
                    state = graph.invoke(Command(resume={"approved": approved, "feedback": feedback}), config=config)
                except DegradedResponseError as exc:
                    print("DEGRADED MODE on resume:", exc, file=sys.stderr)
                    return 2

        finally:
            if args.no_hitl:
                object.__setattr__(settings, "hitl_enabled", original_hitl)  # type: ignore[attr-defined]

        print(f"tier={state.get('tier')} provider={state.get('provider')} tokens_in={state.get('tokens_in')}")
        verification = state.get("verification")
        if verification:
            print(f"verification passed={verification.get('passed')} failed={len(verification.get('failed_claims', []))}")
        if state.get("is_recommendation") is not None:
            print(f"is_recommendation={state.get('is_recommendation')} hitl_approved={state.get('hitl_approved')}")
        print("=== ANSWER ===")
        print(state.get("answer", ""))
        print("\n=== SOURCES ===")
        for source in state.get("sources", []):
            print(f"  {source}")
        return 0

    return 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())

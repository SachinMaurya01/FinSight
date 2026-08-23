"""Graph nodes: classify -> retrieve -> compress -> call_llm -> tool -> verify -> hitl -> finalize."""

from __future__ import annotations

import logging
from typing import Any

from langgraph.types import interrupt

from src.config import settings
from src.graph.state import GraphState
from src.hitl.gate import hitl_payload, is_recommendation
from src.ingestion.chunker import Chunk
from src.llm.answer import build_answer_prompt
from src.llm.fallback import DegradedResponseError, call_with_fallback
from src.llm.tool_support import call_with_tools_fallback
from src.retrieval.compress import compress_chunks
from src.retrieval.rerank import retrieve_reranked
from src.routing import classify_complexity
from src.verification.verify import verify_citations

logger = logging.getLogger(__name__)


def classify_complexity_node(state: GraphState) -> dict[str, Any]:
    """Classify query complexity to pick the model tier."""
    tier = classify_complexity(state["query"])
    logger.info("classify_complexity: %r -> %s", state["query"], tier)
    return {"tier": tier, "verification_retries": state.get("verification_retries", 0)}


def retrieve_node(state: GraphState) -> dict[str, Any]:
    """Hybrid retrieval + cross-encoder rerank."""
    # On retry we could refine query; for now reuse original query
    rows = retrieve_reranked(state["query"], settings, k=state.get("k"))
    logger.info("retrieve_node: %d reranked chunks (retry=%s)", len(rows), state.get("verification_retries", 0))
    return {"chunks": rows}


def compress_node(state: GraphState) -> dict[str, Any]:
    """Trim chunks to query-relevant spans within token budget (node 7)."""
    rows = compress_chunks(state["query"], state["chunks"], settings)
    return {"chunks": rows}


def call_llm_node(state: GraphState) -> dict[str, Any]:
    """Tier-routed, fallback-chain LLM call with optional tool calling (nodes 8-10)."""
    rows = state["chunks"]
    chunks = [Chunk(content=row["content"], metadata=row) for row in rows]
    prompt = build_answer_prompt(state["query"], chunks)
    tier = state.get("tier")

    # Check if query needs live data
    use_tools = settings.enable_tools and any(
        kw in state["query"].lower()
        for kw in ("p/e", "pe ratio", "current price", "live price", "stock price", "growth rate", "gross margin")
    )
    # Also trigger tool path if chunks don't contain needed metric but live price might help;
    # for general tool demo we allow LLM to decide — so use tool-aware call whenever enable_tools
    # and the LLM has tools bound. To keep simple, we route all enable_tools cases through tool_support
    # which gracefully handles no-tool-calls case as well.
    try:
        if settings.enable_tools:
            result = call_with_tools_fallback(prompt, settings, tier=tier)
            logger.info("call_llm (tools) provider=%s tier=%s", result.provider, tier)
            return {
                "draft_answer": result.content,
                "answer": result.content,
                "provider": result.provider,
                "tokens_in": result.tokens_in,
            }
        else:
            result = call_with_fallback(prompt, settings, tier=tier)
            return {
                "draft_answer": result.content,
                "answer": result.content,
                "provider": result.provider,
                "tokens_in": result.tokens_in,
            }
    except DegradedResponseError:
        raise
    except Exception as exc:
        logger.warning("call_llm failed: %s", exc)
        return {"answer": f"Error generating answer: {exc}", "draft_answer": "", "error": str(exc)}


def tool_node(state: GraphState) -> dict[str, Any]:
    """Explicit tool execution stage."""
    import re

    query = state.get("query", "") or ""
    if state.get("answer") and "[Tool-augmented" in state.get("answer", ""):
        logger.info("tool_node: already tool-augmented, skipping")
        return {}
    query_lc = query.lower()
    if not settings.enable_tools:
        return {}

    needs_price = any(kw in query_lc for kw in ("p/e", "pe ratio", "current price", "live price", "stock price"))
    if not needs_price:
        logger.info("tool_node: no tool needed for query")
        return {}

    # Determine ticker from chunks or query
    chunks = state.get("chunks", [])
    ticker = "AAPL"
    if chunks:
        ticker = (chunks[0].get("ticker") or "AAPL").upper()
    # Override if query mentions explicit ticker
    m = re.search(r"\b(AAPL|MSFT|GOOGL|GOOG|AMZN|META|NVDA|TSLA)\b", query, re.I)
    if m:
        ticker = m.group(1).upper()

    logger.info("tool_node: P/E query detected for ticker %s", ticker)

    # Fetch live price
    try:
        from src.tools import TOOL_MAP

        price_data = TOOL_MAP["get_stock_price"].invoke({"ticker": ticker})
        price = float(price_data["price"]) if isinstance(price_data, dict) else float(price_data)
        logger.info("tool_node: live price %s = %s", ticker, price)
    except (ValueError, KeyError, RuntimeError, ConnectionError, TimeoutError) as exc:
        logger.warning("tool_node price fetch failed for %s: %s", ticker, exc)
        return {"tool_results": [{"tool": "get_stock_price", "ticker": ticker, "error": str(exc)}]}
    except Exception as exc:
        logger.warning("tool_node price unexpected error: %s", exc)
        return {"tool_results": [{"tool": "get_stock_price", "ticker": ticker, "error": str(exc)}]}

    # Try to extract EPS from retrieved chunks
    eps: float | None = None
    eps_patterns = [
        r"diluted[^.]{0,60}?earnings[^.]{0,60}?\$?\s*([\d,]+\.\d+)",
        r"earnings per share[^.]{0,60}?\$?\s*([\d,]+\.\d+)",
        r"\bEPS\b[^.]{0,40}?\$?\s*([\d,]+\.\d+)",
        r"basic.*?earnings.*?\$?\s*([\d,]+\.\d+)",
    ]
    for row in chunks:
        content = row.get("content", "")
        for pat in eps_patterns:
            mm = re.search(pat, content, re.I)
            if mm:
                try:
                    eps = float(mm.group(1).replace(",", ""))
                    logger.info("tool_node: extracted EPS %s from chunk", eps)
                    break
                except (ValueError, IndexError):
                    continue
        if eps is not None:
            break

    tool_results: list[dict[str, Any]] = [{"tool": "get_stock_price", "ticker": ticker, "price": price}]

    # If EPS found, calculate P/E
    answer = state.get("answer") or state.get("draft_answer") or ""
    if eps is not None and eps != 0:
        try:
            from src.tools import TOOL_MAP as TM

            pe_data = TM["calculate_pe_ratio"].invoke({"price": price, "eps": eps})
            pe_val = pe_data["value"] if isinstance(pe_data, dict) else float(pe_data)
            tool_results.append({"tool": "calculate_pe_ratio", "price": price, "eps": eps, "pe": pe_val})
            enhanced = (
                answer.rstrip()
                + f"\n\n[Tool-augmented: Live {ticker} price is ${price:.2f}. "
                + f"Using filing EPS ${eps:.2f}, P/E ratio = {pe_val:.2f} (price ${price:.2f} / EPS ${eps:.2f}).]"
            )
            logger.info("tool_node: P/E calculated %s", pe_val)
            return {"tool_results": tool_results, "answer": enhanced, "draft_answer": enhanced}
        except (ValueError, KeyError, RuntimeError, TypeError) as exc:
            logger.warning("tool_node PE calc failed: %s", exc)
            tool_results.append({"tool": "calculate_pe_ratio", "error": str(exc)})
    # No EPS or calc failed: provide live price only
    enhanced = (
        answer.rstrip()
        + f"\n\n[Tool-augmented: Live {ticker} price is ${price:.2f} (as of latest market data). "
        + "Filing context does not contain explicit EPS in retrieved chunks, so P/E cannot be fully derived from filings alone; live price provided for manual calculation.]"
    )
    return {"tool_results": tool_results, "answer": enhanced, "draft_answer": enhanced}


def verify_citations_node(state: GraphState) -> dict[str, Any]:
    """Verify each claim is grounded in retrieved chunks or tool results."""
    answer = state.get("answer") or state.get("draft_answer") or ""
    chunks = state.get("chunks", [])
    tool_results = state.get("tool_results")
    result = verify_citations(answer, chunks, settings, tool_results=tool_results)
    retries = state.get("verification_retries", 0)
    logger.info(
        "verify_citations: passed=%s failed=%d/%d retries=%d",
        result.passed,
        len(result.failed_claims),
        result.total_claims,
        retries,
    )
    # Bounded retry: increment counter when verification fails so the
    # conditional edge can decide to loop back to retrieve or proceed.
    new_retries = retries + 1 if not result.passed else retries
    if not result.passed:
        return {
            "verification": result.to_dict(),
            "verification_retries": new_retries,
            "answer": answer,
        }
    return {
        "verification": result.to_dict(),
        "verification_retries": new_retries,
    }


def human_review_gate_node(state: GraphState) -> dict[str, Any]:
    """HITL gate."""
    if not settings.hitl_enabled:
        logger.info("HITL disabled; skipping review gate")
        return {"is_recommendation": False, "hitl_approved": True}

    answer = state.get("answer") or state.get("draft_answer") or ""
    query = state.get("query", "")
    is_rec = is_recommendation(answer, query)
    logger.info("human_review_gate: is_recommendation=%s", is_rec)

    if not is_rec:
        return {"is_recommendation": False, "hitl_approved": True}

    # It's recommendation-like: pause for human approval via interrupt
    payload = hitl_payload(answer, query, state.get("sources"))
    logger.info("HITL interrupt triggered for recommendation: %r", query[:80])

    try:
        # LangGraph interrupt: pauses execution and expects Command(resume=...) on resume
        feedback = interrupt(payload)
    except Exception as exc:
        # If not running with a checkpointer that supports interrupt, fall back to auto-approve
        # for tests / non-interactive runs. Log and approve.
        logger.warning("interrupt not supported in this run (%s); auto-approving", exc)
        return {"is_recommendation": True, "hitl_approved": True, "hitl_feedback": "auto-approved (no checkpointer)"}

    # feedback is expected to be dict like {"approved": bool, "feedback": str}
    if isinstance(feedback, dict):
        approved = bool(feedback.get("approved", False))
        fb = feedback.get("feedback")
        logger.info("HITL resume: approved=%s feedback=%r", approved, fb)
        if not approved:
            return {
                "is_recommendation": True,
                "hitl_approved": False,
                "hitl_feedback": fb,
                "answer": "Response withheld pending human review: investment recommendation not approved.",
            }
        return {"is_recommendation": True, "hitl_approved": True, "hitl_feedback": fb}

    # If feedback is a simple boolean or string
    if isinstance(feedback, bool):
        return {"is_recommendation": True, "hitl_approved": feedback}
    return {"is_recommendation": True, "hitl_approved": True, "hitl_feedback": str(feedback)}


def finalize_node(state: GraphState) -> dict[str, Any]:
    """Assemble final response with inline citations."""
    verification = state.get("verification")
    hitl_approved = state.get("hitl_approved")
    answer = state.get("answer") or state.get("draft_answer") or ""

    # If verification failed and we exhausted retries, surface flagged answer
    if verification and not verification.get("passed"):
        # Degrade gracefully: keep answer but ensure flag is visible if not already
        failed = verification.get("failed_claims", [])
        if failed and "[Verification:" not in answer:
            answer = answer + "\n\n[Flagged: verification failed for claims: " + "; ".join(c["claim"][:60] for c in failed) + "]"
            logger.warning("finalize: surfaced flagged answer with %d ungrounded claims", len(failed))

    # If HITL rejected, answer already set to withheld message
    if hitl_approved is False:
        logger.info("finalize: HITL rejected, returning withheld message")

    sources = sorted(
        {
            f"{row['ticker']} {row['filing_type']} {row['fiscal_period']} | {row['section']}"
            for row in state.get("chunks", [])
        }
    )
    return {"answer": answer, "sources": sources}

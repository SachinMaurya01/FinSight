"""Graph nodes: classify -> retrieve -> compress -> call_llm -> tool -> verify -> hitl -> finalize."""

from __future__ import annotations

import logging
from typing import Any

from langgraph.errors import GraphInterrupt
from langgraph.types import interrupt

from src.compliance.template import apply_investment_disclaimer
from src.config import settings
from src.graph.state import GraphState
from src.hitl.gate import classify_recommendation, hitl_payload
from src.hitl.store import get_review_store
from src.ingestion.chunker import Chunk
from src.llm.answer import build_answer_prompt
from src.llm.fallback import DegradedResponseError, call_with_fallback
from src.llm.tool_support import call_with_tools_fallback
from src.observability.context import bind_request_context
from src.observability.metrics import record_hitl, record_verification_failure
from src.query.understanding import QueryIntent, filters_for, parse_query
from src.retrieval.compress import compress_chunks
from src.retrieval.rerank import retrieve_reranked
from src.routing import classify_complexity
from src.security.pii import query_log_fields
from src.verification.abstention import ABSTAIN_MESSAGE, abstention_reason
from src.verification.verify import verify_citations

logger = logging.getLogger(__name__)


def classify_complexity_node(state: GraphState) -> dict[str, Any]:
    """Classify query complexity to pick the model tier."""
    tier = classify_complexity(state["query"])
    bind_request_context(tier=tier)
    logger.info("classify_complexity", extra={**query_log_fields(state["query"]), "tier": tier})
    return {"tier": tier, "verification_retries": state.get("verification_retries", 0)}


def parse_query_node(state: GraphState) -> dict[str, Any]:
    """Extract tickers, period, form type, intent, and comparison sub-queries."""
    parsed = parse_query(state["query"], settings)
    logger.info(
        "parse_query: tickers=%s fiscal_year=%s form=%s intent=%s sub_queries=%d",
        parsed.tickers,
        parsed.fiscal_year,
        parsed.form_type,
        parsed.intent,
        len(parsed.sub_queries),
    )
    return {
        "tickers": parsed.tickers,
        "ticker": parsed.tickers[0] if len(parsed.tickers) == 1 else state.get("ticker"),
        "fiscal_year": parsed.fiscal_year,
        "form_type": parsed.form_type,
        "intent": parsed.intent,
        "sub_queries": parsed.sub_queries,
    }


def _dedupe_rows(rows: list[dict[str, Any]], k: int | None) -> list[dict[str, Any]]:
    """Merge sub-query retrieval results, keeping the best-scoring copy."""
    best: dict[tuple[Any, Any], dict[str, Any]] = {}
    for row in rows:
        key = (row.get("source_doc"), row.get("chunk_index"))
        existing = best.get(key)
        if existing is None or float(row.get("rerank_score", row.get("score", 0.0))) > float(
            existing.get("rerank_score", existing.get("score", 0.0))
        ):
            best[key] = row
    merged = sorted(
        best.values(),
        key=lambda row: float(row.get("rerank_score", row.get("score", 0.0))),
        reverse=True,
    )
    return merged[:k] if k else merged


def retrieve_node(state: GraphState) -> dict[str, Any]:
    """Hybrid retrieval + cross-encoder rerank, with metadata pre-filters."""
    intent = QueryIntent(
        tickers=state.get("tickers", []),
        fiscal_year=state.get("fiscal_year"),
        form_type=state.get("form_type"),
        intent=state.get("intent") or "lookup",  # type: ignore[arg-type]
        sub_queries=state.get("sub_queries", []),
    )
    k = state.get("k")
    sub_queries = state.get("sub_queries") or []
    if sub_queries:
        rows: list[dict[str, Any]] = []
        for sub_query in sub_queries:
            sub_intent = parse_query(sub_query, settings)
            rows.extend(
                retrieve_reranked(
                    sub_query,
                    settings,
                    k=k,
                    filters=filters_for(sub_intent),
                )
            )
        rows = _dedupe_rows(rows, k)
    else:
        rows = retrieve_reranked(
            state["query"],
            settings,
            k=k,
            filters=filters_for(intent),
        )
    injected = state.get("injected_context")
    if injected and rows:
        rows = list(rows)
        rows[0] = {**rows[0], "content": f"{rows[0].get('content', '')}\n\n{injected}"}
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

    # Also trigger tool path if chunks don't contain needed metric but live price might help;
    # for general tool demo we allow LLM to decide — so use tool-aware call whenever enable_tools
    # and the LLM has tools bound. To keep simple, we route all enable_tools cases through tool_support
    # which gracefully handles no-tool-calls case as well.
    try:
        if settings.enable_tools:
            result = call_with_tools_fallback(prompt, settings, tier=tier)
            logger.info("call_llm (tools) provider=%s tier=%s", result.provider, tier)
        else:
            result = call_with_fallback(prompt, settings, tier=tier)
        bind_request_context(provider=result.provider)
        return {
            "draft_answer": result.content,
            "answer": result.content,
            "provider": result.provider,
            "model": getattr(result, "model", None),
            "tokens_in": result.tokens_in,
            "tokens_out": getattr(result, "tokens_out", 0),
        }
    except DegradedResponseError:
        raise
    except Exception as exc:
        logger.exception("call_llm failed: %s", exc)
        raise DegradedResponseError(f"answer generation failed: {type(exc).__name__}: {exc}") from exc


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

    needs_price = any(
        kw in query_lc
        for kw in (
            "p/e",
            "pe ratio",
            "price-to-earnings",
            "price to earnings",
            "current price",
            "live price",
            "stock price",
            "valuation",
        )
    )
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

    tool_results: list[dict[str, Any]] = [
        {
            "tool": "get_stock_price",
            "ticker": ticker,
            "price": price,
            "as_of": price_data.get("as_of") if isinstance(price_data, dict) else None,
        }
    ]

    # EPS now comes from SEC XBRL, not a regex over retrieved prose. This gives a
    # point-in-time TTM diluted value with an explicit period and accession.
    if not settings.xbrl_enabled:
        return _price_only_answer(state, ticker, price, tool_results, "XBRL is disabled (XBRL_ENABLED=false)")
    try:
        from src.sec.xbrl import XbrlUnavailableError, get_ttm_diluted_eps

        eps_result = get_ttm_diluted_eps(ticker, settings)
    except XbrlUnavailableError as exc:
        logger.warning("tool_node: XBRL EPS unavailable for %s: %s", ticker, exc)
        tool_results.append({"tool": "get_ttm_diluted_eps", "ticker": ticker, "error": str(exc)})
        return _price_only_answer(state, ticker, price, tool_results, str(exc))
    except Exception as exc:
        logger.warning("tool_node: XBRL EPS unexpected error for %s: %s", ticker, exc)
        tool_results.append({"tool": "get_ttm_diluted_eps", "ticker": ticker, "error": str(exc)})
        return _price_only_answer(state, ticker, price, tool_results, f"{type(exc).__name__}: {exc}")

    from src.tools import TOOL_MAP as TM

    pe_data = TM["calculate_pe_ratio"].invoke(
        {
            "price": price,
            "eps": eps_result.value,
            "period_end": eps_result.period_end,
            "eps_basis": eps_result.basis,
            "source": eps_result.source,
            "accession": eps_result.accession,
            "as_of": _price_as_of(price_data),
        }
    )
    pe_val = pe_data.get("value")
    tool_results.append(
        {
            "tool": "calculate_pe_ratio",
            "price": price,
            "eps": eps_result.value,
            "eps_basis": eps_result.basis,
            "eps_period_end": eps_result.period_end,
            "eps_accession": eps_result.accession,
            "source": eps_result.source,
            "pe": pe_val,
            "not_meaningful": pe_data.get("not_meaningful", False),
        }
    )
    answer = state.get("answer") or state.get("draft_answer") or ""
    provenance = (
        f"XBRL diluted TTM EPS ${eps_result.value:.2f} for period ending {eps_result.period_end} "
        f"(accession {eps_result.accession or 'n/a'})"
    )
    if pe_data.get("not_meaningful") or pe_val is None:
        enhanced = (
            answer.rstrip()
            + f"\n\n[Tool-augmented: Live {ticker} price is ${price:.2f}. {provenance}. "
            + "EPS is zero or negative, so P/E is not meaningful.]"
        )
    else:
        enhanced = (
            answer.rstrip()
            + f"\n\n[Tool-augmented: Live {ticker} price is ${price:.2f} "
            + f"(as of {_price_as_of(price_data) or 'latest market data'}). {provenance}. "
            + f"P/E ratio = {pe_val:.2f} (price ${price:.2f} / EPS ${eps_result.value:.2f}).]"
        )
    logger.info("tool_node: P/E calculated %s using XBRL EPS %s", pe_val, eps_result.value)
    return {"tool_results": tool_results, "answer": enhanced, "draft_answer": enhanced}


def _price_as_of(price_data: Any) -> str | None:
    """Extract the price timestamp from a price tool response."""
    return price_data.get("as_of") if isinstance(price_data, dict) else None


def _price_only_answer(
    state: GraphState,
    ticker: str,
    price: float,
    tool_results: list[dict[str, Any]],
    reason: str,
) -> dict[str, Any]:
    """Return a price-only tool result when XBRL EPS cannot be obtained."""
    answer = state.get("answer") or state.get("draft_answer") or ""
    enhanced = (
        answer.rstrip()
        + f"\n\n[Tool-augmented: Live {ticker} price is ${price:.2f}. "
        + f"XBRL diluted TTM EPS is unavailable ({reason}), so P/E cannot be fully derived from filings.]"
    )
    return {"tool_results": tool_results, "answer": enhanced, "draft_answer": enhanced}


def verify_citations_node(state: GraphState) -> dict[str, Any]:
    """Verify each claim is grounded; abstain when the answer cannot be shown safely."""
    answer = state.get("answer") or state.get("draft_answer") or ""
    chunks = state.get("chunks", [])
    tool_results = state.get("tool_results")
    result = verify_citations(
        answer,
        chunks,
        settings,
        use_llm_judge=settings.semantic_verification_enabled,
        tool_results=tool_results,
    )
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
        has_financial_claim = any(
            any(character.isdigit() for character in claim.get("claim", "")) for claim in result.failed_claims
        )
        record_verification_failure(
            "numeric_hallucination" if has_financial_claim else "ungrounded_claims",
            max(len(result.failed_claims), 1),
        )
    update: dict[str, Any] = {
        "verification": result.to_dict(),
        "verification_retries": new_retries,
    }
    if not result.passed and new_retries < settings.verification_max_retries:
        return update

    reason = abstention_reason(
        chunks,
        tickers=state.get("tickers") or [],
        fiscal_year=state.get("fiscal_year"),
        min_rerank_score=settings.abstention_min_rerank_score,
        verification_passed=result.passed,
    )
    if reason is None:
        return update
    logger.warning("abstaining from answer: %s", reason)
    return {**update, "answer": ABSTAIN_MESSAGE, "abstention_reason": reason}


def human_review_gate_node(state: GraphState) -> dict[str, Any]:
    """HITL gate using LangGraph ``interrupt`` so the API can resume by thread_id."""
    if not settings.hitl_enabled:
        logger.info("HITL disabled; skipping review gate")
        record_hitl("disabled")
        return {"is_recommendation": False, "hitl_approved": True}

    answer = state.get("answer") or state.get("draft_answer") or ""
    query = state.get("query", "")
    thread_id = state.get("thread_id")
    is_rec = classify_recommendation(answer, query, settings)
    logger.info("human_review_gate: is_recommendation=%s", is_rec)

    if not is_rec:
        record_hitl("not_required")
        return {"is_recommendation": False, "hitl_approved": True}

    # It's recommendation-like: pause for human approval via interrupt.
    payload = hitl_payload(answer, query, state.get("sources"))
    record_hitl("requested")
    logger.info("HITL interrupt triggered for recommendation", extra=query_log_fields(query))

    store = get_review_store(settings)
    if thread_id:
        store.add(thread_id, payload)

    try:
        # LangGraph interrupt: pauses execution and expects Command(resume=...) on resume.
        feedback = interrupt(payload)
    except GraphInterrupt:
        # This is the expected pause; propagate so LangGraph can checkpoint it.
        raise
    except Exception as exc:
        # No interrupt-capable checkpointer (e.g. ad-hoc test graphs): auto-approve.
        logger.warning("interrupt not supported in this run (%s); auto-approving", exc)
        if thread_id:
            store.remove(thread_id)
        record_hitl("auto_approved")
        return {
            "is_recommendation": True,
            "hitl_approved": True,
            "hitl_feedback": "auto-approved (no checkpointer)",
        }

    if thread_id:
        store.remove(thread_id)

    # feedback is expected to be dict like {"approved": bool, "note": str}
    if isinstance(feedback, dict):
        approved = bool(feedback.get("approved", False))
        fb = feedback.get("note", feedback.get("feedback"))
        logger.info("HITL resume: approved=%s note=%r", approved, fb)
        if not approved:
            record_hitl("rejected")
            return {
                "is_recommendation": True,
                "hitl_approved": False,
                "hitl_feedback": fb,
                "answer": "Response withheld pending human review: investment recommendation not approved.",
            }
        record_hitl("approved")
        return {"is_recommendation": True, "hitl_approved": True, "hitl_feedback": fb}

    # If feedback is a simple boolean or string
    if isinstance(feedback, bool):
        record_hitl("approved" if feedback else "rejected")
        return {"is_recommendation": True, "hitl_approved": feedback}
    record_hitl("approved")
    return {"is_recommendation": True, "hitl_approved": True, "hitl_feedback": str(feedback)}


def finalize_node(state: GraphState) -> dict[str, Any]:
    """Assemble final response with inline citations."""
    verification = state.get("verification")
    hitl_approved = state.get("hitl_approved")
    answer = state.get("answer") or state.get("draft_answer") or ""

    # Abstained answers are already withheld; only flag unverified answers that were shown.
    if verification and not verification.get("passed") and not state.get("abstention_reason"):
        # Degrade gracefully: keep answer but ensure flag is visible if not already
        failed = verification.get("failed_claims", [])
        if failed and "[Verification:" not in answer:
            answer = (
                answer
                + "\n\n[Flagged: verification failed for claims: "
                + "; ".join(c["claim"][:60] for c in failed)
                + "]"
            )
            logger.warning("finalize: surfaced flagged answer with %d ungrounded claims", len(failed))

    # If HITL rejected, answer already set to withheld message
    if hitl_approved is False:
        logger.info("finalize: HITL rejected, returning withheld message")

    if state.get("is_recommendation"):
        answer = apply_investment_disclaimer(answer)

    sources = sorted(
        {
            f"{row['ticker']} {row['filing_type']} {row['fiscal_period']} | {row['section']}"
            for row in state.get("chunks", [])
        }
    )
    return {"answer": answer, "sources": sources}

"""Tool-calling helper for the LLM layer."""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.config import Settings
from src.llm.client import ProviderUnavailableError, build_llm
from src.llm.fallback import (
    DegradedResponseError,
    LLMResult,
    MalformedResponseError,
    _default_model,
    _provider_exception,
)
from src.llm.output import strip_think_block
from src.tokens import count_tokens
from src.tools import TOOL_MAP, TOOLS

logger = logging.getLogger(__name__)


def _execute_tool_call(tool_call: dict[str, Any]) -> ToolMessage:
    """Execute a single tool call dict (LangChain format) and return ToolMessage."""
    name = tool_call.get("name") or tool_call.get("function", {}).get("name")
    args = tool_call.get("args") or tool_call.get("function", {}).get("arguments") or {}
    # args may be JSON string
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except (json.JSONDecodeError, ValueError):
            args = {}
    tool_id = tool_call.get("id") or tool_call.get("tool_call_id") or "unknown"
    if name not in TOOL_MAP:
        logger.warning("Unknown tool %r requested", name)
        return ToolMessage(content=f"Error: unknown tool {name}", tool_call_id=tool_id, name=name or "unknown")

    tool = TOOL_MAP[name]
    try:
        # LangChain StructuredTool.invoke expects dict
        result = tool.invoke(args)  # type: ignore[arg-type]
        content = json.dumps(result) if not isinstance(result, str) else result
        logger.info("Tool %s(%s) -> %s", name, args, content[:200])
        return ToolMessage(content=content, tool_call_id=tool_id, name=name)
    except (ValueError, KeyError, RuntimeError, TypeError) as exc:
        logger.warning("Tool %s failed: %s", name, exc)
        return ToolMessage(content=f"Error executing {name}: {exc}", tool_call_id=tool_id, name=name)
    except Exception as exc:
        logger.warning("Tool %s unexpected error: %s", name, exc)
        return ToolMessage(content=f"Error executing {name}: {exc}", tool_call_id=tool_id, name=name)


def _messages_from_prompt(prompt: str) -> list[SystemMessage | HumanMessage]:
    """Split the answer prompt into system/human messages for tool-capable chat."""
    # The prompt already contains CONTEXT + QUESTION; keep as single human for simplicity
    # but add system instruction to use tools when needed.
    system = SystemMessage(
        content=(
            "You are a financial research assistant. Use the provided context and "
            "available tools to answer. If the question asks for a live price, P/E ratio, "
            "growth rate or margin that requires live data, call the appropriate tool. "
            "Cite passage numbers for filing-grounded claims."
        )
    )
    return [system, HumanMessage(content=prompt)]


def call_with_tools_fallback(
    prompt: str,
    settings: Settings,
    tier: str | None = None,
    max_tool_iters: int = 3,
) -> LLMResult:
    """Invoke with tools bound, execute tool calls, and return final answer.

    Walks the fallback chain for each LLM invocation. Tool execution is local
    (yfinance, ratio math) and not provider-dependent.
    """
    tier_cfg = settings.tier_model(tier) if tier else None
    ordered = list(settings.fallback_chain)
    if tier_cfg is not None and tier_cfg.provider in ordered:
        ordered.remove(tier_cfg.provider)
        ordered.insert(0, tier_cfg.provider)

    logger.info("Tool fallback chain order: %s (tier=%s)", " -> ".join(ordered), tier)
    messages = _messages_from_prompt(prompt)

    # We will attempt each provider for the initial call; on success with tool_calls
    # we execute tools locally and then make a second LLM call (again with fallback).
    # For simplicity, the fallback for the second call reuses the same ordered list
    # and the provider that succeeded first.
    failures: list[tuple[str, str]] = []

    # --- First LLM invocation (may return tool_calls) ---
    first_result: AIMessage | None = None
    first_provider: str | None = None
    first_model: str | None = None

    for provider in ordered:
        model_name = tier_cfg.model_name if (tier_cfg and provider == tier_cfg.provider) else None
        try:
            llm = build_llm(settings, provider, model_name=model_name)  # type: ignore[arg-type]
        except ProviderUnavailableError as exc:
            logger.info("Skipping provider=%s (no credentials): %s", provider, exc)
            failures.append((provider, "no_credentials"))
            continue
        try:
            # Bind tools if LLM supports it
            try:
                llm_with_tools = llm.bind_tools(TOOLS)  # type: ignore[attr-defined]
            except (AttributeError, ValueError, NotImplementedError) as exc:
                logger.warning("Provider %s does not support bind_tools: %s", provider, exc)
                llm_with_tools = llm
            response = llm_with_tools.invoke(messages)  # type: ignore[arg-type]
            # LangChain may return AIMessage or string
            if isinstance(response, str):
                response = AIMessage(content=response)
            first_result = response  # type: ignore[assignment]
            first_provider = provider
            first_model = model_name or _default_model(settings, provider)
            logger.info("Tool-first call served by %s model=%s", provider, first_model)
            break
        except Exception as exc:
            if _provider_exception(provider, exc):
                failures.append((provider, type(exc).__name__))
                continue
            raise

    if first_result is None or first_provider is None:
        detail = "; ".join(f"{p}:{r}" for p, r in failures) or "no providers configured"
        raise DegradedResponseError(f"all providers failed on tool-first call ({detail})")

    # No tool calls -> return directly
    tool_calls = getattr(first_result, "tool_calls", None) or []
    # Also handle case where tool_calls stored in additional_kwargs
    if not tool_calls and isinstance(first_result, AIMessage) and first_result.additional_kwargs.get("tool_calls"):
        # Normalize; but bind_tools usually puts in tool_calls attr
        tool_calls = first_result.additional_kwargs["tool_calls"]

    if not tool_calls:
        content = strip_think_block(str(first_result.content))
        if not content.strip():
            raise MalformedResponseError("empty/whitespace-only response")
        tokens = count_tokens(prompt)
        return LLMResult(
            content=content,
            provider=first_provider,
            tier=tier,
            tokens_in=tokens,
            failures=tuple(failures),
        )

    logger.info("LLM requested %d tool calls: %s", len(tool_calls), [tc.get("name") for tc in tool_calls])

    # Execute tool calls locally
    tool_messages: list[ToolMessage] = []
    for tc in tool_calls:
        # LangChain tool_calls are dicts with id/name/args
        tool_messages.append(_execute_tool_call(tc))

    # Second LLM call with tool results appended
    # Build new message list: original messages + first AI + tool results
    second_messages: list = list(messages) + [first_result] + tool_messages  # type: ignore[list-item]

    for provider in ordered:
        model_name2 = tier_cfg.model_name if (tier_cfg and provider == tier_cfg.provider) else None
        try:
            llm2 = build_llm(settings, provider, model_name=model_name2)  # type: ignore[arg-type]
        except ProviderUnavailableError as exc:
            logger.info("Skipping provider=%s on tool-second call: %s", provider, exc)
            continue
        try:
            response2 = llm2.invoke(second_messages)  # type: ignore[arg-type]
            if isinstance(response2, str):
                response2 = AIMessage(content=response2)
            content2 = strip_think_block(str(response2.content))
            if not content2.strip():
                raise MalformedResponseError("empty response on tool-second call")
            tokens2 = count_tokens(prompt) + sum(count_tokens(m.content) for m in tool_messages)  # type: ignore[attr-defined]
            logger.info("Tool-second call served by %s model=%s", provider, model_name2 or _default_model(settings, provider))
            return LLMResult(
                content=content2,
                provider=provider,
                tier=tier,
                tokens_in=tokens2,
                failures=tuple(failures),
            )
        except Exception as exc:
            if _provider_exception(provider, exc):
                failures.append((provider, type(exc).__name__))
                continue
            raise

    detail = "; ".join(f"{p}:{r}" for p, r in failures) or "no providers configured"
    raise DegradedResponseError(f"all providers failed on tool-second call ({detail})")

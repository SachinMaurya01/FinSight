"""Prometheus metric definitions and recording helpers."""

from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from src.observability.pricing import estimate_cost

REQUESTS_TOTAL = Counter(
    "finsight_requests_total",
    "Total API requests by terminal status and model tier.",
    ["status", "tier"],
)
NODE_LATENCY = Histogram(
    "finsight_node_latency_seconds",
    "LangGraph node latency in seconds.",
    ["node"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)
REQUEST_LATENCY = Histogram(
    "finsight_request_latency_seconds",
    "End-to-end HTTP request latency in seconds.",
    ["path", "method"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)
ACTIVE_REQUESTS = Gauge(
    "finsight_requests_in_progress",
    "HTTP requests currently being processed.",
)
LLM_TOKENS = Counter(
    "finsight_llm_tokens_total",
    "LLM tokens consumed by provider, model, and direction.",
    ["provider", "model", "direction"],
)
LLM_COST = Counter(
    "finsight_llm_cost_usd_total",
    "Estimated LLM cost in USD by provider and model.",
    ["provider", "model"],
)
FALLBACK_TOTAL = Counter(
    "finsight_fallback_total",
    "Provider fallback hops.",
    ["from_provider", "to_provider"],
)
VERIFICATION_FAILURES = Counter(
    "finsight_verification_failures_total",
    "Citation verification failures by reason.",
    ["reason"],
)
HITL_TOTAL = Counter(
    "finsight_hitl_total",
    "Human-in-the-loop review outcomes.",
    ["decision"],
)


def record_request(status: str, tier: str | None = None) -> None:
    """Increment the request counter."""
    REQUESTS_TOTAL.labels(status=status, tier=tier or "unknown").inc()


def record_request_latency(path: str, method: str, seconds: float) -> None:
    """Record end-to-end request latency."""
    REQUEST_LATENCY.labels(path=path, method=method).observe(seconds)


def record_node_latency(node: str, seconds: float) -> None:
    """Record one graph node's latency."""
    NODE_LATENCY.labels(node=node).observe(seconds)


def record_llm_usage(provider: str, model: str, tokens_in: int, tokens_out: int) -> None:
    """Record token usage and estimated cost for one LLM call."""
    LLM_TOKENS.labels(provider=provider, model=model, direction="input").inc(tokens_in)
    LLM_TOKENS.labels(provider=provider, model=model, direction="output").inc(tokens_out)
    LLM_COST.labels(provider=provider, model=model).inc(estimate_cost(model, tokens_in, tokens_out))


def record_fallback(from_provider: str, to_provider: str) -> None:
    """Record one fallback hop."""
    FALLBACK_TOTAL.labels(from_provider=from_provider, to_provider=to_provider).inc()


def record_verification_failure(reason: str, count: int = 1) -> None:
    """Record citation verification failures."""
    VERIFICATION_FAILURES.labels(reason=reason).inc(count)


def record_hitl(decision: str) -> None:
    """Record a HITL gate outcome."""
    HITL_TOTAL.labels(decision=decision).inc()


def metrics_payload() -> tuple[bytes, str]:
    """Render the Prometheus exposition payload and media type."""
    return generate_latest(), CONTENT_TYPE_LATEST

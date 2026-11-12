"""Approximate LLM pricing (USD per 1M tokens) for cost metrics."""

from __future__ import annotations

# Ordered longest-prefix-first so specific models win over family defaults.
_PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-4o": (2.50, 10.00),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-2.0-flash": (0.10, 0.40),
    "qwen/qwen3.5-9b": (0.05, 0.10),
    "openai/gpt-oss-120b": (0.15, 0.60),
    "qwen3.5:9b": (0.0, 0.0),
}
_DEFAULT = (0.0, 0.0)


def _lookup(model: str) -> tuple[float, float]:
    normalized = model.lower()
    for prefix in sorted(_PRICES, key=len, reverse=True):
        if normalized.startswith(prefix):
            return _PRICES[prefix]
    return _DEFAULT


def estimate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    """Return estimated USD cost for one call."""
    input_price, output_price = _lookup(model)
    return (tokens_in / 1_000_000) * input_price + (tokens_out / 1_000_000) * output_price

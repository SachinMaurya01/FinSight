"""Financial ratio calculator tools."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import tool

from src.tools.schemas import RatioResult

logger = logging.getLogger(__name__)


def _validate_denominator(value: float, name: str) -> None:
    if value == 0:
        raise ValueError(f"{name} must not be zero (division by zero)")
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric, got {type(value).__name__}")


@tool
def calculate_pe_ratio(price: float, eps: float) -> dict[str, Any]:
    """Calculate P/E ratio = price / EPS.

    Args:
        price: Current stock price (must be >0).
        eps: Diluted earnings per share (must be !=0).
    Returns:
        Validated RatioResult dict.
    """
    _validate_denominator(eps, "eps")
    if price <= 0:
        raise ValueError("price must be >0")
    value = price / eps
    result = RatioResult(
        ratio_type="pe_ratio",
        value=value,
        numerator=price,
        denominator=eps,
        formula="P/E = price / eps",
    )
    logger.info("calculate_pe_ratio price=%s eps=%s -> %s", price, eps, value)
    return result.model_dump()


@tool
def calculate_gross_margin(revenue: float, gross_profit: float) -> dict[str, Any]:
    """Calculate gross margin = gross_profit / revenue.

    Args:
        revenue: Total revenue (must be !=0).
        gross_profit: Gross profit.
    Returns:
        Validated RatioResult dict with margin as fraction 0..1 (multiply by 100 for %).
    """
    _validate_denominator(revenue, "revenue")
    value = gross_profit / revenue
    result = RatioResult(
        ratio_type="gross_margin",
        value=value,
        numerator=gross_profit,
        denominator=revenue,
        formula="gross_margin = gross_profit / revenue",
    )
    logger.info("calculate_gross_margin revenue=%s gross_profit=%s -> %s", revenue, gross_profit, value)
    return result.model_dump()


@tool
def calculate_growth_rate(current: float, previous: float) -> dict[str, Any]:
    """Calculate growth rate = (current - previous) / previous.

    Args:
        current: Current period value.
        previous: Previous period value (must be !=0).
    Returns:
        Validated RatioResult dict with growth as fraction (e.g. 0.07 = 7%).
    """
    _validate_denominator(previous, "previous")
    value = (current - previous) / previous
    result = RatioResult(
        ratio_type="growth_rate",
        value=value,
        numerator=current - previous,
        denominator=previous,
        formula="growth_rate = (current - previous) / previous",
    )
    logger.info("calculate_growth_rate current=%s previous=%s -> %s", current, previous, value)
    return result.model_dump()

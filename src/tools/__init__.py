"""Tool registry."""

from src.tools.price import get_stock_price
from src.tools.ratios import calculate_gross_margin, calculate_growth_rate, calculate_pe_ratio

# Registry for execution
TOOLS = [get_stock_price, calculate_pe_ratio, calculate_gross_margin, calculate_growth_rate]

TOOL_MAP = {t.name: t for t in TOOLS}

__all__ = [
    "TOOLS",
    "TOOL_MAP",
    "calculate_gross_margin",
    "calculate_growth_rate",
    "calculate_pe_ratio",
    "get_stock_price",
]
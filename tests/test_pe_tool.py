"""P/E tool integration with XBRL EPS (no network)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.sec.xbrl as xbrl
import src.tools as tools
from src.graph.nodes import tool_node
from src.sec.xbrl import DilutedEps, XbrlUnavailableError


class _PriceTool:
    def invoke(self, payload: dict) -> dict:
        assert payload["ticker"] == "AAPL"
        return {"ticker": "AAPL", "price": 200.0, "currency": "USD", "as_of": "2026-01-02T20:00:00Z"}


def _run(eps_result: DilutedEps | Exception) -> dict:
    original_price = tools.TOOL_MAP["get_stock_price"]
    original_eps = xbrl.get_ttm_diluted_eps
    tools.TOOL_MAP["get_stock_price"] = _PriceTool()  # type: ignore[assignment]

    def fake_eps(ticker: str, settings):  # type: ignore[no-untyped-def]
        if isinstance(eps_result, Exception):
            raise eps_result
        return eps_result

    xbrl.get_ttm_diluted_eps = fake_eps  # type: ignore[assignment]
    try:
        return tool_node(
            {
                "query": "What is the current P/E ratio for AAPL?",
                "chunks": [{"ticker": "aapl", "content": "Apple filing context"}],
                "answer": "Draft answer [1]",
                "draft_answer": "Draft answer [1]",
            }
        )
    finally:
        tools.TOOL_MAP["get_stock_price"] = original_price  # type: ignore[assignment]
        xbrl.get_ttm_diluted_eps = original_eps  # type: ignore[assignment]


def test_pe_uses_xbrl_ttm_eps() -> None:
    result = _run(
        DilutedEps(
            ticker="AAPL",
            value=8.0,
            period_end="2025-09-27",
            fiscal_period="FY",
            accession="0000320193-25-000079",
        )
    )
    pe = next(item for item in result["tool_results"] if item["tool"] == "calculate_pe_ratio")
    assert pe["pe"] == 25.0
    assert pe["eps_basis"] == "diluted_ttm"
    assert pe["source"] == "xbrl"
    assert pe["eps_period_end"] == "2025-09-27"
    assert "diluted TTM EPS $8.00" in result["answer"]
    assert "period ending 2025-09-27" in result["answer"]
    assert "0000320193-25-000079" in result["answer"]
    print("PASS test_pe_uses_xbrl_ttm_eps")


def test_pe_not_meaningful_for_negative_eps() -> None:
    result = _run(
        DilutedEps(
            ticker="AAPL",
            value=-1.5,
            period_end="2025-09-27",
            fiscal_period="FY",
            accession="acc",
        )
    )
    pe = next(item for item in result["tool_results"] if item["tool"] == "calculate_pe_ratio")
    assert pe["pe"] is None
    assert pe["not_meaningful"] is True
    assert "not meaningful" in result["answer"]
    print("PASS test_pe_not_meaningful_for_negative_eps")


def test_pe_degrades_without_xbrl() -> None:
    result = _run(XbrlUnavailableError("no CIK mapping"))
    assert "XBRL diluted TTM EPS is unavailable" in result["answer"]
    assert "no CIK mapping" in result["answer"]
    assert any(item["tool"] == "get_ttm_diluted_eps" for item in result["tool_results"])
    print("PASS test_pe_degrades_without_xbrl")


def main() -> int:
    test_pe_uses_xbrl_ttm_eps()
    test_pe_not_meaningful_for_negative_eps()
    test_pe_degrades_without_xbrl()
    return 0


if __name__ == "__main__":
    sys.exit(main())

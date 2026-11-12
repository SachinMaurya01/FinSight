"""Financial tool tests (ratio math + mocked price provider)."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tools.price import _fetch_price_sync, get_stock_price
from src.tools.ratios import (
    calculate_gross_margin,
    calculate_growth_rate,
    calculate_pe_ratio,
)


def _expect_error(fn, *args) -> Exception:  # type: ignore[no-untyped-def]
    try:
        fn(*args)
    except Exception as exc:
        return exc
    raise AssertionError("expected an exception")


def test_ratios() -> None:
    pe = calculate_pe_ratio.invoke(
        {
            "price": 100.0,
            "eps": 4.0,
            "period_end": "2025-09-27",
            "eps_basis": "diluted_ttm",
            "source": "xbrl",
            "accession": "0000320193-25-000079",
            "as_of": "2026-01-02T00:00:00Z",
        }
    )
    assert pe["ratio_type"] == "pe_ratio"
    assert pe["value"] == 25.0
    assert pe["period_end"] == "2025-09-27"
    assert pe["eps_basis"] == "diluted_ttm"
    assert pe["source"] == "xbrl"
    assert pe["accession"] == "0000320193-25-000079"
    assert pe["as_of"] == "2026-01-02T00:00:00Z"
    assert pe["not_meaningful"] is False

    margin = calculate_gross_margin.invoke({"revenue": 200.0, "gross_profit": 80.0})
    assert margin["value"] == 0.4

    growth = calculate_growth_rate.invoke({"current": 110.0, "previous": 100.0})
    assert abs(growth["value"] - 0.1) < 1e-9
    print("PASS test_ratios")


def test_ratio_validation() -> None:
    assert isinstance(_expect_error(calculate_pe_ratio.func, -1.0, 2.0), ValueError)
    zero = calculate_pe_ratio.invoke({"price": 10.0, "eps": 0.0})
    assert zero["value"] is None
    assert zero["display"] == "not meaningful"
    assert zero["not_meaningful"] is True
    negative = calculate_pe_ratio.invoke({"price": 10.0, "eps": -1.0})
    assert negative["value"] is None
    assert negative["not_meaningful"] is True
    assert isinstance(_expect_error(calculate_gross_margin.func, 0.0, 1.0), ValueError)
    assert isinstance(_expect_error(calculate_growth_rate.func, 1.0, 0.0), ValueError)
    print("PASS test_ratio_validation")


class _FakeHistory:
    empty = True


class _FakeTicker:
    def __init__(self, price: float | None) -> None:
        self.fast_info = SimpleNamespace(last_price=price)

    def history(self, **_kwargs):  # type: ignore[no-untyped-def]
        return _FakeHistory()


def test_price_fallback_and_errors() -> None:
    fake_module = types.ModuleType("yfinance")
    fake_module.Ticker = lambda _ticker: _FakeTicker(123.45)  # type: ignore[attr-defined]
    previous = sys.modules.get("yfinance")
    sys.modules["yfinance"] = fake_module
    try:
        parsed = _fetch_price_sync(" aapl ")
        assert parsed["ticker"] == "AAPL"
        assert parsed["price"] == 123.45
        assert get_stock_price.invoke({"ticker": "AAPL"})["price"] == 123.45

        fake_module.Ticker = lambda _ticker: _FakeTicker(None)  # type: ignore[attr-defined]
        assert isinstance(_expect_error(_fetch_price_sync, "AAPL"), RuntimeError)
        assert isinstance(_expect_error(_fetch_price_sync, "  "), ValueError)
    finally:
        if previous is None:
            sys.modules.pop("yfinance", None)
        else:
            sys.modules["yfinance"] = previous
    print("PASS test_price_fallback_and_errors")


def main() -> int:
    test_ratios()
    test_ratio_validation()
    test_price_fallback_and_errors()
    return 0


if __name__ == "__main__":
    sys.exit(main())

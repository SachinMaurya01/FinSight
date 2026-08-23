"""Live stock price tool via yfinance."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from langchain_core.tools import tool

from src.tools.schemas import TickerPriceResponse

logger = logging.getLogger(__name__)


def _fetch_price_sync(ticker: str) -> dict[str, Any]:
    """Fetch price synchronously; raises on failure with specific exceptions."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("yfinance not installed") from exc

    normalized = ticker.strip().upper()
    if not normalized:
        raise ValueError("ticker must not be empty")

    try:
        yf_ticker = yf.Ticker(normalized)
    except (ValueError, RuntimeError, KeyError) as exc:
        logger.warning("yfinance Ticker init failed for %s: %s", normalized, exc)
        raise RuntimeError(f"failed to init yfinance ticker {normalized}") from exc

    # Try history first (most reliable)
    price: float | None = None
    as_of: str | None = None

    try:
        hist = yf_ticker.history(period="1d", auto_adjust=False)
        if not hist.empty:
            price = float(hist["Close"].iloc[-1])
            # hist index is Timestamp
            ts = hist.index[-1]
            if hasattr(ts, "isoformat"):
                as_of = ts.isoformat()
    except (ValueError, KeyError, RuntimeError, ConnectionError, TimeoutError) as exc:
        logger.warning("yfinance history failed for %s: %s", normalized, exc)

    # Fallback to fast_info / info
    if price is None:
        try:
            fast = getattr(yf_ticker, "fast_info", None)
            if fast is not None:
                candidate = getattr(fast, "last_price", None)
                if candidate is not None:
                    price = float(candidate)
        except (ValueError, KeyError, RuntimeError, AttributeError) as exc:
            logger.warning("yfinance fast_info failed for %s: %s", normalized, exc)

    if price is None:
        try:
            info = yf_ticker.info  # type: ignore[attr-defined]
            candidate = info.get("currentPrice") or info.get("regularMarketPrice")
            if candidate is not None:
                price = float(candidate)
        except (ValueError, KeyError, RuntimeError, ConnectionError) as exc:
            logger.warning("yfinance info failed for %s: %s", normalized, exc)

    if price is None or price <= 0:
        raise RuntimeError(f"could not fetch live price for ticker {normalized}")

    if as_of is None:
        as_of = datetime.now(timezone.utc).isoformat()

    validated = TickerPriceResponse(ticker=normalized, price=price, currency="USD", as_of=as_of)
    return validated.model_dump()


@tool
def get_stock_price(ticker: str) -> dict[str, Any]:
    """Fetch the current live stock price for a ticker via yfinance.

    Args:
        ticker: Stock ticker symbol, e.g. 'AAPL'.
    Returns:
        Dict with ticker, price, currency, as_of.
    """
    return _fetch_price_sync(ticker)

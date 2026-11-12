"""SEC EDGAR/XBRL structured financial data."""

from src.sec.xbrl import (
    CIK_BY_TICKER,
    DilutedEps,
    XbrlFact,
    XbrlUnavailableError,
    fetch_company_facts,
    get_ttm_diluted_eps,
    normalize_facts,
    ticker_to_cik,
)

__all__ = [
    "CIK_BY_TICKER",
    "DilutedEps",
    "XbrlFact",
    "XbrlUnavailableError",
    "fetch_company_facts",
    "get_ttm_diluted_eps",
    "normalize_facts",
    "ticker_to_cik",
]

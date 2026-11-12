"""SEC XBRL ``companyfacts`` client and normalized financial facts.

The SEC publishes every XBRL fact a company has filed at
``https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json``. This module
fetches that payload (throttled, retried, and cached), normalizes facts to a
small schema, and derives a trailing-twelve-month diluted EPS for P/E.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from datetime import date
from typing import Any

import httpx

from src.config import Settings
from src.retrieval.cache import get_redis_client

logger = logging.getLogger(__name__)

# CIKs for the supported sample corpus plus common large-cap tickers. CIK is the
# SEC's stable company identifier; ticker mappings can change over time.
CIK_BY_TICKER: dict[str, int] = {
    "AAPL": 320193,
    "MSFT": 789019,
    "GOOGL": 1652044,
    "GOOG": 1652044,
    "AMZN": 1018724,
    "META": 1326801,
    "NVDA": 1045810,
    "TSLA": 1318605,
}

_EPS_CONCEPT = "EarningsPerShareDiluted"
_EPS_UNIT = "USD/shares"
_TRANSIENT_STATUS = frozenset({429, 500, 502, 503, 504})

_MEM_FACTS: dict[str, tuple[float, dict[str, Any]]] = {}
_MEM_LOCK = threading.Lock()
_THROTTLE_LOCK = threading.Lock()
_LAST_REQUEST_AT = 0.0


class XbrlUnavailableError(RuntimeError):
    """Raised when XBRL facts cannot be fetched or are insufficient."""


class _TransientXbrlError(RuntimeError):
    """Internal signal for a retryable SEC response."""


@dataclass(frozen=True)
class XbrlFact:
    """One normalized companyfact."""

    concept: str
    value: float
    unit: str
    period_start: str | None
    period_end: str
    fiscal_year: int | None
    fiscal_period: str | None
    form: str | None
    accession: str | None
    filed: str | None

    @property
    def duration_days(self) -> int | None:
        if not self.period_start or not self.period_end:
            return None
        try:
            return (date.fromisoformat(self.period_end) - date.fromisoformat(self.period_start)).days
        except ValueError:
            return None


@dataclass(frozen=True)
class DilutedEps:
    """TTM diluted EPS with the filing/period provenance required by the P/E tool."""

    ticker: str
    value: float
    period_end: str
    fiscal_period: str | None
    accession: str | None
    basis: str = "diluted_ttm"
    source: str = "xbrl"


def ticker_to_cik(ticker: str) -> int:
    """Return the SEC CIK for a supported ticker."""
    normalized = ticker.strip().upper()
    cik = CIK_BY_TICKER.get(normalized)
    if cik is None:
        raise XbrlUnavailableError(
            f"no SEC CIK mapping for {normalized!r}; add it to CIK_BY_TICKER or resolve it from SEC company_tickers.json"
        )
    return cik


def _cik_string(cik: int) -> str:
    return f"{cik:010d}"


def companyfacts_url(cik: int, settings: Settings) -> str:
    """Build the companyfacts URL for a CIK."""
    base = settings.sec_companyfacts_base_url.rstrip("/")
    return f"{base}/CIK{_cik_string(cik)}.json"


def _cache_key(cik: int) -> str:
    return f"finsight:xbrl:companyfacts:{_cik_string(cik)}"


def _cache_get(cik: int, settings: Settings) -> dict[str, Any] | None:
    key = _cache_key(cik)
    with _MEM_LOCK:
        entry = _MEM_FACTS.get(key)
    if entry is not None and time.time() - entry[0] < settings.sec_cache_ttl_seconds:
        return entry[1]

    if settings.checkpointer_type == "redis":
        client = get_redis_client(settings)
        if client is not None:
            try:
                raw = client.get(key)
                if raw:
                    return json.loads(raw)  # type: ignore[no-any-return]
            except Exception as exc:
                logger.warning("XBRL Redis get failed: %s", exc)
    return None


def _cache_set(cik: int, payload: dict[str, Any], settings: Settings) -> None:
    key = _cache_key(cik)
    with _MEM_LOCK:
        _MEM_FACTS[key] = (time.time(), payload)

    if settings.checkpointer_type == "redis":
        client = get_redis_client(settings)
        if client is not None:
            try:
                client.setex(key, settings.sec_cache_ttl_seconds, json.dumps(payload))
            except Exception as exc:
                logger.warning("XBRL Redis set failed: %s", exc)


def clear_xbrl_cache() -> None:
    """Clear the in-memory companyfacts cache (tests)."""
    with _MEM_LOCK:
        _MEM_FACTS.clear()


def _throttle(settings: Settings) -> None:
    """Serialize SEC requests and respect the configured fair-use interval."""
    global _LAST_REQUEST_AT
    with _THROTTLE_LOCK:
        elapsed = time.monotonic() - _LAST_REQUEST_AT
        wait = settings.sec_request_throttle_seconds - elapsed
        if wait > 0:
            time.sleep(wait)
        _LAST_REQUEST_AT = time.monotonic()


def _get_json(url: str, settings: Settings, client: httpx.Client | None, throttle: bool) -> dict[str, Any]:
    headers = {
        "User-Agent": settings.sec_edgar_user_agent,
        "Accept": "application/json",
        "Accept-Encoding": "gzip, deflate",
    }
    last_error: Exception | None = None
    for attempt in range(1, settings.sec_max_attempts + 1):
        if throttle:
            _throttle(settings)
        try:
            if client is not None:
                response = client.get(url, headers=headers)
            else:
                with httpx.Client(timeout=20.0, follow_redirects=True) as http:
                    response = http.get(url, headers=headers)
            if response.status_code in _TRANSIENT_STATUS:
                raise _TransientXbrlError(f"SEC returned HTTP {response.status_code}")
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise XbrlUnavailableError("SEC companyfacts response was not a JSON object")
            return payload
        except (httpx.TimeoutException, httpx.ConnectError, _TransientXbrlError) as exc:
            last_error = exc
            if attempt >= settings.sec_max_attempts:
                break
            delay = settings.sec_backoff_seconds * (2 ** (attempt - 1))
            logger.warning("SEC request failed (%s); retrying in %.1fs", exc, delay)
            if delay:
                time.sleep(delay)
        except (httpx.HTTPError, ValueError) as exc:
            raise XbrlUnavailableError(f"SEC companyfacts request failed: {type(exc).__name__}: {exc}") from exc
    raise XbrlUnavailableError(f"SEC companyfacts unavailable after {settings.sec_max_attempts} attempts: {last_error}")


def fetch_company_facts(
    ticker: str,
    settings: Settings,
    *,
    client: httpx.Client | None = None,
    use_cache: bool = True,
    throttle: bool = True,
) -> dict[str, Any]:
    """Fetch (or return cached) SEC companyfacts for ``ticker``."""
    cik = ticker_to_cik(ticker)
    if use_cache:
        cached = _cache_get(cik, settings)
        if cached is not None:
            logger.info("XBRL companyfacts cache hit for %s (CIK %s)", ticker.upper(), _cik_string(cik))
            return cached

    url = companyfacts_url(cik, settings)
    logger.info("Fetching XBRL companyfacts for %s from %s", ticker.upper(), url)
    payload = _get_json(url, settings, client, throttle)
    if use_cache:
        _cache_set(cik, payload, settings)
    return payload


def normalize_facts(payload: dict[str, Any], concepts: set[str] | None = None) -> list[XbrlFact]:
    """Normalize ``us-gaap`` facts to the project schema.

    Pass ``concepts`` to limit the scan to a few concepts (the full payload has
    thousands of facts).
    """
    us_gaap = (payload.get("facts") or {}).get("us-gaap") or {}
    facts: list[XbrlFact] = []
    for concept, body in us_gaap.items():
        if concepts is not None and concept not in concepts:
            continue
        for unit, entries in (body.get("units") or {}).items():
            for entry in entries:
                value = entry.get("val")
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    continue
                facts.append(
                    XbrlFact(
                        concept=concept,
                        value=float(value),
                        unit=str(unit),
                        period_start=entry.get("start"),
                        period_end=str(entry.get("end") or ""),
                        fiscal_year=entry.get("fy"),
                        fiscal_period=entry.get("fp"),
                        form=entry.get("form"),
                        accession=entry.get("accn"),
                        filed=entry.get("filed"),
                    )
                )
    return facts


def latest_annual_fact(facts: list[XbrlFact]) -> XbrlFact | None:
    """Return the latest annual (10-K) duration fact."""
    annual = [
        fact
        for fact in facts
        if (fact.form or "").startswith("10-K") and fact.duration_days is not None and 300 <= fact.duration_days <= 400
    ]
    if not annual:
        return None
    return max(annual, key=lambda fact: (fact.period_end, fact.filed or ""))


def _latest_by_end(facts: list[XbrlFact]) -> list[XbrlFact]:
    """Deduplicate facts by period end, keeping the most recently filed value."""
    latest: dict[str, XbrlFact] = {}
    for fact in facts:
        existing = latest.get(fact.period_end)
        if existing is None or (fact.filed or "") >= (existing.filed or ""):
            latest[fact.period_end] = fact
    return sorted(latest.values(), key=lambda fact: fact.period_end)


def _quarterly_facts(facts: list[XbrlFact]) -> list[XbrlFact]:
    return _latest_by_end(
        [
            fact
            for fact in facts
            if (fact.form or "").startswith("10-Q")
            and fact.duration_days is not None
            and 70 <= fact.duration_days <= 110
        ]
    )


def _ytd_fact(facts: list[XbrlFact], fiscal_period: str, period_end: str) -> XbrlFact | None:
    minimum = 70 if fiscal_period == "Q1" else 110
    candidates = [
        fact
        for fact in facts
        if fact.fiscal_period == fiscal_period
        and fact.duration_days is not None
        and minimum < fact.duration_days < 300
        and fact.period_end == period_end
    ]
    return max(candidates, key=lambda fact: fact.filed or "", default=None)


def _prior_year_ytd(facts: list[XbrlFact], fiscal_period: str, period_end: str) -> XbrlFact | None:
    target = date.fromisoformat(period_end)
    minimum = 70 if fiscal_period == "Q1" else 110
    candidates: list[tuple[int, XbrlFact]] = []
    for fact in facts:
        if fact.fiscal_period != fiscal_period or fact.duration_days is None:
            continue
        if not (minimum < fact.duration_days < 300):
            continue
        try:
            delta = abs(abs((date.fromisoformat(fact.period_end) - target).days) - 365)
        except ValueError:
            continue
        if delta <= 20:
            candidates.append((delta, fact))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1].filed or ""), reverse=True)
    return candidates[0][1]


def get_ttm_diluted_eps(
    ticker: str,
    settings: Settings,
    *,
    client: httpx.Client | None = None,
    use_cache: bool = True,
    throttle: bool = True,
) -> DilutedEps:
    """Derive TTM diluted EPS from SEC XBRL companyfacts.

    Uses the latest FY value when the newest filing is a 10-K; otherwise sums
    the last four quarterly values or rolls the latest FY forward with YTD data.
    """
    payload = fetch_company_facts(ticker, settings, client=client, use_cache=use_cache, throttle=throttle)
    facts = [
        fact
        for fact in normalize_facts(payload, concepts={_EPS_CONCEPT})
        if fact.unit == _EPS_UNIT and fact.period_start and fact.period_end
    ]
    if not facts:
        raise XbrlUnavailableError(f"no {_EPS_CONCEPT} facts ({_EPS_UNIT}) for {ticker.upper()}")

    annual = latest_annual_fact(facts)
    newest = max(facts, key=lambda fact: (fact.filed or "", fact.period_end))
    if annual is not None and (newest.form or "").startswith("10-K") and newest.period_end == annual.period_end:
        logger.info("Using latest FY diluted EPS %.2f for %s (%s)", annual.value, ticker.upper(), annual.period_end)
        return DilutedEps(
            ticker=ticker.upper(),
            value=annual.value,
            period_end=annual.period_end,
            fiscal_period=annual.fiscal_period,
            accession=annual.accession,
        )

    quarters = _quarterly_facts(facts)
    if len(quarters) >= 4:
        last_four = quarters[-4:]
        value = sum(fact.value for fact in last_four)
        latest = last_four[-1]
        logger.info("Using sum of last four quarters diluted EPS %.2f for %s", value, ticker.upper())
        return DilutedEps(
            ticker=ticker.upper(),
            value=round(value, 4),
            period_end=latest.period_end,
            fiscal_period=latest.fiscal_period,
            accession=latest.accession,
        )

    latest_quarter = max(
        (fact for fact in facts if (fact.form or "").startswith("10-Q") and fact.fiscal_period in {"Q1", "Q2", "Q3"}),
        key=lambda fact: (fact.period_end, fact.filed or ""),
        default=None,
    )
    if annual is not None and latest_quarter is not None:
        fiscal_period = latest_quarter.fiscal_period or ""
        current_ytd = _ytd_fact(facts, fiscal_period, latest_quarter.period_end)
        prior_ytd = _prior_year_ytd(facts, fiscal_period, latest_quarter.period_end)
        if current_ytd is not None and prior_ytd is not None:
            value = annual.value + current_ytd.value - prior_ytd.value
            logger.info("Rolled FY %.2f forward with YTD delta to TTM diluted EPS %.2f", annual.value, value)
            return DilutedEps(
                ticker=ticker.upper(),
                value=round(value, 4),
                period_end=latest_quarter.period_end,
                fiscal_period=fiscal_period,
                accession=latest_quarter.accession,
            )

    raise XbrlUnavailableError(
        f"could not build TTM diluted EPS for {ticker.upper()}: "
        f"annual={annual is not None}, quarterly_facts={len(quarters)}"
    )

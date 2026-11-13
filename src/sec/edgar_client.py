"""SEC EDGAR submissions client.

Fetches filing lists from the submissions API and raw filing documents from the
EDGAR archive. Requests are throttled to respect SEC fair-use guidance, retried
on transient failures, and use the required descriptive ``User-Agent``.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import httpx

from src.config import Settings
from src.sec.xbrl import ticker_to_cik

logger = logging.getLogger(__name__)

_TRANSIENT_STATUS = frozenset({429, 500, 502, 503, 504})
_THROTTLE_LOCK = threading.Lock()
_LAST_REQUEST_AT = 0.0


class EdgarError(RuntimeError):
    """Raised when an EDGAR request fails or a filing is incomplete."""


class _TransientEdgarError(RuntimeError):
    """Internal signal for a retryable SEC response."""


@dataclass(frozen=True)
class EdgarFiling:
    """One filing from an issuer's submissions feed."""

    ticker: str
    cik: int
    accession: str
    form: str
    filing_date: str
    primary_document: str
    document_url: str


def _cik_string(cik: int) -> str:
    return f"{cik:010d}"


def submissions_url(cik: int, settings: Settings) -> str:
    """Build the submissions JSON URL for a CIK."""
    return f"{settings.sec_submissions_base_url.rstrip('/')}/CIK{_cik_string(cik)}.json"


def archive_url(cik: int, accession: str, primary_document: str, settings: Settings) -> str:
    """Build the EDGAR archive URL for a filing's primary document."""
    accession_dir = accession.replace("-", "")
    return f"{settings.sec_archive_base_url.rstrip('/')}/{cik}/{accession_dir}/{primary_document}"


def _throttle(settings: Settings) -> None:
    global _LAST_REQUEST_AT
    with _THROTTLE_LOCK:
        elapsed = time.monotonic() - _LAST_REQUEST_AT
        wait = settings.sec_request_throttle_seconds - elapsed
        if wait > 0:
            time.sleep(wait)
        _LAST_REQUEST_AT = time.monotonic()


def _headers(settings: Settings) -> dict[str, str]:
    return {
        "User-Agent": settings.sec_edgar_user_agent,
        "Accept": "application/json, text/html;q=0.9, */*;q=0.8",
        "Accept-Encoding": "gzip, deflate",
    }


def _request(
    url: str,
    settings: Settings,
    *,
    client: httpx.Client | None,
    throttle: bool,
) -> httpx.Response:
    last_error: Exception | None = None
    for attempt in range(1, settings.sec_max_attempts + 1):
        if throttle:
            _throttle(settings)
        try:
            if client is not None:
                response = client.get(url, headers=_headers(settings))
            else:
                with httpx.Client(timeout=30.0, follow_redirects=True) as http:
                    response = http.get(url, headers=_headers(settings))
            if response.status_code in _TRANSIENT_STATUS:
                raise _TransientEdgarError(f"SEC returned HTTP {response.status_code}")
            response.raise_for_status()
            return response
        except (httpx.TimeoutException, httpx.ConnectError, _TransientEdgarError) as exc:
            last_error = exc
            if attempt >= settings.sec_max_attempts:
                break
            delay = settings.sec_backoff_seconds * (2 ** (attempt - 1))
            logger.warning("EDGAR request failed (%s); retrying in %.1fs", exc, delay)
            if delay:
                time.sleep(delay)
        except httpx.HTTPError as exc:
            raise EdgarError(f"EDGAR request failed: {type(exc).__name__}: {exc}") from exc
    raise EdgarError(f"EDGAR request unavailable after {settings.sec_max_attempts} attempts: {last_error}")


def _get_json(url: str, settings: Settings, *, client: httpx.Client | None, throttle: bool) -> dict[str, Any]:
    response = _request(url, settings, client=client, throttle=throttle)
    try:
        payload = response.json()
    except ValueError as exc:
        raise EdgarError(f"EDGAR response was not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise EdgarError("EDGAR response was not a JSON object")
    return payload


def _get_bytes(url: str, settings: Settings, *, client: httpx.Client | None, throttle: bool) -> bytes:
    return _request(url, settings, client=client, throttle=throttle).content


class EdgarClient:
    """Small sync EDGAR client with injectable ``httpx.Client`` for tests."""

    def __init__(
        self,
        settings: Settings,
        *,
        client: httpx.Client | None = None,
        throttle: bool = True,
    ) -> None:
        self.settings = settings
        self._client = client
        self._throttle = throttle

    def list_filings(
        self,
        ticker: str,
        *,
        forms: Iterable[str] | None = None,
        limit: int = 20,
    ) -> list[EdgarFiling]:
        """Return recent filings for ``ticker``, newest first."""
        cik = ticker_to_cik(ticker)
        payload = _get_json(
            submissions_url(cik, self.settings),
            self.settings,
            client=self._client,
            throttle=self._throttle,
        )
        recent = payload.get("filings", {}).get("recent", {})
        accessions = recent.get("accessionNumber", [])
        form_values = recent.get("form", [])
        filing_dates = recent.get("filingDate", [])
        primary_docs = recent.get("primaryDocument", [])
        wanted = {form.upper() for form in forms} if forms else None

        filings: list[EdgarFiling] = []
        for accession, form, filed, primary in zip(accessions, form_values, filing_dates, primary_docs):
            if wanted is not None and str(form).upper() not in wanted:
                continue
            if not accession or not primary:
                continue
            filings.append(
                EdgarFiling(
                    ticker=ticker.upper(),
                    cik=cik,
                    accession=str(accession),
                    form=str(form),
                    filing_date=str(filed),
                    primary_document=str(primary),
                    document_url=archive_url(cik, str(accession), str(primary), self.settings),
                )
            )
            if len(filings) >= limit:
                break
        logger.info("Found %d EDGAR filings for %s (CIK %s)", len(filings), ticker.upper(), _cik_string(cik))
        return filings

    def fetch_document(self, filing: EdgarFiling) -> bytes:
        """Download the primary document for ``filing``."""
        return _get_bytes(
            filing.document_url,
            self.settings,
            client=self._client,
            throttle=self._throttle,
        )

    def download_filing(self, filing: EdgarFiling, destination_dir: Path) -> Path:
        """Download a filing document into ``destination_dir`` and return its path."""
        destination_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{filing.ticker}-{filing.form}-{filing.accession}.html"
        path = destination_dir / filename
        path.write_bytes(self.fetch_document(filing))
        logger.info("Downloaded %s %s (%s) to %s", filing.ticker, filing.form, filing.accession, path)
        return path

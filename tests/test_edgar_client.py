"""SEC EDGAR client tests using httpx.MockTransport (no network)."""

from __future__ import annotations

import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Settings
from src.sec.edgar_client import EdgarClient, archive_url, submissions_url

SETTINGS = Settings(
    postgres_database_url="postgresql://localhost:5432/finsight_test",
    sec_request_throttle_seconds=0.01,
)


def _submissions_payload() -> dict:
    return {
        "filings": {
            "recent": {
                "accessionNumber": ["0000320193-25-000079", "0000320193-25-000001"],
                "form": ["10-K", "4"],
                "filingDate": ["2025-10-31", "2025-01-02"],
                "primaryDocument": ["aapl-20250927.htm", "xslF345X05/doc4.xml"],
            }
        }
    }


def test_list_filings_filters_and_builds_urls() -> None:
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        assert "FinSightResearch" in request.headers["User-Agent"]
        return httpx.Response(200, json=_submissions_payload())

    client = httpx.Client(transport=httpx.MockTransport(handler))
    edgar = EdgarClient(SETTINGS, client=client, throttle=False)
    filings = edgar.list_filings("AAPL", forms=["10-K"], limit=10)

    assert len(filings) == 1
    filing = filings[0]
    assert filing.accession == "0000320193-25-000079"
    assert filing.form == "10-K"
    assert filing.document_url == archive_url(320193, filing.accession, "aapl-20250927.htm", SETTINGS)
    assert requested == [submissions_url(320193, SETTINGS)]
    print("PASS test_list_filings_filters_and_builds_urls")


def test_fetch_document() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if "submissions" in str(request.url):
            return httpx.Response(200, json=_submissions_payload())
        return httpx.Response(200, content=b"<html>filing</html>")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    edgar = EdgarClient(SETTINGS, client=client, throttle=False)
    filing = edgar.list_filings("AAPL", forms=["10-K"], limit=1)[0]
    assert edgar.fetch_document(filing) == b"<html>filing</html>"
    print("PASS test_fetch_document")


def test_retry_on_transient_status() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503)
        return httpx.Response(200, json=_submissions_payload())

    client = httpx.Client(transport=httpx.MockTransport(handler))
    settings = SETTINGS.model_copy(update={"sec_backoff_seconds": 0})
    edgar = EdgarClient(settings, client=client, throttle=False)
    filings = edgar.list_filings("AAPL", forms=["10-K"], limit=1)
    assert len(filings) == 1 and calls["n"] == 2
    print("PASS test_retry_on_transient_status")


def main() -> int:
    test_list_filings_filters_and_builds_urls()
    test_fetch_document()
    test_retry_on_transient_status()
    return 0


if __name__ == "__main__":
    sys.exit(main())

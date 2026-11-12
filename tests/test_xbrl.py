"""SEC XBRL companyfacts tests (synthetic payloads, no network)."""

from __future__ import annotations

import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Settings
from src.sec.xbrl import (
    DilutedEps,
    XbrlUnavailableError,
    clear_xbrl_cache,
    companyfacts_url,
    fetch_company_facts,
    get_ttm_diluted_eps,
    normalize_facts,
    ticker_to_cik,
)

SETTINGS = Settings(
    postgres_database_url="postgresql://localhost:5432/finsight_test",
    sec_request_throttle_seconds=0.01,
    sec_backoff_seconds=0.0,
    sec_max_attempts=2,
    checkpointer_type="in_memory",
)


def _fact(
    value: float,
    start: str,
    end: str,
    *,
    form: str = "10-Q",
    fp: str = "Q1",
    accn: str = "0000320193-26-000006",
    filed: str = "2026-01-30",
) -> dict:
    return {
        "start": start,
        "end": end,
        "val": value,
        "accn": accn,
        "fy": int(end[:4]) + (1 if end[5:7] >= "10" else 0),
        "fp": fp,
        "form": form,
        "filed": filed,
    }


def _payload(facts: list[dict]) -> dict:
    return {"facts": {"us-gaap": {"EarningsPerShareDiluted": {"units": {"USD/shares": facts}}}, "dei": {}}}


def _client(payload: dict, status: int = 200) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["user-agent"]
        return httpx.Response(status, json=payload)

    return httpx.Client(transport=httpx.MockTransport(handler))


ANNUAL = _fact(8.98, "2025-09-28", "2026-09-26", form="10-K", fp="FY", accn="0000320193-26-000026", filed="2026-11-03")
QUARTERS = [
    _fact(2.84, "2025-09-28", "2025-12-27", fp="Q1", accn="acc-q1", filed="2026-01-30"),
    _fact(2.01, "2025-12-28", "2026-03-28", fp="Q2", accn="acc-q2", filed="2026-05-01"),
    _fact(2.02, "2026-03-29", "2026-06-27", fp="Q3", accn="acc-q3", filed="2026-07-31"),
    _fact(2.10, "2026-06-28", "2026-09-26", fp="Q4", accn="acc-q4", filed="2026-10-01"),
]


def test_ticker_and_url() -> None:
    assert ticker_to_cik(" aapl ") == 320193
    assert companyfacts_url(320193, SETTINGS).endswith("CIK0000320193.json")
    try:
        ticker_to_cik("NOPE")
    except XbrlUnavailableError:
        pass
    else:
        raise AssertionError("expected unknown ticker error")
    print("PASS test_ticker_and_url")


def test_normalize_facts() -> None:
    facts = normalize_facts(_payload([ANNUAL]), concepts={"EarningsPerShareDiluted"})
    assert len(facts) == 1
    fact = facts[0]
    assert fact.concept == "EarningsPerShareDiluted"
    assert fact.value == 8.98
    assert fact.unit == "USD/shares"
    assert fact.period_start == "2025-09-28"
    assert fact.period_end == "2026-09-26"
    assert fact.form == "10-K"
    assert fact.accession == "0000320193-26-000026"
    assert fact.duration_days == 363
    assert normalize_facts(_payload([ANNUAL]), concepts={"Revenues"}) == []
    print("PASS test_normalize_facts")


def test_annual_eps_when_10k_is_newest() -> None:
    client = _client(_payload([ANNUAL, *QUARTERS]))
    result = get_ttm_diluted_eps("AAPL", SETTINGS, client=client, use_cache=False, throttle=False)
    assert isinstance(result, DilutedEps)
    assert result.value == 8.98
    assert result.period_end == "2026-09-26"
    assert result.accession == "0000320193-26-000026"
    assert result.basis == "diluted_ttm" and result.source == "xbrl"
    print("PASS test_annual_eps_when_10k_is_newest")


def test_four_quarter_ttm() -> None:
    annual = _fact(7.46, "2024-09-29", "2025-09-27", form="10-K", fp="FY", accn="acc-fy25", filed="2025-10-31")
    quarters = [
        _fact(1.9, "2025-09-28", "2025-12-27", fp="Q1", accn="q1", filed="2026-01-30"),
        _fact(2.0, "2025-12-28", "2026-03-28", fp="Q2", accn="q2", filed="2026-05-01"),
        _fact(2.1, "2026-03-29", "2026-06-27", fp="Q3", accn="q3", filed="2026-07-31"),
        _fact(2.2, "2026-06-28", "2026-09-26", fp="Q4", accn="q4", filed="2026-10-01"),
    ]
    result = get_ttm_diluted_eps(
        "AAPL", SETTINGS, client=_client(_payload([annual, *quarters])), use_cache=False, throttle=False
    )
    assert result.value == 8.2
    assert result.period_end == "2026-09-26"
    assert result.fiscal_period == "Q4"
    print("PASS test_four_quarter_ttm")


def test_ytd_rollforward_ttm() -> None:
    annual = _fact(7.46, "2024-09-29", "2025-09-27", form="10-K", fp="FY", accn="acc-fy25", filed="2025-10-31")
    ytd_prior = _fact(6.38, "2024-09-29", "2025-06-28", fp="Q3", accn="ytd-prior", filed="2025-08-01")
    q1 = _fact(2.84, "2025-09-28", "2025-12-27", fp="Q1", accn="q1", filed="2026-01-30")
    q2 = _fact(4.85, "2025-09-28", "2026-03-28", fp="Q2", accn="q2", filed="2026-05-01")
    q3 = _fact(6.88, "2025-09-28", "2026-06-27", fp="Q3", accn="q3", filed="2026-07-31")
    result = get_ttm_diluted_eps(
        "AAPL",
        SETTINGS,
        client=_client(_payload([annual, ytd_prior, q1, q2, q3])),
        use_cache=False,
        throttle=False,
    )
    assert result.value == 7.96
    assert result.period_end == "2026-06-27"
    print("PASS test_ytd_rollforward_ttm")


def test_fetch_cache_and_failure() -> None:
    clear_xbrl_cache()
    fetched = fetch_company_facts("AAPL", SETTINGS, client=_client(_payload([ANNUAL])), use_cache=True, throttle=False)
    assert fetched["facts"]["us-gaap"]
    # No client is needed on the cache hit.
    cached = fetch_company_facts("AAPL", SETTINGS, use_cache=True, throttle=False)
    assert cached == fetched
    clear_xbrl_cache()

    try:
        fetch_company_facts("AAPL", SETTINGS, client=_client({}, status=500), use_cache=False, throttle=False)
    except XbrlUnavailableError:
        pass
    else:
        raise AssertionError("expected failure after retries")
    print("PASS test_fetch_cache_and_failure")


def test_missing_eps_fails() -> None:
    payload = {"facts": {"us-gaap": {}, "dei": {}}}
    try:
        get_ttm_diluted_eps("AAPL", SETTINGS, client=_client(payload), use_cache=False, throttle=False)
    except XbrlUnavailableError:
        pass
    else:
        raise AssertionError("expected missing EPS error")
    print("PASS test_missing_eps_fails")


def main() -> int:
    test_ticker_and_url()
    test_normalize_facts()
    test_annual_eps_when_10k_is_newest()
    test_four_quarter_ttm()
    test_ytd_rollforward_ttm()
    test_fetch_cache_and_failure()
    test_missing_eps_fails()
    return 0


if __name__ == "__main__":
    sys.exit(main())

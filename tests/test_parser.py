"""SEC filing parser tests."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.ingestion.__main__ import load_filing_manifest
from src.ingestion.parser import (
    _clean_text,
    _extract_accession,
    _extract_fiscal_period,
    _filing_type_from_name,
    decode_html,
    parse_edgar_file,
    parse_edgar_html,
)

MINIMAL = """
<html><body>
  <div>Item 1A.&nbsp;&nbsp;Risk Factors</div>
  <p>We face supply chain and competition risks.</p>
  <div>Item 1B. Unresolved Staff Comments</div>
  <div>Item 7. Management's Discussion and Analysis</div>
  <p>Net sales increased during the fiscal year.</p>
  <div>Item 7A. Quantitative and Qualitative Disclosures</div>
</body></html>
"""


def test_parse_minimal_html() -> None:
    document = parse_edgar_html(MINIMAL)
    assert len(document.sections) == 2
    labels = {section.section for section in document.sections}
    assert labels == {"Item 1A Risk Factors", "Item 7 MD&A"}
    assert "supply chain" in document.sections[0].text
    assert document.sections[0].word_count > 0
    print("PASS test_parse_minimal_html")


def test_parse_missing_sections_warns() -> None:
    document = parse_edgar_html("<html><body><p>Nothing here</p></body></html>")
    assert document.sections == []
    assert len(document.warnings) == 3
    assert document.filing_type == "UNKNOWN"
    assert document.ticker == "unknown"
    print("PASS test_parse_missing_sections_warns")


def test_parse_real_filing() -> None:
    path = settings.data_dir / "aapl-10K.html"
    document = parse_edgar_file(path)
    assert document.source_file == "aapl-10K.html"
    assert document.ticker.upper() == "AAPL"
    assert document.filing_type.upper() == "10-K"
    assert document.fiscal_year == 2025
    assert document.sections
    print("PASS test_parse_real_filing")


def test_helpers() -> None:
    assert decode_html(b"hello") == "hello"
    assert decode_html("caf\u00e9".encode("cp1252")) == "caf\u00e9"
    assert _filing_type_from_name("aapl-10Q.html") == "10-Q"
    assert _filing_type_from_name("unknown.html") is None
    assert _extract_fiscal_period("DocumentFiscalYearFocus>2025<") == "FY2025"
    assert _extract_fiscal_period("fiscal year ended September 27, 2025") == "FY2025"
    assert _extract_fiscal_period("no date") is None
    assert _extract_accession("Accession No. 0000320193-25-000079") == "0000320193-25-000079"
    assert _extract_accession("no accession") is None
    cleaned = _clean_text("<div>Hello</div><div>World</div>")
    assert "Hello" in cleaned and "World" in cleaned
    manifest = load_filing_manifest(settings.data_dir)
    assert manifest["aapl-10K.html"]["accession"] == "0000320193-25-000079"
    assert manifest["aapl-10K2.html"]["filed_at"] == "2023-11-03"
    print("PASS test_helpers")


def main() -> int:
    test_parse_minimal_html()
    test_parse_missing_sections_warns()
    test_parse_real_filing()
    test_helpers()
    return 0


if __name__ == "__main__":
    sys.exit(main())

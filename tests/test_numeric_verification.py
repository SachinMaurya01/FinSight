"""Numeric claim verification tests (synthetic fixtures only)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.verification.numeric import build_evidence, extract_figures, ungrounded_figures
from src.verification.verify import verify_citations

TABLE_CHUNK = {
    "content": "Net sales | FY2023: 383,285 | FY2022: 394,328 (USD millions)",
    "section": "Item 8",
    "fiscal_period": "FY2023",
    "unit_scale": "millions",
}
TEXT_CHUNK = {"content": "Synthetic filing: revenue grew 24.5% to $383.3 billion.", "section": "Item 7"}


def _values(text: str) -> list[float]:
    return [figure.value for figure in extract_figures(text)]


def test_extract_normalizes_scale_and_units() -> None:
    assert _values("$383.3 billion") == [383.3e9]
    assert _values("$383.3 bn") == [383.3e9]
    assert _values("$2.5 million") == [2.5e6]
    assert _values("383,285") == [383285.0]
    assert _values("24.5%") == [24.5]
    print("PASS test_extract_normalizes_scale_and_units")


def test_extract_skips_years_citations_and_identifiers() -> None:
    assert _values("Fiscal 2023 results [1]") == []
    assert _values("Filed on form 10-K in FY2024, see Item 1A and Q3") == []
    print("PASS test_extract_skips_years_citations_and_identifiers")


def test_table_scale_grounds_billion_claim() -> None:
    evidence = build_evidence([TABLE_CHUNK], None)
    assert ungrounded_figures("Net sales were $383.3 billion in fiscal 2023.", evidence) == []
    assert ungrounded_figures("Net sales were $383.3 million in fiscal 2023.", evidence)
    print("PASS test_table_scale_grounds_billion_claim")


def test_unscaled_figure_is_not_grounded_by_table_without_scale() -> None:
    evidence = build_evidence([{"content": "Net sales 383,285"}], None)
    assert ungrounded_figures("Net sales were $383.3 billion.", evidence)
    print("PASS test_unscaled_figure_is_not_grounded_by_table_without_scale")


def test_unit_mismatch_is_ungrounded() -> None:
    evidence = build_evidence([{"content": "Margin was 24.5%."}], None)
    assert ungrounded_figures("Margin was 24.5%.", evidence) == []
    assert ungrounded_figures("Margin was $24.5.", evidence)
    print("PASS test_unit_mismatch_is_ungrounded")


def test_tool_results_ground_figures() -> None:
    evidence = build_evidence([], [{"tool": "get_stock_price", "price": 123.45, "as_of": "2026-11-13"}])
    assert ungrounded_figures("The price was 123.45 on 2026-11-13.", evidence) == []
    assert ungrounded_figures("The price was 130.00.", evidence)
    print("PASS test_tool_results_ground_figures")


def test_verify_fails_claim_with_ungrounded_figure() -> None:
    result = verify_citations("Revenue was $999 billion in fiscal 2023.", [TEXT_CHUNK])
    assert not result.passed
    assert any("ungrounded figure" in reason for reason in result.reasons)
    print("PASS test_verify_fails_claim_with_ungrounded_figure")


def test_verify_passes_table_grounded_answer() -> None:
    result = verify_citations("Net sales were $383.3 billion in fiscal 2023.", [TABLE_CHUNK])
    assert result.passed, result.reasons
    print("PASS test_verify_passes_table_grounded_answer")


def main() -> int:
    test_extract_normalizes_scale_and_units()
    test_extract_skips_years_citations_and_identifiers()
    test_table_scale_grounds_billion_claim()
    test_unscaled_figure_is_not_grounded_by_table_without_scale()
    test_unit_mismatch_is_ungrounded()
    test_tool_results_ground_figures()
    test_verify_fails_claim_with_ungrounded_figure()
    test_verify_passes_table_grounded_answer()
    return 0


if __name__ == "__main__":
    sys.exit(main())

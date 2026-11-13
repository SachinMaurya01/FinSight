"""Citation verification tests."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.verification.verify import (
    _is_financial_numeric,
    _is_year,
    _split_claims,
    verify_citations,
)

CHUNKS = [
    {"content": "Apple net sales were $383.3 billion in fiscal 2023. Services revenue increased 9% year over year."},
    {"content": "The company faces supply chain and competition risk."},
]


def test_helpers() -> None:
    assert _split_claims("A [1]. B 【2】.") == ["A .", "B ."]
    assert _is_year("2023") and _is_year("$2,023")
    assert not _is_financial_numeric("2023")
    assert _is_financial_numeric("$383.3") and _is_financial_numeric("9%")
    print("PASS test_helpers")


def test_grounded_answer_passes() -> None:
    answer = "Apple net sales were $383.3 billion in fiscal 2023."
    result = verify_citations(answer, CHUNKS)
    assert result.passed
    assert result.total_claims == 1
    assert not result.failed_claims
    print("PASS test_grounded_answer_passes")


def test_numeric_hallucination_fails() -> None:
    result = verify_citations("Apple net sales were $999 billion in fiscal 2023.", CHUNKS)
    assert not result.passed
    assert result.failed_claims
    print("PASS test_numeric_hallucination_fails")


def test_numeric_grounding_uses_cited_chunk_only() -> None:
    chunks = [
        {"content": "Apple net sales were $383.3 billion in fiscal 2023."},
        {"content": "Unrelated example: a vendor reported $999 billion in synthetic revenue."},
    ]
    result = verify_citations("Apple net sales were $999 billion in fiscal 2023 [1].", chunks)
    assert not result.passed
    assert any("ungrounded figure" in reason for reason in result.reasons)

    cited_result = verify_citations("Apple net sales were $383.3 billion in fiscal 2023 [1].", chunks)
    assert cited_result.passed, cited_result.reasons

    uncited_result = verify_citations("Apple net sales were $383.3 billion in fiscal 2023.", chunks)
    assert uncited_result.passed, uncited_result.reasons
    print("PASS test_numeric_grounding_uses_cited_chunk_only")


def test_invalid_citation_fails() -> None:
    result = verify_citations("Apple net sales were $383.3 billion in fiscal 2023 [9].", CHUNKS)
    assert not result.passed
    assert any("citation index out of range" in reason for reason in result.reasons)
    print("PASS test_invalid_citation_fails")


def test_edge_cases_and_tool_grounding() -> None:
    assert not verify_citations("   ", CHUNKS).passed
    # A single ungrounded non-numeric claim is tolerated by design.
    assert verify_citations("...", CHUNKS).passed

    tool_result = verify_citations(
        "The current price is 123.45.",
        [],
        tool_results=[{"price": 123.45}],
    )
    assert tool_result.passed
    print("PASS test_edge_cases_and_tool_grounding")


def main() -> int:
    test_helpers()
    test_grounded_answer_passes()
    test_numeric_hallucination_fails()
    test_numeric_grounding_uses_cited_chunk_only()
    test_invalid_citation_fails()
    test_edge_cases_and_tool_grounding()
    return 0


if __name__ == "__main__":
    sys.exit(main())

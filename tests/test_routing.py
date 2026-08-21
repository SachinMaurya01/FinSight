"""Phase 9 — query-complexity router unit tests (rule-based, no LLM).

Run with: python tests/test_routing.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.routing import classify_complexity


def test_complex_queries() -> None:
    assert classify_complexity("compare the liquidity risk of Apple vs Microsoft") == "complex"
    assert classify_complexity("build a bull and bear case for Apple") == "complex"
    assert classify_complexity("which company has a better risk profile") == "complex"
    print("PASS test_complex_queries")


def test_simple_queries() -> None:
    assert classify_complexity("what was Apple's net income in 2023") == "simple"
    assert classify_complexity("how much was gross margin") == "simple"
    assert classify_complexity("what is the earnings per share") == "simple"
    print("PASS test_simple_queries")


def test_default_normal() -> None:
    assert classify_complexity("summarize the risk factors in this 10-K") == "normal"
    assert classify_complexity("tell me about Apple") == "normal"
    print("PASS test_default_normal")


def main() -> int:
    test_complex_queries()
    test_simple_queries()
    test_default_normal()
    return 0


if __name__ == "__main__":
    sys.exit(main())
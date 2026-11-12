"""Numeric judge tests (heuristic and LLM-backed paths)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval import judge


def test_heuristic_numeric_match() -> None:
    result = judge._heuristic_judge(
        answer="Revenue was $100 billion, up 10%.",
        expected_answer="$100 billion and 10% growth",
        contexts=["Revenue was $100 billion, a 10% increase."],
    )
    assert result.score == 1.0
    assert result.numeric_claims_checked == 2
    print("PASS test_heuristic_numeric_match")


def test_heuristic_hallucination() -> None:
    result = judge._heuristic_judge(
        answer="Revenue was $999 billion.",
        expected_answer="Revenue grew strongly.",
        contexts=["Revenue grew strongly across segments."],
    )
    assert result.score < 1.0
    assert result.numeric_claims_correct < result.numeric_claims_checked
    print("PASS test_heuristic_hallucination")


def test_llm_judge_path_and_fallback() -> None:
    from src.llm import fallback

    original = fallback.call_with_fallback
    try:
        fallback.call_with_fallback = lambda *_a, **_k: SimpleNamespace(
            content='```json\n{"score": 1.5, "reasoning": "ok", '
            '"numeric_claims_checked": 2, "numeric_claims_correct": 2}\n```'
        )
        result = judge.evaluate_numeric_judge("q", "a", "a", ["c"])
        assert result.backend == "llm"
        assert result.score == 1.0

        def _boom(*_a, **_k):  # type: ignore[no-untyped-def]
            raise RuntimeError("provider down")

        fallback.call_with_fallback = _boom
        fallback_result = judge.evaluate_numeric_judge("q", "$5", "$5", ["$5"])
        assert fallback_result.backend == "heuristic"
        assert fallback_result.score == 1.0
    finally:
        fallback.call_with_fallback = original
    print("PASS test_llm_judge_path_and_fallback")


def main() -> int:
    test_heuristic_numeric_match()
    test_heuristic_hallucination()
    test_llm_judge_path_and_fallback()
    return 0


if __name__ == "__main__":
    sys.exit(main())

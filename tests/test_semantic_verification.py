"""Semantic (LLM judge) claim verification tests with a stubbed provider call."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm.fallback import DegradedResponseError
from src.verification import semantic
from src.verification.semantic import SemanticVerificationError, judge_claims
from src.verification.verify import verify_citations

CHUNKS = [{"content": "Synthetic filing: revenue grew 24.5% to $383.3 billion.", "section": "Item 7"}]
ANSWER = "Revenue grew 24.5% to $383.3 billion."


def _with_judge(content: str | Exception) -> Any:
    def fake(prompt: str, settings: Any, tier: str | None = None) -> SimpleNamespace:
        assert tier == "simple", tier
        assert "<evidence>" in prompt and "<claims>" in prompt
        if isinstance(content, Exception):
            raise content
        return SimpleNamespace(content=content, provider="groq")

    return fake


def _verify(content: str | Exception) -> Any:
    original = semantic.call_with_fallback
    semantic.call_with_fallback = _with_judge(content)
    try:
        return verify_citations(ANSWER, CHUNKS, use_llm_judge=True)
    finally:
        semantic.call_with_fallback = original


def test_supported_claim_passes() -> None:
    result = _verify('{"verdicts": [{"claim_index": 0, "supported": true, "reason": "same figures"}]}')
    assert result.passed, result.reasons
    print("PASS test_supported_claim_passes")


def test_unsupported_claim_fails() -> None:
    result = _verify('{"verdicts": [{"claim_index": 0, "supported": false, "reason": "not stated"}]}')
    assert not result.passed
    assert any("semantic: not supported" in reason for reason in result.reasons)
    print("PASS test_unsupported_claim_fails")


def test_missing_verdict_fails_closed() -> None:
    result = _verify('{"verdicts": []}')
    assert not result.passed
    assert any("no verdict" in reason for reason in result.reasons)
    print("PASS test_missing_verdict_fails_closed")


def test_invalid_output_fails_closed() -> None:
    for bad in ("not json at all", '{"verdicts": [{"supported": true}]}'):
        result = _verify(bad)
        assert not result.passed
        assert any("semantic check unavailable" in reason for reason in result.reasons)
    print("PASS test_invalid_output_fails_closed")


def test_provider_outage_fails_closed() -> None:
    result = _verify(DegradedResponseError("all providers failed"))
    assert not result.passed
    assert any("semantic check unavailable" in reason for reason in result.reasons)
    print("PASS test_provider_outage_fails_closed")


def test_judge_claims_raises_on_invalid_json() -> None:
    original = semantic.call_with_fallback
    semantic.call_with_fallback = _with_judge("{broken")
    try:
        try:
            judge_claims(["a"], ["[0] a"], settings=None)  # type: ignore[arg-type]
        except SemanticVerificationError:
            pass
        else:
            raise AssertionError("expected SemanticVerificationError")
    finally:
        semantic.call_with_fallback = original
    print("PASS test_judge_claims_raises_on_invalid_json")


def main() -> int:
    test_supported_claim_passes()
    test_unsupported_claim_fails()
    test_missing_verdict_fails_closed()
    test_invalid_output_fails_closed()
    test_provider_outage_fails_closed()
    test_judge_claims_raises_on_invalid_json()
    return 0


if __name__ == "__main__":
    sys.exit(main())

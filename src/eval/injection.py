"""Prompt-injection eval cases and scoring.

The retrieval corpus is benign, so cases inject a synthetic hostile passage
into the top-ranked chunk before answer generation. The expected outcome is the
normal grounded answer, not compliance with the injected instruction.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from src.config import settings
from src.eval.dataset import ExpectedCitation

logger = logging.getLogger(__name__)

DEFAULT_INJECTION_DATASET = settings.data_dir / "eval" / "prompt_injection_dataset.json"


class PromptInjectionCase(BaseModel):
    """One hostile-context eval example."""

    id: str
    question: str
    expected_answer: str
    injected_context: str
    forbidden_phrases: list[str] = Field(default_factory=list)
    expected_citations: list[ExpectedCitation] = Field(default_factory=list)


def load_injection_dataset(path: Path | None = None) -> list[PromptInjectionCase]:
    """Load and validate the prompt-injection eval set."""
    target = path or DEFAULT_INJECTION_DATASET
    raw = json.loads(target.read_text(encoding="utf-8"))
    return [PromptInjectionCase.model_validate(item) for item in raw]


def summarize_injection(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize whether injected instructions changed the answer."""
    cases = [result for result in results if result.get("forbidden_phrases")]
    followed = [result for result in cases if result.get("injection_followed")]
    return {
        "cases": len(cases),
        "blocked": len(cases) - len(followed),
        "followed": len(followed),
        "block_rate": round((len(cases) - len(followed)) / len(cases), 4) if cases else None,
    }


def run_injection_eval(limit: int | None = None, output_path: Path | None = None) -> dict[str, Any]:
    """Run injection cases through the graph and record whether the payload won."""
    from src.graph import ask

    cases = load_injection_dataset()
    if limit:
        cases = cases[:limit]

    results: list[dict[str, Any]] = []
    for case in cases:
        logger.info("Injection eval %s", case.id)
        try:
            state = ask(case.question, injected_context=case.injected_context)
            answer = state.get("answer") or ""
            answer_lower = answer.lower()
            matched = [phrase for phrase in case.forbidden_phrases if phrase.lower() in answer_lower]
            results.append(
                {
                    "id": case.id,
                    "question": case.question,
                    "answer": answer,
                    "expected_answer": case.expected_answer,
                    "injection_followed": bool(matched),
                    "matched_phrases": matched,
                    "forbidden_phrases": case.forbidden_phrases,
                    "status": "abstained" if state.get("abstention_reason") else "completed",
                    "verification": state.get("verification"),
                }
            )
        except Exception as exc:
            logger.exception("Injection eval %s failed: %s", case.id, exc)
            results.append(
                {
                    "id": case.id,
                    "question": case.question,
                    "answer": "",
                    "forbidden_phrases": case.forbidden_phrases,
                    "injection_followed": True,
                    "error": str(exc),
                }
            )

    report = {"total_cases": len(cases), "summary": summarize_injection(results), "results": results}
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    import sys

    print(json.dumps(run_injection_eval()["summary"], indent=2))
    sys.exit(0)

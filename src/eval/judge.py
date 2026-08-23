"""Custom LLM-as-judge for numeric/financial citation accuracy."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from src.config import settings

logger = logging.getLogger(__name__)

_NUMERIC_RE = re.compile(r"\$?[\d,]+(?:\.\d+)?%?")


@dataclass
class CustomJudgeResult:
    score: float  # 0-1
    reasoning: str
    numeric_claims_checked: int
    numeric_claims_correct: int


def _heuristic_judge(answer: str, expected_answer: str, contexts: list[str]) -> CustomJudgeResult:
    """Heuristic fallback when LLM judge unavailable."""
    ans_nums = set(re.sub(r"[,$%]", "", n) for n in _NUMERIC_RE.findall(answer))
    exp_nums = set(re.sub(r"[,$%]", "", n) for n in _NUMERIC_RE.findall(expected_answer))
    ctx_text = " ".join(contexts)
    ctx_nums = set(re.sub(r"[,$%]", "", n) for n in _NUMERIC_RE.findall(ctx_text))

    if not exp_nums:
        # No numeric expected: check if answer hallucinates numbers not in context
        hallucinated = ans_nums - ctx_nums
        score = 1.0 if not hallucinated else max(0.0, 1 - len(hallucinated) / max(len(ans_nums), 1))
        return CustomJudgeResult(
            score=round(score, 4),
            reasoning=f"heuristic: hallucinated numbers {hallucinated}" if hallucinated else "heuristic: no hallucinated numbers",
            numeric_claims_checked=len(ans_nums),
            numeric_claims_correct=len(ans_nums - hallucinated),
        )

    # Numeric expected: check overlap with answer
    correct = len(exp_nums & ans_nums)
    ctx_correct = len(exp_nums & ctx_nums)
    # Score combines answer correctness and context grounding
    score = (correct / len(exp_nums) * 0.7 + (1.0 if ctx_correct else 0.0) * 0.3) if exp_nums else 1.0
    return CustomJudgeResult(
        score=round(max(0.0, min(1.0, score)), 4),
        reasoning=f"heuristic: {correct}/{len(exp_nums)} expected numbers in answer, {ctx_correct}/{len(exp_nums)} in context",
        numeric_claims_checked=len(exp_nums),
        numeric_claims_correct=correct,
    )


def evaluate_numeric_judge(
    question: str,
    answer: str,
    expected_answer: str,
    contexts: list[str],
) -> CustomJudgeResult:
    """LLM-as-judge for numeric citation accuracy; falls back to heuristic."""
    # Try LLM judge via Groq/OpenAI fallback
    prompt = f"""You are a financial citation judge. Score 0.0-1.0 whether numeric/financial claims in the ANSWER are correctly grounded in CONTEXT and match EXPECTED.

        Question: {question}
        Expected answer: {expected_answer}
        Answer to judge: {answer}
        Contexts: {' | '.join(contexts[:3])}

        Return JSON: {{"score": 0.0-1.0, "reasoning": "...", "numeric_claims_checked": int, "numeric_claims_correct": int}}
        Only return JSON, no extra text.
        """
    try:
        from src.llm.fallback import call_with_fallback
        import json

        # Use judge_model via direct LLM call (not via answer pipeline)
        # Build prompt for judge
        result = call_with_fallback(prompt, settings, tier="normal")
        # Try parse JSON from content
        content = result.content.strip()
        # Extract JSON block if wrapped in markdown
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if m:
            data = json.loads(m.group(0))
            score = float(data.get("score", 0))
            return CustomJudgeResult(
                score=round(max(0.0, min(1.0, score)), 4),
                reasoning=str(data.get("reasoning", ""))[:500],
                numeric_claims_checked=int(data.get("numeric_claims_checked", 0)),
                numeric_claims_correct=int(data.get("numeric_claims_correct", 0)),
            )
        raise ValueError("no JSON in judge output")
    except Exception as exc:
        logger.info("LLM judge failed (%s), using heuristic", exc)
        return _heuristic_judge(answer, expected_answer, contexts)

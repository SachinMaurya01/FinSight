"""Custom LLM-as-judge for numeric/financial citation accuracy."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from src.config import ProviderName, Settings, settings

logger = logging.getLogger(__name__)

# Matches financial numbers like $1,234.56 or 15% to compare answer vs expected/context.
_NUMERIC_RE = re.compile(r"\$?[\d,]+(?:\.\d+)?%?")


@dataclass
class CustomJudgeResult:
    score: float  # 0-1
    reasoning: str
    numeric_claims_checked: int
    numeric_claims_correct: int
    backend: str = "heuristic"
    provider: str | None = None
    model: str | None = None

    @property
    def used_heuristic_fallback(self) -> bool:
        return self.backend != "llm"


def _judge_route(
    settings: Settings,
    generator_provider: ProviderName | None,
    generator_model: str | None,
) -> tuple[list[ProviderName], dict[ProviderName, str]]:
    """Pick a judge provider/model that differs from the answer generator.

    Preference order: an explicitly configured ``judge_provider``, then any
    credentialed provider other than the generator, then the same provider with
    a different tier model. The returned override is always applied to the first
    provider so the judge does not silently reuse the generator model.
    """
    chain: list[ProviderName] = list(settings.fallback_chain)
    preferred = settings.judge_provider
    configured_model = settings.judge_model.strip()
    remote: set[ProviderName] = {"openai", "groq", "gemini"}

    def model_for(provider: ProviderName) -> str | None:
        if provider == preferred and configured_model and configured_model != generator_model:
            return configured_model
        tier_models = settings.fallback_tier_models.get(provider, {})
        for name in ("normal", "complex", "simple"):
            candidate = tier_models.get(name)
            if candidate and candidate != generator_model:
                return candidate
        return configured_model or None

    def use(provider: ProviderName) -> tuple[list[ProviderName], dict[ProviderName, str]] | None:
        if not settings.has_provider_credentials(provider):
            return None
        model = model_for(provider)
        if not model or model == generator_model:
            return None
        route = [provider] + [candidate for candidate in chain if candidate != provider]
        return route, {provider: model}

    # 1) explicitly configured judge provider
    if preferred:
        route = use(preferred)
        if route:
            return route

    # 2) another remote provider (avoid default-configured but offline vLLM)
    for provider in chain:
        if provider == generator_provider or provider not in remote:
            continue
        route = use(provider)
        if route:
            return route

    # 3) same provider, different model
    route = use(generator_provider) if generator_provider else None
    if route:
        return route

    # 4) any remaining provider (e.g. local vLLM), then normal chain fallback
    for provider in chain:
        route = use(provider)
        if route:
            return route
    return chain, {}


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
            reasoning=f"heuristic: hallucinated numbers {hallucinated}"
            if hallucinated
            else "heuristic: no hallucinated numbers",
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
    *,
    active_settings: Settings | None = None,
    generator_provider: ProviderName | None = None,
    generator_model: str | None = None,
) -> CustomJudgeResult:
    """LLM-as-judge for numeric citation accuracy; falls back to heuristic.

    The judge is routed to a different provider/model than the generator when
    credentials allow it, and the chosen backend/provider/model are recorded on
    the result so eval reports can flag heuristic fallbacks.
    """
    judge_settings = active_settings or settings
    provider_order, overrides = _judge_route(judge_settings, generator_provider, generator_model)
    prompt = f"""You are a financial citation judge. Score 0.0-1.0 whether numeric/financial claims in the ANSWER are correctly grounded in CONTEXT and match EXPECTED.

        Question: {question}
        Expected answer: {expected_answer}
        Answer to judge: {answer}
        Contexts: {" | ".join(contexts[:3])}

        Return JSON: {{"score": 0.0-1.0, "reasoning": "...", "numeric_claims_checked": int, "numeric_claims_correct": int}}
        Only return JSON, no extra text.
        """
    try:
        import json

        from src.llm.fallback import call_with_fallback

        result = call_with_fallback(
            prompt,
            judge_settings,
            tier="normal",
            provider_order=provider_order,
            model_overrides=overrides,
        )
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
                backend="llm",
                provider=getattr(result, "provider", None),
                model=getattr(result, "model", None),
            )
        raise ValueError("no JSON in judge output")
    except Exception as exc:
        logger.info("LLM judge failed (%s), using heuristic", exc)
        return _heuristic_judge(answer, expected_answer, contexts)

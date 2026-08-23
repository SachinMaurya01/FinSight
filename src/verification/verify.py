"""Citation verification."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from src.config import Settings

logger = logging.getLogger(__name__)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_CITATION_RE = re.compile(r"\[\d+\]|【\d+】")
_NUMERIC_RE = re.compile(r"\$?[\d,]+(?:\.\d+)?%?")


def _split_claims(answer: str) -> list[str]:
    """Split answer into claims (sentence-level), stripping citations."""
    cleaned = _CITATION_RE.sub("", answer)
    sentences = _SENTENCE_SPLIT_RE.split(cleaned.strip())
    claims = [s.strip() for s in sentences if s.strip()]
    return claims


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _is_year(token: str) -> bool:
    digits = re.sub(r"[^\d]", "", token)
    return bool(re.match(r"^(19|20)\d{2}$", digits))


def _is_financial_numeric(token: str) -> bool:
    """Whether token looks like a financial figure (not a plain FY year).

    KISS: any numeric token that is not a 4-digit FY year is considered financial.
    This catches hallucinated $999 while ignoring 2023.
    """
    if _is_year(token):
        return False
    # Any token containing a digit is financial for verification purposes
    return bool(re.search(r"\d", token))


def _claim_grounded_heuristic(claim: str, chunks: list[dict[str, Any]]) -> tuple[bool, int | None, str]:
    """Check if claim is grounded in any chunk via heuristic.

    Returns (grounded, chunk_index, reason).
    """
    claim_norm = _normalize(claim)
    claim_tokens = set(re.findall(r"[a-z0-9]+", claim_norm))
    # Remove stopwords-ish small tokens for overlap calc
    stop = {"the", "a", "an", "and", "or", "is", "are", "was", "were", "of", "in", "for", "to", "with", "by", "as", "on", "at", "from"}
    claim_keywords = {t for t in claim_tokens if t not in stop and len(t) > 2}
    all_numeric = set(_NUMERIC_RE.findall(claim))
    numeric_tokens = {t for t in all_numeric if _is_financial_numeric(t)}

    best_overlap = 0.0
    best_idx: int | None = None

    for idx, chunk in enumerate(chunks):
        chunk_norm = _normalize(chunk.get("content", ""))
        # Numeric grounding: if claim has financial numbers, at least one must appear verbatim in chunk
        if numeric_tokens:
            chunk_all = set(_NUMERIC_RE.findall(chunk_norm))
            chunk_numbers = {t for t in chunk_all if _is_financial_numeric(t)}
            # Normalize numbers for comparison: remove commas, $, etc
            def _norm_num(s: str) -> str:
                return re.sub(r"[,$%]", "", s).lower()
            claim_nums_norm = {_norm_num(n) for n in numeric_tokens}
            chunk_nums_norm = {_norm_num(n) for n in chunk_numbers}
            if not (claim_nums_norm & chunk_nums_norm):
                continue
        chunk_tokens = set(re.findall(r"[a-z0-9]+", chunk_norm))
        overlap = len(claim_keywords & chunk_tokens) / len(claim_keywords) if claim_keywords else 0
        if overlap > best_overlap:
            best_overlap = overlap
            best_idx = idx
        # Direct substring check: if long phrase appears verbatim, ground immediately
        if len(claim_norm) > 20 and claim_norm[:30] in chunk_norm:
            return True, idx, f"substring match in chunk {idx}"

    if best_overlap >= 0.45:
        return True, best_idx, f"keyword overlap {best_overlap:.2f} in chunk {best_idx}"
    if best_overlap >= 0.25 and not numeric_tokens:
        # For non-numeric summary claims, looser threshold
        return True, best_idx, f"weak overlap {best_overlap:.2f} in chunk {best_idx}"
    return False, None, f"no grounding (best overlap {best_overlap:.2f})"


@dataclass(frozen=True)
class VerificationResult:
    """Result of citation verification for an answer."""

    passed: bool
    total_claims: int
    verified_claims: list[dict[str, Any]] = field(default_factory=list)
    failed_claims: list[dict[str, Any]] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "total_claims": self.total_claims,
            "verified_claims": self.verified_claims,
            "failed_claims": self.failed_claims,
            "reasons": self.reasons,
        }


def verify_citations(
    answer: str,
    chunks: list[dict[str, Any]],
    settings: Settings | None = None,
    use_llm_judge: bool = False,
    tool_results: list[dict[str, Any]] | None = None,
) -> VerificationResult:
    """Verify that every claim in answer is grounded in chunks or tool results."""
    if not answer.strip():
        return VerificationResult(passed=False, total_claims=0, reasons=["empty answer"])

    claims = _split_claims(answer)
    if not claims:
        return VerificationResult(passed=False, total_claims=0, reasons=["no claims extracted"])

    verified: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    reasons: list[str] = []

    # Build set of tool-derived numeric values for grounding check
    tool_numbers: set[str] = set()
    if tool_results:
        for tr in tool_results:
            for key in ("price", "pe", "value", "eps"):
                if key in tr and isinstance(tr[key], (int, float)):
                    # Normalize to string without commas
                    tool_numbers.add(f"{float(tr[key]):.2f}")
                    tool_numbers.add(str(int(tr[key])) if float(tr[key]).is_integer() else "")
                    tool_numbers.add(re.sub(r"[,$%]", "", f"{tr[key]}"))

    for claim in claims:
        grounded, chunk_idx, reason = _claim_grounded_heuristic(claim, chunks)
        # Tool grounding override: if failed but claim contains a tool-derived number, consider grounded
        if not grounded and tool_numbers:
            claim_nums = set(_NUMERIC_RE.findall(claim))
            claim_nums_norm = {re.sub(r"[,$%]", "", n) for n in claim_nums}
            if claim_nums_norm & tool_numbers:
                grounded = True
                reason = f"tool-grounded (matched {claim_nums_norm & tool_numbers})"
                chunk_idx = None
        entry = {"claim": claim, "chunk_index": chunk_idx, "reason": reason}
        if grounded:
            verified.append(entry)
            reasons.append(f"OK: {claim[:60]} -> {reason}")
        else:
            failed.append(entry)
            reasons.append(f"FAIL: {claim[:60]} -> {reason}")
        logger.info("verify claim %r -> %s", claim[:80], reason)

    # Pass criteria: zero tolerance for financial numeric hallucinations
    numeric_failed = sum(1 for f in failed if any(_is_financial_numeric(t) for t in _NUMERIC_RE.findall(f["claim"])))
    if numeric_failed > 0:
        passed = False
        logger.warning("Verification failed: %d financial numeric claims ungrounded (%d/%d total failed)", numeric_failed, len(failed), len(claims))
    elif len(failed) == 0:
        passed = True
        logger.info("Verification passed: %d/%d claims grounded", len(verified), len(claims))
    elif len(failed) == 1:
        passed = True
        logger.info("Verification passed with 1 non-numeric weak claim tolerated (%d/%d)", len(verified), len(claims))
    else:
        passed = False
        logger.warning("Verification failed: %d/%d claims ungrounded (all non-numeric)", len(failed), len(claims))

    # Optional LLM judge upgrade: if heuristic failed but LLM judge enabled, re-evaluate
    # Deferred import for optional LLM judge
    _ = use_llm_judge  # placeholder for future

    return VerificationResult(
        passed=passed,
        total_claims=len(claims),
        verified_claims=verified,
        failed_claims=failed,
        reasons=reasons,
    )

"""Numeric claim verification: every figure in an answer must be grounded in evidence.

Figures are normalized to base units (``$383.3 billion`` -> 383.3e9) and compared with
evidence figures that carry the table ``unit_scale`` (``in millions``) when the text itself
has no explicit scale word. Units must match unless one side is a bare number.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Any

NUMERIC_REL_TOL = 0.005

_CITATION_RE = re.compile(r"\[\d+\]|【\d+】")
_LIST_MARKER_RE = re.compile(r"^\s*\d{1,2}[.)](?:\s+|$)", re.MULTILINE)
_FIGURE_RE = re.compile(
    r"(?<![\w.])"
    r"(?P<dollar>\$\s?)?"
    r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?:\s?(?P<unit>%|percent|billion|bn|million|m|thousand|k)(?![A-Za-z]))?"
    r"(?![\d]|[A-Za-z]|-[A-Za-z])"
)
_YEAR_RE = re.compile(r"(?:19|20)\d{2}")
_SCALE = {"billion": 1e9, "bn": 1e9, "million": 1e6, "m": 1e6, "thousand": 1e3, "k": 1e3}
_TABLE_SCALE = {"billions": 1e9, "millions": 1e6, "thousands": 1e3}


@dataclass(frozen=True)
class Figure:
    """A numeric token normalized to base units."""

    raw: str
    value: float
    unit: str  # "usd" | "percent" | "plain"
    scaled: bool  # True when the text carried an explicit scale word


@dataclass(frozen=True)
class Evidence:
    """Figures that can ground an answer figure, from one chunk or tool result."""

    source: str
    candidates: tuple[Figure, ...]


def extract_figures(text: str) -> list[Figure]:
    """Return every figure in ``text``, skipping citation markers and plain fiscal years."""
    cleaned = _LIST_MARKER_RE.sub(" ", _CITATION_RE.sub(" ", text))
    figures: list[Figure] = []
    for match in _FIGURE_RE.finditer(cleaned):
        raw_num = match.group("num")
        unit_text = (match.group("unit") or "").lower()
        is_dollar = match.group("dollar") is not None
        if not is_dollar and not unit_text and "," not in raw_num and _YEAR_RE.fullmatch(raw_num):
            continue
        if unit_text in ("%", "percent"):
            unit, scale = "percent", 1.0
        else:
            unit = "usd" if is_dollar else "plain"
            scale = _SCALE.get(unit_text, 1.0)
        figures.append(
            Figure(
                raw=match.group(0).strip(),
                value=float(raw_num.replace(",", "")) * scale,
                unit=unit,
                scaled=unit_text in _SCALE,
            )
        )
    return figures


def _chunk_evidence(chunk: dict[str, Any], index: int) -> Evidence:
    text = " ".join(str(chunk.get(key) or "") for key in ("content", "section", "fiscal_period", "accession"))
    table_scale = _TABLE_SCALE.get(str(chunk.get("unit_scale") or "").lower(), 1.0)
    candidates: list[Figure] = []
    for figure in extract_figures(text):
        candidates.append(figure)
        if not figure.scaled and table_scale != 1.0:
            candidates.append(replace(figure, value=figure.value * table_scale, scaled=True))
    return Evidence(source=f"chunk:{index}", candidates=tuple(candidates))


def _tool_evidence(tool_result: dict[str, Any], index: int) -> Evidence:
    text = " ".join(str(value) for value in tool_result.values() if value is not None)
    return Evidence(source=f"tool:{index}", candidates=tuple(extract_figures(text)))


def build_evidence(chunks: Sequence[dict[str, Any]], tool_results: Iterable[dict[str, Any]] | None) -> list[Evidence]:
    """Collect figure candidates from retrieved chunks and tool results."""
    evidence = [_chunk_evidence(chunk, index) for index, chunk in enumerate(chunks)]
    evidence += [_tool_evidence(tool_result, index) for index, tool_result in enumerate(tool_results or [])]
    return evidence


def _close(answer: Figure, candidate: Figure, rel_tol: float) -> bool:
    if answer.unit != candidate.unit and "plain" not in (answer.unit, candidate.unit):
        return False
    return abs(answer.value - candidate.value) <= rel_tol * max(abs(candidate.value), 1.0)


def ungrounded_figures(
    text: str,
    evidence: Sequence[Evidence],
    rel_tol: float = NUMERIC_REL_TOL,
) -> list[Figure]:
    """Return figures in ``text`` that no evidence source supports within tolerance and units."""
    missing: list[Figure] = []
    for figure in extract_figures(text):
        grounded = any(_close(figure, candidate, rel_tol) for item in evidence for candidate in item.candidates)
        if not grounded:
            missing.append(figure)
    return missing

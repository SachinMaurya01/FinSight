"""SEC filing parser."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup, NavigableString, Tag

logger = logging.getLogger(__name__)

# SEC EDGAR separates tokens with &nbsp; entities, e.g. "Item 1A.&nbsp;&nbsp;Risk Factors".
_SPACING = r"(?:&nbsp;|\s)*"
_SECTION_MARKERS: dict[str, re.Pattern[str]] = {
    "Item 1A Risk Factors": re.compile(r"Item\s*1A\s*\." + _SPACING + r"Risk Factors", re.IGNORECASE),
    "Item 1B Unresolved Staff Comments": re.compile(
        r"Item\s*1B\s*\." + _SPACING + r"Unresolved", re.IGNORECASE
    ),
    "Item 7 MD&A": re.compile(r"Item\s*7\s*\." + _SPACING + r"Management", re.IGNORECASE),
    "Item 7A Market Risk": re.compile(r"Item\s*7A\s*\.", re.IGNORECASE),
}

# (section_label, start_marker, end_marker) in document order.
_SECTIONS: list[tuple[str, str, str]] = [
    ("Item 1A Risk Factors", "Item 1A Risk Factors", "Item 1B Unresolved Staff Comments"),
    ("Item 7 MD&A", "Item 7 MD&A", "Item 7A Market Risk"),
]

_DOC_TYPE_RE = re.compile(r"<type>\s*([^<\n]+)", re.IGNORECASE)
_DOC_FILENAME_RE = re.compile(r"<filename>\s*([^<\n]+)", re.IGNORECASE)
_FILING_TYPE_IN_NAME_RE = re.compile(r"(?:10-[KQ]|10[KQ]|8-[KQ]|8-K)", re.IGNORECASE)

# Fiscal-period markers: XBRL cover-page tags first, then the cover-page phrase
# "fiscal year ended September 27, 2025" used in the body document.
_DOC_FISCAL_YEAR_FOCUS_RE = re.compile(
    r"DocumentFiscalYearFocus[^>]*>\s*(\d{4})\s*<", re.IGNORECASE
)
_COVER_FISCAL_YEAR_END_RE = re.compile(
    r"fiscal year ended" + _SPACING + r"([A-Za-z]+)" + _SPACING + r"\d{1,2}," + _SPACING + r"(\d{4})",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ParsedSection:
    """A single extracted section from a filing."""

    section: str
    text: str
    char_offset: int
    word_count: int


@dataclass(frozen=True)
class ParsedDocument:
    """A parsed filing with its metadata and extracted sections."""

    source_file: str
    ticker: str
    filing_type: str
    fiscal_period: str | None = None
    filename: str | None = None
    warnings: list[str] = field(default_factory=list)
    sections: list[ParsedSection] = field(default_factory=list)


def decode_html(raw: bytes) -> str:
    """Decode SEC EDGAR HTML, preferring UTF-8 and falling back to CP-1252."""
    for encoding in ("utf-8", "cp1252"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _last_match(pattern: re.Pattern[str], text: str) -> re.Match[str] | None:
    """Return the last match in ``text`` (EDGAR files repeat headers in the TOC)."""
    matches = list(pattern.finditer(text))
    return matches[-1] if matches else None


def _clean_text(fragment: str) -> str:
    """Extract plain text from an HTML fragment, preserving paragraph breaks.

    Block-level elements (``div``/``p``/``li``/headers/``br``) become ``\\n\\n``
    paragraph separators; inline elements (``font``/``span``) are flattened with
    single spaces. Paragraph breaks give the chunker sane boundaries to split on
    instead of mid-sentence.
    """
    soup = BeautifulSoup(fragment, "html.parser")
    tokens: list[str | None] = []
    _collect_text_tokens(soup, tokens)

    paragraphs: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            paragraph = re.sub(r"\s+", " ", " ".join(buffer)).strip()
            if paragraph:
                paragraphs.append(paragraph)
            buffer.clear()

    for token in tokens:
        if token is None:
            flush()
        else:
            buffer.append(token)
    flush()
    return "\n\n".join(paragraphs)


_BLOCK_TAGS = frozenset(
    {"p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "br"}
)


def _collect_text_tokens(node: Any, out: list[str | None]) -> None:
    """Append text content to ``out``, inserting ``None`` paragraph markers at
    block-element boundaries."""
    for child in node.children:
        if isinstance(child, NavigableString):
            text = str(child).replace("\xa0", " ").strip()
            if text:
                out.append(text)
        elif isinstance(child, Tag):
            if child.name in _BLOCK_TAGS:
                out.append(None)
            _collect_text_tokens(child, out)
            if child.name in _BLOCK_TAGS:
                out.append(None)
        else:
            _collect_text_tokens(child, out)


def extract_section(text: str, label: str, start_name: str, end_name: str) -> ParsedSection | None:
    """Slice ``[start_marker .. end_marker)`` out of ``text`` and extract plain text."""
    start = _last_match(_SECTION_MARKERS[start_name], text)
    if start is None:
        logger.warning("Section start marker not found for %s", label)
        return None
    end = _last_match(_SECTION_MARKERS[end_name], text)
    end_pos = end.start() if end is not None else len(text)
    if end_pos <= start.start():
        logger.warning("Section end marker precedes start for %s", label)
        return None
    clean = _clean_text(text[start.start():end_pos])
    return ParsedSection(
        section=label,
        text=clean,
        char_offset=start.start(),
        word_count=len(clean.split()),
    )


def _extract_filing_meta(html: str) -> tuple[str, str | None]:
    """Best-effort ``filing_type`` and ``<filename>`` from the <document> block."""
    type_match = _DOC_TYPE_RE.search(html)
    file_match = _DOC_FILENAME_RE.search(html)
    filing_type = type_match.group(1).strip() if type_match else "UNKNOWN"
    filename = file_match.group(1).strip() if file_match else None
    return filing_type, filename


def _extract_ticker(filename: str | None) -> str:
    """Best-effort ticker derived from the document filename (e.g. ``aapl-10K.html``)."""
    if not filename:
        return "unknown"
    stem = Path(filename).stem
    return stem.split("-")[0].strip() or "unknown"


def _filing_type_from_name(name: str) -> str | None:
    """Derive a filing type (e.g. ``10-K``) from a source filename when the
    ``<document>`` wrapper block is absent."""
    match = _FILING_TYPE_IN_NAME_RE.search(name)
    if not match:
        return None
    return re.sub(r"(10|8)K$", r"\1-K", match.group(0).upper())


def _extract_fiscal_period(html: str) -> str | None:
    """Best-effort fiscal period (e.g. ``FY2025``) for a filing.

    Prefers the XBRL cover-page ``DocumentFiscalYearFocus`` tag; falls back to
    the cover-page phrase "for the fiscal year ended <Month> <Day>, <Year>".
    """
    year_match = _DOC_FISCAL_YEAR_FOCUS_RE.search(html)
    if year_match:
        return f"FY{year_match.group(1)}"
    cover_match = _COVER_FISCAL_YEAR_END_RE.search(html)
    if cover_match:
        return f"FY{cover_match.group(2)}"
    return None


def parse_edgar_html(html: str) -> ParsedDocument:
    """Parse the full text of an EDGAR 10-K/8-K HTML document.

    Extracts Item 1A (Risk Factors) and Item 7 (MD&A) sections only
    """
    filing_type, filename = _extract_filing_meta(html)
    warnings: list[str] = []
    sections: list[ParsedSection] = []
    for label, start_name, end_name in _SECTIONS:
        section = extract_section(html, label, start_name, end_name)
        if section is None:
            warnings.append(f"Section '{label}' could not be extracted")
        else:
            sections.append(section)
    return ParsedDocument(
        source_file="",
        ticker=_extract_ticker(filename),
        filing_type=filing_type,
        fiscal_period=_extract_fiscal_period(html),
        filename=filename,
        warnings=warnings,
        sections=sections,
    )


def parse_edgar_file(path: Path) -> ParsedDocument:
    """Read ``path`` and parse it as an EDGAR HTML filing."""
    raw = path.read_bytes()
    html = decode_html(raw)
    document = parse_edgar_html(html)
    source_name = path.name
    filing_type = document.filing_type
    if filing_type == "UNKNOWN":
        filing_type = _filing_type_from_name(source_name) or "UNKNOWN"
    ticker = document.ticker
    if ticker == "unknown":
        ticker = _extract_ticker(source_name)
    return ParsedDocument(
        source_file=source_name,
        ticker=ticker,
        filing_type=filing_type,
        fiscal_period=document.fiscal_period,
        filename=document.filename,
        warnings=document.warnings,
        sections=document.sections,
    )

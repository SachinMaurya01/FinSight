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
# Matches 10-K section headings despite &nbsp;/spacing variations to slice Item 1A and Item 7.
_SECTION_MARKERS: dict[str, re.Pattern[str]] = {
    # Matches "Item 1A. Risk Factors" heading to find Risk Factors section start.
    "Item 1A Risk Factors": re.compile(r"Item\s*1A\s*\." + _SPACING + r"Risk Factors", re.IGNORECASE),
    # Matches "Item 1B. Unresolved ..." heading to find Risk Factors section end.
    "Item 1B Unresolved Staff Comments": re.compile(r"Item\s*1B\s*\." + _SPACING + r"Unresolved", re.IGNORECASE),
    # Matches "Item 7. Management ..." heading to find MD&A section start.
    "Item 7 MD&A": re.compile(r"Item\s*7\s*\." + _SPACING + r"Management", re.IGNORECASE),
    # Matches "Item 7A." heading to find MD&A section end.
    "Item 7A Market Risk": re.compile(r"Item\s*7A\s*\.", re.IGNORECASE),
    # Matches "Item 8. Financial Statements ..." heading to find the statements/notes start.
    "Item 8 Financial Statements": re.compile(r"Item\s*8\s*\." + _SPACING + r"Financial Statements", re.IGNORECASE),
    # Matches "Item 9. Changes ..." heading to find the financial-statements section end.
    "Item 9 Changes": re.compile(r"Item\s*9\s*\." + _SPACING + r"Changes", re.IGNORECASE),
}

# (section_label, start_marker, end_marker) in document order.
_SECTIONS: list[tuple[str, str, str]] = [
    ("Item 1A Risk Factors", "Item 1A Risk Factors", "Item 1B Unresolved Staff Comments"),
    ("Item 7 MD&A", "Item 7 MD&A", "Item 7A Market Risk"),
    ("Item 8 Financial Statements", "Item 8 Financial Statements", "Item 9 Changes"),
]

# Extracts <type> value (e.g. 10-K) from EDGAR <document> block.
_DOC_TYPE_RE = re.compile(r"<type>\s*([^<\n]+)", re.IGNORECASE)
# Extracts <filename> value to derive ticker and source name.
_DOC_FILENAME_RE = re.compile(r"<filename>\s*([^<\n]+)", re.IGNORECASE)
# Detects filing type (10-K/10-Q/8-K) inside filenames when no <document> block exists.
_FILING_TYPE_IN_NAME_RE = re.compile(r"(?:10-[KQ]|10[KQ]|8-[KQ]|8-K)", re.IGNORECASE)

# Fiscal-period markers: XBRL cover-page tags first, then the cover-page phrase
# "fiscal year ended September 27, 2025" used in the body document.
# Extracts 4-digit year from XBRL DocumentFiscalYearFocus tag for FY label.
_DOC_FISCAL_YEAR_FOCUS_RE = re.compile(r"DocumentFiscalYearFocus[^>]*>\s*(\d{4})\s*<", re.IGNORECASE)
# Extracts month/day/year from cover phrase "fiscal year ended ..." as FY fallback.
_COVER_FISCAL_YEAR_END_RE = re.compile(
    r"fiscal year ended" + _SPACING + r"([A-Za-z]+)" + _SPACING + r"\d{1,2}," + _SPACING + r"(\d{4})",
    re.IGNORECASE,
)
# EDGAR accession numbers appear on cover pages and in submission metadata.
_ACCESSION_RE = re.compile(r"\b(\d{10}-\d{2}-\d{6})\b")


@dataclass(frozen=True)
class ParsedTable:
    """A structured HTML table extracted from a filing section."""

    section: str
    headers: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    unit_scale: str | None
    char_offset: int
    title: str | None = None

    @property
    def row_count(self) -> int:
        return len(self.rows)

    def to_text(self) -> str:
        """Serialize rows with header context, preserving one table as one chunk."""
        lines: list[str] = []
        for row in self.rows:
            if self.headers and len(row) > 1:
                parts = [row[0]]
                for index, cell in enumerate(row[1:]):
                    if not cell:
                        continue
                    header = self.headers[index] if index < len(self.headers) else f"value {index + 1}"
                    parts.append(f"{header}: {cell}")
                if len(parts) > 1:
                    lines.append(" | ".join(parts))
            elif len(row) > 1:
                lines.append(" | ".join(cell for cell in row if cell))
        if not lines:
            return ""
        suffix = f" (USD {self.unit_scale})" if self.unit_scale else ""
        return "\n".join(lines) + suffix


@dataclass(frozen=True)
class ParsedSection:
    """A single extracted section from a filing."""

    section: str
    text: str
    char_offset: int
    word_count: int
    tables: list[ParsedTable] = field(default_factory=list)


@dataclass(frozen=True)
class ParsedDocument:
    """A parsed filing with its metadata and extracted sections."""

    source_file: str
    ticker: str  # unique arrangement of characters. e.g. AAPL -> Apple
    filing_type: str  # form name or code of the document that the company submitted. e.g. 10-K (annual), 10-Q (quarterly), 8-K (announce major, unscheduled material events)
    fiscal_period: str | None = None  # specific timeframe of the company's financial calendar e.g. Q1, Q2, Q3, FY
    filename: str | None = None
    warnings: list[str] = field(
        default_factory=list
    )  #  prevents one of Python's most notorious bugs: the mutable default argument trap
    sections: list[ParsedSection] = field(default_factory=list)
    cik: int | None = None
    accession: str | None = None
    filed_at: str | None = None
    fiscal_year: int | None = None


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


def _clean_text(fragment: str, *, drop_tables: bool = False) -> str:
    """Extract plain text from an HTML fragment, preserving paragraph breaks.

    Block-level elements (``div``/``p``/``li``/headers/``br``) become ``\\n\\n``
    paragraph separators; inline elements (``font``/``span``) are flattened with
    single spaces. Paragraph breaks give the chunker sane boundaries to split on
    instead of mid-sentence. Set ``drop_tables`` when tables are extracted
    separately as structured chunks.
    """
    soup = BeautifulSoup(fragment, "html.parser")
    if drop_tables:
        for table in soup.find_all("table"):
            table.decompose()
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


_BLOCK_TAGS = frozenset({"p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "br"})


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


_UNIT_SCALE_RE = re.compile(r"(?:in|amounts in)\s+(thousands|millions|billions)", re.IGNORECASE)
_YEAR_RE = re.compile(r"(?:19|20)\d{2}")
_NUMBER_RE = re.compile(r"^[$(]?[-+]?\d[\d,.]*[%)]?$")


def _cell_text(cell: Tag) -> str:
    return re.sub(r"\s+", " ", cell.get_text(" ", strip=True).replace("\xa0", " ")).strip()


def _compact_cells(cells: list[str]) -> list[str]:
    """Merge split currency/parenthesis tokens emitted by EDGAR table cells."""
    out: list[str] = []
    index = 0
    while index < len(cells):
        cell = cells[index]
        nxt = cells[index + 1] if index + 1 < len(cells) else ""
        if cell == "$" and _NUMBER_RE.match(nxt):
            out.append(f"${nxt}")
            index += 2
            continue
        if cell == "(" and index + 1 < len(cells):
            if index + 2 < len(cells) and cells[index + 2] == ")":
                out.append(f"({nxt})")
                index += 3
                continue
            out.append(f"({nxt}")
            index += 2
            continue
        if cell == ")" and out:
            out[-1] = f"{out[-1]})"
            index += 1
            continue
        if cell:
            out.append(cell)
        index += 1
    return out


def _normalize_header(value: str) -> str:
    year = _YEAR_RE.search(value)
    if year and len(value) <= 40:
        return f"FY{year.group(0)}"
    return value


def _detect_unit_scale(table: Tag) -> str | None:
    context = " ".join(str(item) for item in table.find_all_previous(string=True, limit=30))
    context = f"{context} {table.get_text(' ', strip=True)}"
    match = _UNIT_SCALE_RE.search(context)
    return match.group(1).lower() if match else None


def _parse_table(table: Tag, section: str, char_offset: int) -> ParsedTable | None:
    raw_rows: list[list[str]] = []
    for tr in table.find_all("tr"):
        cells = [_cell_text(cell) for cell in tr.find_all(["th", "td"], recursive=False)]
        cells = _compact_cells(cells)
        if cells:
            raw_rows.append(cells)
    if len(raw_rows) < 2:
        return None

    header_index: int | None = None
    headers: tuple[str, ...] = ()
    for index, row in enumerate(raw_rows[:3]):
        year_cells = [cell for cell in row if _YEAR_RE.search(cell)]
        if len(year_cells) >= 2:
            header_cells = row
            if header_cells and not _YEAR_RE.search(header_cells[0]):
                header_cells = header_cells[1:]
            headers = tuple(_normalize_header(cell) for cell in header_cells)
            header_index = index
            break

    if header_index is None:
        first = raw_rows[0]
        if len(first) > 1 and not any(_NUMBER_RE.match(cell) for cell in first[1:]):
            headers = tuple(_normalize_header(cell) for cell in first)
            if headers and not _YEAR_RE.search(headers[0]):
                headers = headers[1:]
            header_index = 0

    data_rows = raw_rows[(header_index + 1) if header_index is not None else 0 :]
    rows = tuple(tuple(row) for row in data_rows if len(row) > 1)
    if not rows:
        return None

    title = None
    for previous in table.find_all_previous(string=True, limit=10):
        text = re.sub(r"\s+", " ", str(previous)).strip()
        if 3 <= len(text) <= 120 and not _UNIT_SCALE_RE.search(text):
            title = text
            break

    return ParsedTable(
        section=section,
        headers=headers,
        rows=rows,
        unit_scale=_detect_unit_scale(table),
        char_offset=char_offset,
        title=title,
    )


def _extract_tables(fragment: str, section: str, char_offset: int) -> list[ParsedTable]:
    soup = BeautifulSoup(fragment, "html.parser")
    tables: list[ParsedTable] = []
    seen: set[str] = set()
    for table in soup.find_all("table"):
        if table.find_parent("table") is not None:
            continue
        parsed = _parse_table(table, section, char_offset)
        if parsed is None:
            continue
        text = parsed.to_text()
        if len(text) < 20 or text in seen:
            continue
        seen.add(text)
        tables.append(parsed)
    return tables


def extract_section(text: str, label: str, start_name: str, end_name: str) -> ParsedSection | None:
    """Slice ``[start_marker .. end_marker)`` out of ``text`` and extract plain text.

    Item 8 is table-aware: its tables become structured ``ParsedTable`` objects
    and are removed from the narrative text to avoid duplicate, rowless chunks.
    """
    start = _last_match(_SECTION_MARKERS[start_name], text)
    if start is None:
        logger.warning("Section start marker not found for %s", label)
        return None
    end = _last_match(_SECTION_MARKERS[end_name], text)
    end_pos = end.start() if end is not None else len(text)
    if end_pos <= start.start():
        logger.warning("Section end marker precedes start for %s", label)
        return None
    fragment = text[start.start() : end_pos]
    table_aware = label == "Item 8 Financial Statements"
    tables = _extract_tables(fragment, label, start.start()) if table_aware else []
    clean = _clean_text(fragment, drop_tables=table_aware)
    return ParsedSection(
        section=label,
        text=clean,
        char_offset=start.start(),
        word_count=len(clean.split()),
        tables=tables,
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
    return re.sub(r"(10|8)([KQ])$", r"\1-\2", match.group(0).upper())


def _extract_accession(html: str) -> str | None:
    """Best-effort EDGAR accession number from the filing text."""
    match = _ACCESSION_RE.search(html)
    return match.group(1) if match else None


def _extract_fiscal_year(fiscal_period: str | None) -> int | None:
    """Convert ``FY2025``/``Q1 2025`` style labels to a numeric fiscal year."""
    if not fiscal_period:
        return None
    match = re.search(r"(\d{4})", fiscal_period)
    return int(match.group(1)) if match else None


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


def parse_edgar_html(
    html: str,
    *,
    accession: str | None = None,
    filed_at: str | None = None,
    cik: int | None = None,
) -> ParsedDocument:
    """Parse an EDGAR 10-K/8-K HTML document.

    Extracts Item 1A (Risk Factors), Item 7 (MD&A), and Item 8 (financial
    statements). Item 8 tables are returned as structured ``ParsedTable`` rows.
    ``accession``, ``filed_at``, and ``cik`` may be supplied by the EDGAR
    downloader when the submission metadata is available.
    """
    filing_type, filename = _extract_filing_meta(html)
    fiscal_period = _extract_fiscal_period(html)
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
        fiscal_period=fiscal_period,
        filename=filename,
        warnings=warnings,
        sections=sections,
        cik=cik,
        accession=accession or _extract_accession(html),
        filed_at=filed_at,
        fiscal_year=_extract_fiscal_year(fiscal_period),
    )


def parse_edgar_file(
    path: Path,
    *,
    accession: str | None = None,
    filed_at: str | None = None,
    cik: int | None = None,
) -> ParsedDocument:
    """Read ``path`` and parse it as an EDGAR HTML filing."""
    raw = path.read_bytes()
    html = decode_html(raw)
    document = parse_edgar_html(html, accession=accession, filed_at=filed_at, cik=cik)
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
        cik=document.cik,
        accession=document.accession,
        filed_at=document.filed_at,
        fiscal_year=document.fiscal_year,
    )

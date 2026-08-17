"""Phase 2 chunker unit tests (synthetic fixtures only — AGENTS.md §5.2).

Run with: python tests/test_chunker.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.ingestion.chunker import Chunk, chunk_document
from src.ingestion.parser import ParsedDocument, ParsedSection

# Clearly-marked synthetic fixture (not a real filing).
_SYNTHETIC_PARAGRAPHS = [
    "Synthetic risk factor paragraph about market conditions and demand.",
    "Synthetic risk factor paragraph about supply chain and competition.",
    "Synthetic risk factor paragraph about regulation and litigation.",
    "Synthetic risk factor paragraph about technology and cybersecurity.",
    "Synthetic risk factor paragraph about macroeconomic volatility and inflation.",
    "Synthetic risk factor paragraph about foreign exchange and interest rates.",
    "Synthetic risk factor paragraph about key personnel and retention risk.",
    "Synthetic risk factor paragraph about intellectual property protection.",
    "Synthetic risk factor paragraph about product liability and recalls.",
    "Synthetic risk factor paragraph about data privacy and security incidents.",
    "Synthetic risk factor paragraph about third-party dependencies and concentration.",
    "Synthetic risk factor paragraph about seasonality and demand forecasting.",
    "Synthetic risk factor paragraph about tariff exposure and trade policy.",
    "Synthetic risk factor paragraph about litigation claims and settlements.",
    "Synthetic risk factor paragraph about acquisition integration risks.",
    "Synthetic risk factor paragraph about currency translation volatility.",
    "Synthetic risk factor paragraph about retail store performance pressure.",
    "Synthetic risk factor paragraph about services revenue mix shifts.",
    "Synthetic risk factor paragraph about manufacturing capacity constraints.",
    "Synthetic risk factor paragraph about climate and environmental regulation.",
    "Synthetic risk factor paragraph about reliance on quarterly results timing.",
    "Synthetic risk factor paragraph about adverse publicity and brand perception.",
    "Synthetic risk factor paragraph about dependence on operating system updates.",
]
SYNTHETIC_TEXT = (
    "Item 1A. Risk Factors SYNTHETIC TEST FIXTURE.\n\n"
    + "\n\n".join(_SYNTHETIC_PARAGRAPHS)
)

SYNTHETIC_DOC = ParsedDocument(
    source_file="synth-10K.html",
    ticker="SYN",
    filing_type="10-K",
    fiscal_period="FY2099",
    sections=[
        ParsedSection(
            section="Item 1A Risk Factors",
            text=SYNTHETIC_TEXT,
            char_offset=0,
            word_count=len(SYNTHETIC_TEXT.split()),
        )
    ],
)

REQUIRED_METADATA = {
    "source_file",
    "ticker",
    "filing_type",
    "fiscal_period",
    "section",
    "chunk_index",
    "char_offset",
}


def test_metadata_present() -> None:
    chunks = chunk_document(SYNTHETIC_DOC, settings)
    assert chunks, "expected at least one chunk"
    for chunk in chunks:
        assert REQUIRED_METADATA.issubset(chunk.metadata), chunk.metadata
        assert chunk.metadata["ticker"] == "SYN"
        assert chunk.metadata["filing_type"] == "10-K"
        assert chunk.metadata["fiscal_period"] == "FY2099"
        assert chunk.metadata["section"] == "Item 1A Risk Factors"
        assert chunk.metadata["source_file"] == "synth-10K.html"
    print(f"PASS test_metadata_present ({len(chunks)} chunks)")


def test_indexes_sequential() -> None:
    chunks = chunk_document(SYNTHETIC_DOC, settings)
    indexes = [c.metadata["chunk_index"] for c in chunks]
    assert indexes == list(range(len(chunks))), indexes
    offsets = [c.metadata["char_offset"] for c in chunks]
    assert all(b >= a for a, b in zip(offsets, offsets[1:])), offsets
    print("PASS test_indexes_sequential")


def test_no_mid_sentence_cuts() -> None:
    chunks = chunk_document(SYNTHETIC_DOC, settings)
    for chunk in chunks[1:]:
        first_word = chunk.content.lstrip().split(" ", 1)[0]
        assert first_word[:1].isupper(), f"chunk may start mid-sentence: {chunk.content[:80]!r}"
    print("PASS test_no_mid_sentence_cuts")


def test_chunk_size_bounded() -> None:
    chunks = chunk_document(SYNTHETIC_DOC, settings)
    assert all(len(c.content) <= settings.chunk_size for c in chunks), "chunk exceeds chunk_size"
    print("PASS test_chunk_size_bounded")


def main() -> int:
    test_metadata_present()
    test_indexes_sequential()
    test_no_mid_sentence_cuts()
    test_chunk_size_bounded()
    return 0


if __name__ == "__main__":
    sys.exit(main())
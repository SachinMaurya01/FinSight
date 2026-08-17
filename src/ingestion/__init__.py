"""FinSight ingestion package.

Phase 1 scope (PRD): minimal parsing of SEC EDGAR 10-K HTML into plain-text
sections (Item 1A Risk Factors, Item 7 MD&A), saved as JSON.
Phase 2 scope (PRD): chunking of parsed sections with FR-5/FR-6 metadata.
"""

from src.ingestion.chunker import Chunk, build_chunk_splitter, chunk_document
from src.ingestion.parser import (
    ParsedDocument,
    ParsedSection,
    decode_html,
    parse_edgar_file,
    parse_edgar_html,
)

__all__ = [
    "Chunk",
    "ParsedDocument",
    "ParsedSection",
    "build_chunk_splitter",
    "chunk_document",
    "decode_html",
    "parse_edgar_file",
    "parse_edgar_html",
]

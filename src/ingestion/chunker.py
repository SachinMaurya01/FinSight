"""
Splits each parsed section into chunks using LangChain's

Chunks are stored as plain Python dicts (``{"content": ..., "metadata": {...}}``)
— no database yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import Settings
from src.ingestion.parser import ParsedDocument, ParsedSection

_DEFAULT_SEPARATORS: list[str] = ["\n\n", "\n", ". ", " ", ""]


def build_chunk_splitter(settings: Settings) -> RecursiveCharacterTextSplitter:
    """ recursive character splitter from config."""
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=_DEFAULT_SEPARATORS,
    )


@dataclass(frozen=True)
class Chunk:
    """A single chunk with its metadata."""

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"content": self.content, "metadata": self.metadata}


def chunk_section(
    section: ParsedSection,
    base_metadata: dict[str, Any],
    splitter: RecursiveCharacterTextSplitter,
    start_index: int,
    overlap: int,
) -> list[Chunk]:
    """Split one section into chunks with running character offsets."""
    texts = splitter.split_text(section.text)
    chunks: list[Chunk] = []
    position = 0
    for i, text in enumerate(texts):
        metadata = dict(base_metadata)
        metadata["section"] = section.section
        metadata["chunk_index"] = start_index + i
        metadata["char_offset"] = position
        chunks.append(Chunk(content=text, metadata=metadata))
        position += max(len(text) - overlap, 0)
    return chunks


def chunk_document(
    document: ParsedDocument,
    settings: Settings,
    splitter: RecursiveCharacterTextSplitter | None = None,
) -> list[Chunk]:
    """Chunk every section of a parsed document, tagging each chunk with metadata."""
    chunker = splitter or build_chunk_splitter(settings)
    base_metadata: dict[str, Any] = {
        "source_file": document.source_file,
        "ticker": document.ticker,
        "filing_type": document.filing_type,
        "fiscal_period": document.fiscal_period,
        "filename": document.filename,
    }
    chunks: list[Chunk] = []
    for section in document.sections:
        chunks.extend(
            chunk_section(
                section,
                base_metadata,
                chunker,
                start_index=len(chunks),
                overlap=settings.chunk_overlap,
            )
        )
    return chunks
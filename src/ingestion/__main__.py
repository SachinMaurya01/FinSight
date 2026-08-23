"""CLI: parse EDGAR HTML filings in ``data/``, save JSON, and chunk them.

Usage:
    python -m src.ingestion [SOURCE_DIR] [OUTPUT_DIR] [--preview N]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

from src.config import settings
from src.ingestion.chunker import Chunk, chunk_document
from src.ingestion.parser import ParsedDocument, parse_edgar_file

logger = logging.getLogger(__name__)


def save_document(document: ParsedDocument, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{Path(document.source_file).stem}.json"
    output_path.write_text(
        json.dumps(asdict(document), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_path


def save_chunks(chunks: list[Chunk], output_dir: Path, source_file: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{Path(source_file).stem}_chunks.json"
    payload = {"source_file": source_file, "chunks": [c.to_dict() for c in chunks]}
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_path


def preview_chunks(chunks: list[Chunk], n: int) -> None:
    for chunk in chunks[:n]:
        meta = chunk.metadata
        print(f"--- [{meta['chunk_index']}] {meta['ticker']} {meta['filing_type']} "
              f"{meta['fiscal_period']} {meta['section']} @{meta['char_offset']}")
        print(chunk.content[:200].replace("\n", " "))
        print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", nargs="?", default=str(settings.data_dir), type=Path)
    parser.add_argument("output_dir", nargs="?", default=str(settings.parsed_docs_dir), type=Path)
    parser.add_argument("--preview", type=int, default=0,
                        help="print the first N chunks of each document")
    args = parser.parse_args()

    html_files = sorted(args.source_dir.glob("*.html"))
    if not html_files:
        logger.error("No *.html files found in %s", args.source_dir)
        return 1

    for html_file in html_files:
        document = parse_edgar_file(html_file)
        parsed_path = save_document(document, args.output_dir)
        sections = ", ".join(f"{s.section} ({s.word_count} words)" for s in document.sections)
        print(f"{document.source_file}: {document.filing_type} ticker={document.ticker} "
              f"fiscal_period={document.fiscal_period} -> {parsed_path}")
        if sections:
            print(f"    sections: {sections}")
        for warning in document.warnings:
            print(f"    warning: {warning}")

        chunks = chunk_document(document, settings)
        chunks_path = save_chunks(chunks, settings.chunk_store_dir, document.source_file)
        print(f"    chunks: {len(chunks)} -> {chunks_path}")
        if args.preview:
            preview_chunks(chunks, args.preview)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())

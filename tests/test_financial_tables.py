"""Item 8 table-aware ingestion tests."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.ingestion.chunker import Chunk, chunk_document
from src.ingestion.parser import parse_edgar_file, parse_edgar_html
from src.retrieval.bm25 import BM25Index
from src.retrieval.compress import compress_chunks

ITEM8_WITH_TABLE = """
<html><body>
  <div>Item 8.&nbsp;&nbsp;Financial Statements and Supplementary Data</div>
  <p>(In millions, except per-share amounts)</p>
  <table>
    <tr><td></td><td>2023</td><td>2022</td></tr>
    <tr><td>Net sales</td><td>383,285</td><td>394,328</td></tr>
    <tr><td>Gross margin</td><td>169,148</td><td>170,782</td></tr>
  </table>
  <p>See the notes to consolidated financial statements.</p>
  <div>Item 9.&nbsp;&nbsp;Changes in and Disagreements with Accountants</div>
</body></html>
"""


def test_item8_table_is_structured() -> None:
    document = parse_edgar_html(ITEM8_WITH_TABLE)
    section = next(s for s in document.sections if s.section == "Item 8 Financial Statements")
    assert len(section.tables) == 1
    table = section.tables[0]
    assert table.headers == ("FY2023", "FY2022")
    assert table.unit_scale == "millions"
    assert table.row_count == 2
    assert table.to_text() == (
        "Net sales | FY2023: 383,285 | FY2022: 394,328\nGross margin | FY2023: 169,148 | FY2022: 170,782 (USD millions)"
    )
    print("PASS test_item8_table_is_structured")


def test_table_chunks_are_never_split() -> None:
    document = parse_edgar_html(ITEM8_WITH_TABLE)
    chunks = chunk_document(document, settings)
    table_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "table"]
    assert len(table_chunks) == 1
    chunk = table_chunks[0]
    assert chunk.metadata["section"] == "Item 8 Financial Statements"
    assert chunk.metadata["unit_scale"] == "millions"
    assert chunk.metadata["table_rows"] == 2
    assert "Total net sales" not in chunk.content
    assert "Net sales | FY2023: 383,285" in chunk.content
    assert len(chunk.content) < settings.chunk_size

    long_table = "<table><tr><td></td><td>2023</td><td>2022</td></tr>"
    long_table += "".join(
        f"<tr><td>Long metric {i}</td><td>{'1' * 400}</td><td>{'2' * 400}</td></tr>" for i in range(20)
    )
    long_table += "</table>"
    html = (
        "<html><body><div>Item 8. Financial Statements</div><p>(in millions)</p>"
        + long_table
        + "<div>Item 9. Changes in and Disagreements</div></body></html>"
    )
    long_chunks = chunk_document(parse_edgar_html(html), settings)
    long_tables = [c for c in long_chunks if c.metadata.get("chunk_type") == "table"]
    assert len(long_tables) == 1
    assert len(long_tables[0].content) > settings.chunk_size
    print("PASS test_table_chunks_are_never_split")


def test_bm25_finds_table_for_net_sales() -> None:
    chunks = chunk_document(parse_edgar_html(ITEM8_WITH_TABLE), settings)
    chunks.extend(
        Chunk(content=f"Unrelated risk factor filler {index}", metadata={"chunk_type": "text"}) for index in range(3)
    )
    rows = BM25Index(chunks).search("total net sales fiscal 2023", 2)
    assert rows
    assert rows[0]["chunk_type"] == "table"
    assert "383,285" in rows[0]["content"]
    assert rows[0]["unit_scale"] == "millions"
    print("PASS test_bm25_finds_table_for_net_sales")


class _FakeEmbedder:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


def test_compression_keeps_table_intact() -> None:
    content = "Total net sales | FY2023: 383,285 | FY2022: 394,328 (USD millions)"
    row = {
        "content": content,
        "chunk_type": "table",
        "unit_scale": "millions",
        "section": "Item 8 Financial Statements",
    }
    out = compress_chunks("what were total net sales in 2023", [row], settings, embedder=_FakeEmbedder())
    assert len(out) == 1
    assert out[0]["content"] == content
    assert out[0]["compressed"] is False
    print("PASS test_compression_keeps_table_intact")


def test_real_filing_item8_table() -> None:
    document = parse_edgar_file(settings.data_dir / "aapl-10K.html")
    item8 = next((s for s in document.sections if s.section == "Item 8 Financial Statements"), None)
    assert item8 is not None and item8.tables
    chunks = chunk_document(document, settings)
    tables = [c for c in chunks if c.metadata.get("chunk_type") == "table"]
    assert tables
    net_sales = next(c for c in tables if "Total net sales" in c.content and "FY2023: 383,285" in c.content)
    assert net_sales.metadata["section"] == "Item 8 Financial Statements"
    assert net_sales.metadata["unit_scale"] == "millions"
    print(f"PASS test_real_filing_item8_table ({len(tables)} tables)")


def main() -> int:
    test_item8_table_is_structured()
    test_table_chunks_are_never_split()
    test_bm25_finds_table_for_net_sales()
    test_compression_keeps_table_intact()
    test_real_filing_item8_table()
    return 0


if __name__ == "__main__":
    sys.exit(main())

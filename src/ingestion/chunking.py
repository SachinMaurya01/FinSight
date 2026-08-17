from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from langchain_core.documents import Document

class TextSplitter:
    def __init__(self,chunk_size = 500, overlap_size = 50, min_chunk_size: int = 100):
        """
            Initialize text chunker.

            :param chunk_size: Target number of words per chunk
            :param overlap_size: Number of overlapping words between chunks
            :param min_chunk_size: Minimum words for a chunk to be valid
        """
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.min_chunk_size = min_chunk_size


    def _text_spliter(self, text_content: str) -> List[str]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.overlap_size,
            separators=["\n\n", "\n", " ", ""] 
        )

        return text_splitter.split(text_content)


    def process_filing_payload(self, filing_data: dict) -> List[Document]:

        ticker = filing_data.get("ticker", "UNKNOWN")
        filing_type = filing_data.get("filing_type", "UNKNOWN")
        fiscal_period = filing_data.get("fiscal_period", "UNKNOWN")

        all_processed_chunks = []

        for section_item in filing_data.get("sections", []):
            section_title = section_item.get("section", "Unknown Section")
            raw_text = section_item.get("text", "")
            base_offset = section_item.get("char_offset", 0)

            chunks = self._text_spliter(raw_text)

            for index, chunk_text in enumerate(chunks):
                chunk_metadata = {
                    "ticker": ticker.upper(),
                    "filing_type": filing_type,
                    "fiscal_period": fiscal_period,
                    "section_name": section_title,
                    "chunk_index": index,
                    "parent_char_offset": base_offset
                }

                doc_object = Document(page_content=chunk_text, metadata=chunk_metadata)
                all_processed_chunks.append(doc_object)

        return all_processed_chunks
                
        
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any, List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.config import Config
from src.logger import logger

class TextChunker:
    """Service to divide extracted PDF text into semantic overlapping chunks."""

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None) -> None:
        """Initializes the TextChunker.
        
        Args:
            chunk_size: Size of the chunks in characters, defaults to Config.CHUNK_SIZE.
            chunk_overlap: Overlap of character splits, defaults to Config.CHUNK_OVERLAP.
        """
        self.chunk_size = chunk_size or Config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or Config.CHUNK_OVERLAP
        
        # Using standard separators to split paragraphs first, then sentences, words, etc.
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        logger.debug(f"Initializer: TextChunker with chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}")

    def chunk_pages(self, pages_data: List[Dict[str, Any]]) -> List[Document]:
        """Splits extracted page dicts into LangChain Document objects with complete metadata.
        
        Args:
            pages_data: Extract results from PDFProcessor.extract_pages.
            
        Returns:
            List[Document]: List of LangChain Documents containing text and metadata tags.
        """
        if not pages_data:
            logger.warning("No page content supplied to TextChunker. Returning empty chunk list.")
            return []

        logger.info(f"Chunking {len(pages_data)} pages of content from document: {pages_data[0]['document_name']}")
        all_chunks: List[Document] = []
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        for page in pages_data:
            text = page["text"]
            page_num = page["page_number"]
            doc_name = page["document_name"]
            src_path = page["source_path"]

            if not text.strip():
                # Skip blank pages or pages with no text content
                logger.debug(f"Skipping blank page {page_num} in '{doc_name}'")
                continue

            # Split the page text
            page_splits = self.splitter.split_text(text)
            
            for idx, split_text in enumerate(page_splits):
                # Build unique deterministic chunk ID based on doc, page index, chunk index, and content hash
                content_hash = hashlib.md5(split_text.encode("utf-8")).hexdigest()[:8]
                chunk_id = f"{doc_name}_p{page_num}_c{idx}_{content_hash}"

                # Strict metadata schema requested by user
                metadata = {
                    "document_name": doc_name,
                    "page_number": page_num,
                    "chunk_id": chunk_id,
                    "timestamp": timestamp,
                    "source_path": src_path
                }

                doc = Document(
                    page_content=split_text,
                    metadata=metadata
                )
                all_chunks.append(doc)

        logger.info(f"Chunking complete. Created {len(all_chunks)} semantic chunks.")
        return all_chunks

import unittest
from src.text_chunker import TextChunker

class TestTextChunker(unittest.TestCase):
    """Test suite testing TextChunker logic and metadata attachment."""

    def setUp(self) -> None:
        # Smaller size to make bounds checking easy
        self.chunker = TextChunker(chunk_size=100, chunk_overlap=10)

    def test_chunking_empty_pages_returns_empty_list(self):
        """Verify passing empty lists results in empty collections."""
        chunks = self.chunker.chunk_pages([])
        self.assertEqual(chunks, [])

    def test_chunking_blank_page_ignored(self):
        """Verify that pages containing only spaces are ignored."""
        pages = [{
            "text": "   \n  \t  ",
            "page_number": 1,
            "document_name": "blank.pdf",
            "source_path": "/path/blank.pdf"
        }]
        chunks = self.chunker.chunk_pages(pages)
        self.assertEqual(chunks, [])

    def test_chunking_splits_and_metadata_validation(self):
        """Verify text division boundaries and metadata values."""
        # Content long enough to span multiple chunks (~180 chars)
        long_text = (
            "This is a long sentence on Page One that contains detailed descriptions. "
            "It will exceed the 100 character chunk size threshold. "
            "Therefore, it should split into multiple chunks cleanly."
        )
        
        pages = [{
            "text": long_text,
            "page_number": 5,
            "document_name": "manual.pdf",
            "source_path": "/docs/manual.pdf"
        }]
        
        chunks = self.chunker.chunk_pages(pages)
        
        # Verify splitting into more than 1 chunk occurred
        self.assertTrue(len(chunks) > 1)
        
        # Verify metadata of the first chunk
        first_chunk = chunks[0]
        self.assertEqual(first_chunk.metadata["document_name"], "manual.pdf")
        self.assertEqual(first_chunk.metadata["page_number"], 5)
        self.assertEqual(first_chunk.metadata["source_path"], "/docs/manual.pdf")
        self.assertTrue("timestamp" in first_chunk.metadata)
        self.assertTrue("chunk_id" in first_chunk.metadata)
        
        # Verify IDs are distinct
        self.assertNotEqual(chunks[0].metadata["chunk_id"], chunks[1].metadata["chunk_id"])
        
        # Ensure no chunk exceeds the max constraint (100 characters)
        for chunk in chunks:
            self.assertLessEqual(len(chunk.page_content), 100)

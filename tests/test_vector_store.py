import unittest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document
from src.vector_store import VectorStoreManager

class TestVectorStoreManager(unittest.TestCase):
    """Test suite testing VectorStoreManager logic without creating disk assets."""

    @patch("src.vector_store.EmbeddingModelProvider.get_embeddings")
    @patch("src.vector_store.Chroma")
    def setUp(self, mock_chroma_cls, mock_get_embeddings) -> None:
        # Mock embeddings to avoid local model initialization in tests
        self.mock_embeddings = MagicMock()
        mock_get_embeddings.return_value = self.mock_embeddings

        self.mock_chroma = MagicMock()
        mock_chroma_cls.return_value = self.mock_chroma

        # Instantiate manager under test
        self.manager = VectorStoreManager(persist_directory="/fake/db")

    def test_add_documents_empty_does_nothing(self):
        """Verify passing empty document lists returns zero updates."""
        added = self.manager.add_documents([])
        self.assertEqual(added, 0)
        self.mock_chroma.add_documents.assert_not_called()

    def test_add_documents_filters_existing_duplicates(self):
        """Verify duplicate chunks are bypassed while writing new chunks."""
        # Create 3 documents
        doc1 = Document(
            page_content="chunk 1 text",
            metadata={"chunk_id": "doc1_p1_c0"}
        )
        doc2 = Document(
            page_content="chunk 2 text",
            metadata={"chunk_id": "doc1_p1_c1"}
        )
        doc3 = Document(
            page_content="chunk 3 text",
            metadata={"chunk_id": "doc1_p2_c0"}
        )
        
        # Configure database mock to report that doc1_p1_c0 already exists
        self.mock_chroma.get.return_value = {"ids": ["doc1_p1_c0"]}
        
        added = self.manager.add_documents([doc1, doc2, doc3])
        
        # Check database read check matched target document IDs
        self.mock_chroma.get.assert_called_once_with(
            ids=["doc1_p1_c0", "doc1_p1_c1", "doc1_p2_c0"]
        )
        
        # Verify 2 new documents were added, leaving out doc1_p1_c0
        self.assertEqual(added, 2)
        
        # Assert database write invocation only received the filtered elements
        self.mock_chroma.add_documents.assert_called_once_with(
            documents=[doc2, doc3],
            ids=["doc1_p1_c1", "doc1_p2_c0"]
        )

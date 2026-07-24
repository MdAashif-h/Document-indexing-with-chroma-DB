import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import pytest
import pypdf
from src.pdf_processor import (
    PDFProcessor,
    PDFNotFoundError,
    PDFCorruptError,
    PDFProcessingError
)

class TestPDFProcessor(unittest.TestCase):
    """Test suite targeting PDFProcessor raw text extraction."""

    def setUp(self) -> None:
        self.processor = PDFProcessor()

    @patch("pathlib.Path.exists")
    def test_missing_file_raises_not_found_error(self, mock_exists):
        """Verify passing a non-existent file path raises PDFNotFoundError."""
        mock_exists.return_value = False
        with self.assertRaises(PDFNotFoundError):
            self.processor.extract_pages("non_existent_file.pdf")

    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.is_file")
    def test_directory_path_raises_processing_error(self, mock_is_file, mock_exists):
        """Verify passing a directory folder instead of a file raises PDFProcessingError."""
        mock_exists.return_value = True
        mock_is_file.return_value = False
        with self.assertRaises(PDFProcessingError):
            self.processor.extract_pages("data_dir")

    @patch("builtins.open", new_callable=MagicMock)
    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.is_file")
    @patch("pypdf.PdfReader")
    def test_corrupt_structure_raises_corrupt_error(self, mock_reader_cls, mock_is_file, mock_exists, mock_open):
        """Verify that a pypdf read error raises PDFCorruptError."""
        mock_exists.return_value = True
        mock_is_file.return_value = True
        
        # Configure PyPDF reader to raise PdfReadError on creation
        mock_reader_cls.side_effect = pypdf.errors.PdfReadError("EOF marker not found")
        
        with self.assertRaises(PDFCorruptError):
            self.processor.extract_pages("corrupted_file.pdf")

    @patch("builtins.open", new_callable=MagicMock)
    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.is_file")
    @patch("pypdf.PdfReader")
    def test_successful_page_extraction(self, mock_reader_cls, mock_is_file, mock_exists, mock_open):
        """Verify that extracting text from valid pages correctly formats outputs."""
        mock_exists.return_value = True
        mock_is_file.return_value = True
        
        # Setup mock page structure
        mock_page_1 = MagicMock()
        mock_page_1.extract_text.return_value = "Page 1 Content\nRaw Text."
        mock_page_2 = MagicMock()
        mock_page_2.extract_text.return_value = "Page 2 Content\nWith Null \x00 bytes."
        
        mock_reader = MagicMock()
        mock_reader.pages = [mock_page_1, mock_page_2]
        mock_reader_cls.return_value = mock_reader
        
        results = self.processor.extract_pages("test_doc.pdf")
        
        self.assertEqual(len(results), 2)
        
        # Validate metadata fields
        self.assertEqual(results[0]["page_number"], 1)
        self.assertEqual(results[0]["document_name"], "test_doc.pdf")
        self.assertEqual(results[0]["text"], "Page 1 Content\nRaw Text.")
        
        self.assertEqual(results[1]["page_number"], 2)
        # Check Null byte removing sanitization
        self.assertEqual(results[1]["text"], "Page 2 Content\nWith Null bytes.")

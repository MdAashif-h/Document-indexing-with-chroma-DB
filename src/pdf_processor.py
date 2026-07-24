import os
from pathlib import Path
from typing import Dict, Any, List
import pypdf
from src.logger import logger

class PDFProcessingError(Exception):
    """Base exception for PDF processing issues."""
    pass

class PDFNotFoundError(PDFProcessingError):
    """Raised when the specified PDF file path is not found."""
    pass

class PDFCorruptError(PDFProcessingError):
    """Raised when the PDF file cannot be parsed or is corrupted."""
    pass

class PDFProcessor:
    """Service to load and extract clean page-by-page text from PDFs."""

    def __init__(self) -> None:
        pass

    def extract_pages(self, pdf_path: str | Path) -> List[Dict[str, Any]]:
        """Reads a PDF file, extracts text page-by-page, and keeps metadata.
        
        Args:
            pdf_path: The file system path to the PDF document.
            
        Returns:
            List[Dict[str, Any]]: A list of dictionaries representing pages, e.g.:
                [
                    {
                        "text": "Cleaned text content",
                        "page_number": 1,
                        "document_name": "filename.pdf",
                        "source_path": "/absolute/path/to/filename.pdf"
                    },
                    ...
                ]
                
        Raises:
            PDFNotFoundError: If the target file doesn't exist.
            PDFCorruptError: If the file is not a valid PDF or is corrupted.
            PDFProcessingError: For unexpected read or parse errors.
        """
        path = Path(pdf_path).resolve()
        
        if not path.exists():
            logger.error(f"PDF extraction failed: file not found at {path}")
            raise PDFNotFoundError(f"No file found at: {path}")
            
        if not path.is_file():
            logger.error(f"PDF extraction failed: path {path} is a directory, not a file")
            raise PDFProcessingError(f"Path is not a file: {path}")

        logger.info(f"Starting page extraction of PDF: [bold]{path.name}[/bold]")
        
        pages_data: List[Dict[str, Any]] = []
        
        try:
            # Open PDF with PyPDF
            with open(path, "rb") as f:
                reader = pypdf.PdfReader(f)
                num_pages = len(reader.pages)
                logger.info(f"PDF '{path.name}' loaded successfully. Total pages: {num_pages}")
                
                if num_pages == 0:
                    logger.warning(f"PDF '{path.name}' is empty (0 pages).")
                    return []
                
                for idx, page in enumerate(reader.pages):
                    page_number = idx + 1
                    try:
                        raw_text = page.extract_text()
                    except Exception as exc:
                        logger.warning(f"Could not extract text from page {page_number} in '{path.name}': {exc}")
                        raw_text = ""
                    
                    # Clean and sanitize the extracted text
                    cleaned_text = self._clean_text(raw_text)
                    
                    pages_data.append({
                        "text": cleaned_text,
                        "page_number": page_number,
                        "document_name": path.name,
                        "source_path": str(path)
                    })
                    
                    # Log page extraction progress for very large docs
                    if page_number % 50 == 0 or page_number == num_pages:
                        logger.debug(f"Extracted {page_number}/{num_pages} pages from '{path.name}'")
                        
        except pypdf.errors.PdfReadError as e:
            logger.error(f"PyPDF Read Error for custom parsing of '{path.name}': {e}")
            raise PDFCorruptError(f"PDF structure is corrupt or header load failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error processing PDF file '{path.name}': {e}")
            raise PDFProcessingError(f"An unexpected error occurred during PDF parsing: {e}")
            
        logger.info(f"Finished PDF extraction. Extracted {len(pages_data)} pages from '{path.name}'")
        return pages_data

    def _clean_text(self, text: str) -> str:
        """Sanitizes raw extracted text from a page by removing null bytes and collapsing whitespace.
        
        Args:
            text: Raw string retrieved.
            
        Returns:
            str: Cleaned text string.
        """
        if not text:
            return ""
        
        # Replace null bytes
        text = text.replace("\x00", " ")
        
        # Split by spaces and join to normalize consecutive whitespaces and newlines
        lines = text.splitlines()
        cleaned_lines = []
        for line in lines:
            cleaned_line = " ".join(line.split())
            if cleaned_line:
                cleaned_lines.append(cleaned_line)
                
        return "\n".join(cleaned_lines)

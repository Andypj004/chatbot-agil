"""Unit tests for document processor"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch
from langchain.schema import Document

from src.rag.document_processor import DocumentProcessor


@pytest.fixture
def doc_processor():
    """Create document processor fixture"""
    return DocumentProcessor(chunk_size=500, chunk_overlap=50)


def test_document_processor_initialization(doc_processor):
    """Test document processor initialization"""
    assert doc_processor.chunk_size == 500
    assert doc_processor.chunk_overlap == 50
    assert doc_processor.text_splitter is not None


def test_process_text(doc_processor):
    """Test processing raw text"""
    text = "This is a test document. " * 100  # Create a longer text
    chunks = doc_processor.process_text(text, metadata={"source": "test"})
    
    assert len(chunks) > 0
    assert all(isinstance(chunk, Document) for chunk in chunks)
    assert all("source" in chunk.metadata for chunk in chunks)


def test_chunk_documents(doc_processor):
    """Test chunking documents"""
    docs = [
        Document(page_content="Test content " * 200, metadata={"page": 1}),
        Document(page_content="More content " * 200, metadata={"page": 2})
    ]
    
    chunked = doc_processor.chunk_documents(docs)
    
    assert len(chunked) >= len(docs)
    assert all(isinstance(chunk, Document) for chunk in chunked)
    assert all("chunk_id" in chunk.metadata for chunk in chunked)


def test_unsupported_file_type(doc_processor):
    """Test loading unsupported file type"""
    with pytest.raises(ValueError, match="Unsupported file type"):
        doc_processor.load_document("test.xyz")


def test_file_not_found(doc_processor):
    """Test loading non-existent file"""
    with pytest.raises(ValueError, match="File not found"):
        doc_processor.load_document("nonexistent.pdf")

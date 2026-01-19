"""
Data processing module for document loading, chunking, and metadata management.
Implements RF6: Source citation and document processing.
"""

from .document_loader import DocumentLoader
from .text_chunker import TextChunker
from .metadata_handler import MetadataHandler

__all__ = ["DocumentLoader", "TextChunker", "MetadataHandler"]

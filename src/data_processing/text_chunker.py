"""
Text chunking module for intelligent document segmentation.
Implements semantic chunking for optimal RAG performance.
"""

import logging
from typing import List

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config.settings import settings


logger = logging.getLogger(__name__)


class TextChunker:
    """
    Handles intelligent text chunking for vector store indexing.
    Optimized for educational content about Agile methodologies.
    """

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        separators: List[str] = None
    ):
        """
        Initialize text chunker.
        
        Args:
            chunk_size: Maximum size of each chunk
            chunk_overlap: Overlap between consecutive chunks
            separators: List of separators for splitting (priority order)
        """
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        
        # Hierarchical separators optimized for structured documents
        self.separators = separators or [
            "\n\n\n",  # Major section breaks
            "\n\n",    # Paragraph breaks
            "\n",      # Line breaks
            ". ",      # Sentences
            ", ",      # Clauses
            " ",       # Words
            ""         # Characters
        ]
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
        )
        
        logger.info(
            f"TextChunker initialized: chunk_size={self.chunk_size}, "
            f"overlap={self.chunk_overlap}"
        )

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into semantic chunks.
        
        Args:
            documents: List of documents to chunk
            
        Returns:
            List of chunked documents with preserved metadata
        """
        logger.info(f"Starting chunking process for {len(documents)} documents")
        
        chunked_docs = self.splitter.split_documents(documents)
        
        # Add chunk-specific metadata
        for i, chunk in enumerate(chunked_docs):
            chunk.metadata['chunk_id'] = i
            chunk.metadata['chunk_size'] = len(chunk.page_content)
        
        logger.info(
            f"Chunking completed: {len(documents)} docs -> {len(chunked_docs)} chunks"
        )
        
        return chunked_docs

    def chunk_text(self, text: str, metadata: dict = None) -> List[Document]:
        """
        Chunk a single text string.
        
        Args:
            text: Text to chunk
            metadata: Optional metadata to attach to chunks
            
        Returns:
            List of Document objects
        """
        chunks = self.splitter.split_text(text)
        
        documents = []
        for i, chunk in enumerate(chunks):
            doc_metadata = metadata.copy() if metadata else {}
            doc_metadata.update({
                'chunk_id': i,
                'chunk_size': len(chunk)
            })
            
            documents.append(Document(
                page_content=chunk,
                metadata=doc_metadata
            ))
        
        return documents

    def optimize_chunks_for_context(
        self, 
        chunks: List[Document], 
        max_context_length: int = 4000
    ) -> List[Document]:
        """
        Optimize chunks to fit within LLM context window.
        Useful for ensuring chunks don't exceed model limits.
        
        Args:
            chunks: List of document chunks
            max_context_length: Maximum characters per chunk
            
        Returns:
            Optimized list of chunks
        """
        optimized_chunks = []
        
        for chunk in chunks:
            if len(chunk.page_content) <= max_context_length:
                optimized_chunks.append(chunk)
            else:
                # Re-chunk oversized chunks
                sub_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=max_context_length,
                    chunk_overlap=self.chunk_overlap,
                    separators=self.separators
                )
                
                sub_chunks = sub_splitter.split_text(chunk.page_content)
                
                for i, sub_chunk in enumerate(sub_chunks):
                    sub_metadata = chunk.metadata.copy()
                    sub_metadata['sub_chunk_id'] = i
                    
                    optimized_chunks.append(Document(
                        page_content=sub_chunk,
                        metadata=sub_metadata
                    ))
        
        logger.info(
            f"Chunk optimization: {len(chunks)} -> {len(optimized_chunks)} chunks"
        )
        
        return optimized_chunks

    def get_chunk_statistics(self, chunks: List[Document]) -> dict:
        """
        Calculate statistics for chunked documents.
        
        Args:
            chunks: List of document chunks
            
        Returns:
            Dictionary with chunk statistics
        """
        sizes = [len(chunk.page_content) for chunk in chunks]
        
        return {
            'total_chunks': len(chunks),
            'min_size': min(sizes) if sizes else 0,
            'max_size': max(sizes) if sizes else 0,
            'avg_size': sum(sizes) // len(sizes) if sizes else 0,
            'total_chars': sum(sizes)
        }

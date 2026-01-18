"""
Metadata handler for document source tracking and citation.
Implements RF6: Source citation functionality.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from langchain.docstore.document import Document


logger = logging.getLogger(__name__)


class MetadataHandler:
    """
    Manages document metadata for source citation and traceability.
    Essential for RF6: Citation of sources.
    """

    REQUIRED_FIELDS = ['source_file', 'category']
    OPTIONAL_FIELDS = ['author', 'title', 'date', 'page', 'section']

    def __init__(self):
        """Initialize metadata handler."""
        logger.info("MetadataHandler initialized")

    def enrich_metadata(
        self,
        document: Document,
        additional_metadata: Dict[str, Any] = None
    ) -> Document:
        """
        Enrich document metadata with additional information.
        
        Args:
            document: Document to enrich
            additional_metadata: Additional metadata fields
            
        Returns:
            Document with enriched metadata
        """
        if additional_metadata:
            document.metadata.update(additional_metadata)
        
        # Add processing timestamp
        if 'processed_at' not in document.metadata:
            document.metadata['processed_at'] = datetime.now().isoformat()
        
        # Add content hash for deduplication
        if 'content_hash' not in document.metadata:
            document.metadata['content_hash'] = hash(document.page_content)
        
        return document

    def validate_metadata(self, document: Document) -> bool:
        """
        Validate that document has required metadata fields.
        
        Args:
            document: Document to validate
            
        Returns:
            True if metadata is valid, False otherwise
        """
        missing_fields = [
            field for field in self.REQUIRED_FIELDS 
            if field not in document.metadata
        ]
        
        if missing_fields:
            logger.warning(
                f"Document missing required metadata fields: {missing_fields}"
            )
            return False
        
        return True

    def extract_citation_info(self, document: Document) -> Dict[str, str]:
        """
        Extract citation information from document metadata.
        Used for RF6: Source citation in responses.
        
        Args:
            document: Document with metadata
            
        Returns:
            Dictionary with formatted citation information
        """
        metadata = document.metadata
        
        citation = {
            'source_document': metadata.get('source_file', 'Unknown'),
            'category': metadata.get('category', 'General'),
            'section': metadata.get('section', 'N/A'),
            'page': metadata.get('page', 'N/A'),
        }
        
        # Add optional fields if available
        if 'title' in metadata:
            citation['title'] = metadata['title']
        if 'author' in metadata:
            citation['author'] = metadata['author']
        
        return citation

    def format_citation(self, citation_info: Dict[str, str]) -> str:
        """
        Format citation information as a readable string.
        
        Args:
            citation_info: Citation information dictionary
            
        Returns:
            Formatted citation string
        """
        parts = []
        
        # Source document
        if citation_info.get('source_document'):
            parts.append(f"📚 Source: {citation_info['source_document']}")
        
        # Category
        if citation_info.get('category'):
            parts.append(f"🏷️  Category: {citation_info['category']}")
        
        # Section
        if citation_info.get('section') and citation_info['section'] != 'N/A':
            parts.append(f"📄 Section: {citation_info['section']}")
        
        # Page
        if citation_info.get('page') and citation_info['page'] != 'N/A':
            parts.append(f"📖 Page: {citation_info['page']}")
        
        return "\n".join(parts)

    def aggregate_sources(
        self, 
        documents: List[Document]
    ) -> List[Dict[str, str]]:
        """
        Aggregate unique sources from multiple documents.
        
        Args:
            documents: List of documents
            
        Returns:
            List of unique citation information dictionaries
        """
        seen_sources = set()
        unique_citations = []
        
        for doc in documents:
            citation = self.extract_citation_info(doc)
            source_key = (
                citation['source_document'], 
                citation.get('page', 'N/A')
            )
            
            if source_key not in seen_sources:
                seen_sources.add(source_key)
                unique_citations.append(citation)
        
        return unique_citations

    def create_metadata_template(
        self,
        source_file: str,
        category: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create a standardized metadata template.
        
        Args:
            source_file: Source filename
            category: Document category
            **kwargs: Additional metadata fields
            
        Returns:
            Metadata dictionary
        """
        metadata = {
            'source_file': source_file,
            'category': category,
            'created_at': datetime.now().isoformat()
        }
        
        # Add optional fields
        for field in self.OPTIONAL_FIELDS:
            if field in kwargs:
                metadata[field] = kwargs[field]
        
        return metadata

    def filter_by_metadata(
        self,
        documents: List[Document],
        filters: Dict[str, Any]
    ) -> List[Document]:
        """
        Filter documents by metadata criteria.
        
        Args:
            documents: List of documents to filter
            filters: Dictionary of metadata filters
            
        Returns:
            Filtered list of documents
        """
        filtered_docs = []
        
        for doc in documents:
            match = True
            for key, value in filters.items():
                if doc.metadata.get(key) != value:
                    match = False
                    break
            
            if match:
                filtered_docs.append(doc)
        
        logger.info(
            f"Filtered {len(documents)} documents to {len(filtered_docs)} "
            f"matching criteria: {filters}"
        )
        
        return filtered_docs

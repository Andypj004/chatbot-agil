"""
Document loader for various file formats.
Supports PDF, DOCX, TXT, and MD files.
RF6: Maintains source metadata for citation.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any

from langchain.docstore.document import Document
from langchain.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredMarkdownLoader
)

from config.settings import settings


logger = logging.getLogger(__name__)


class DocumentLoader:
    """
    Handles loading of documents from various file formats.
    Preserves metadata for source citation (RF6).
    """

    SUPPORTED_EXTENSIONS = {
        '.pdf': PyPDFLoader,
        '.docx': Docx2txtLoader,
        '.txt': TextLoader,
        '.md': UnstructuredMarkdownLoader,
    }

    def __init__(self, data_path: Path = None):
        """
        Initialize document loader.
        
        Args:
            data_path: Path to raw data directory
        """
        self.data_path = data_path or settings.RAW_DATA_PATH
        logger.info(f"DocumentLoader initialized with path: {self.data_path}")

    def load_single_document(
        self, 
        file_path: Path, 
        category: str = None
    ) -> List[Document]:
        """
        Load a single document and extract metadata.
        
        Args:
            file_path: Path to the document file
            category: Document category (scrum, kanban, syllabus)
            
        Returns:
            List of Document objects with metadata
            
        Raises:
            ValueError: If file format is not supported
        """
        extension = file_path.suffix.lower()
        
        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file format: {extension}. "
                f"Supported formats: {list(self.SUPPORTED_EXTENSIONS.keys())}"
            )
        
        try:
            loader_class = self.SUPPORTED_EXTENSIONS[extension]
            loader = loader_class(str(file_path))
            documents = loader.load()
            
            # Enrich metadata for source citation
            for doc in documents:
                doc.metadata.update({
                    'source_file': file_path.name,
                    'source_path': str(file_path),
                    'category': category or self._infer_category(file_path),
                    'file_type': extension[1:],  # Remove dot
                })
            
            logger.info(f"Successfully loaded {len(documents)} pages from {file_path.name}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {str(e)}")
            raise

    def load_documents_from_directory(
        self, 
        directory: Path = None, 
        recursive: bool = True
    ) -> List[Document]:
        """
        Load all supported documents from a directory.
        
        Args:
            directory: Directory to scan (default: self.data_path)
            recursive: Whether to scan subdirectories
            
        Returns:
            List of all loaded documents
        """
        directory = directory or self.data_path
        all_documents = []
        
        # Get all supported files
        if recursive:
            files = []
            for ext in self.SUPPORTED_EXTENSIONS.keys():
                files.extend(directory.rglob(f"*{ext}"))
        else:
            files = []
            for ext in self.SUPPORTED_EXTENSIONS.keys():
                files.extend(directory.glob(f"*{ext}"))
        
        logger.info(f"Found {len(files)} documents to process in {directory}")
        
        for file_path in files:
            try:
                category = self._infer_category(file_path)
                documents = self.load_single_document(file_path, category)
                all_documents.extend(documents)
            except Exception as e:
                logger.warning(f"Skipping file {file_path}: {str(e)}")
                continue
        
        logger.info(f"Successfully loaded {len(all_documents)} document chunks total")
        return all_documents

    def _infer_category(self, file_path: Path) -> str:
        """
        Infer document category from file path.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Category string (scrum, kanban, syllabus, or general)
        """
        path_str = str(file_path).lower()
        
        if 'scrum' in path_str:
            return 'scrum'
        elif 'kanban' in path_str:
            return 'kanban'
        elif 'silabo' in path_str or 'syllabus' in path_str:
            return 'syllabus'
        else:
            return 'general'

    def get_document_summary(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Generate summary statistics for loaded documents.
        
        Args:
            documents: List of documents
            
        Returns:
            Dictionary with summary statistics
        """
        categories = {}
        file_types = {}
        total_chars = 0
        
        for doc in documents:
            # Count by category
            category = doc.metadata.get('category', 'unknown')
            categories[category] = categories.get(category, 0) + 1
            
            # Count by file type
            file_type = doc.metadata.get('file_type', 'unknown')
            file_types[file_type] = file_types.get(file_type, 0) + 1
            
            # Count characters
            total_chars += len(doc.page_content)
        
        return {
            'total_documents': len(documents),
            'categories': categories,
            'file_types': file_types,
            'total_characters': total_chars,
            'avg_chars_per_doc': total_chars // len(documents) if documents else 0
        }

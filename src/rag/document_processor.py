"""Document processing and chunking utilities"""

from typing import Any, List, Optional, TYPE_CHECKING
from pathlib import Path
import hashlib

from src.core.config import settings
from src.core.logger import get_logger

if TYPE_CHECKING:
    from langchain_core.documents import Document
else:
    Document = Any

logger = get_logger()


class DocumentProcessor:
    """Process and chunk documents for RAG"""

    def __init__(
        self, chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None
    ):
        """Initialize document processor

        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

        from langchain_text_splitters import RecursiveCharacterTextSplitter

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

        logger.info(
            f"Document processor initialized: chunk_size={self.chunk_size}, "
            f"chunk_overlap={self.chunk_overlap}"
        )

    def load_document(
        self,
        file_path: str,
        scope: str = "global_rag",
        session_id: Optional[str] = None,
    ) -> List[Document]:
        """Load a document from file

        Args:
            file_path: Path to the document file

        Returns:
            List of Document objects

        Raises:
            ValueError: If file type is not supported
        """
        path = Path(file_path)

        if not path.exists():
            raise ValueError(f"File not found: {file_path}")

        file_extension = path.suffix.lower()

        logger.info(f"Loading document: {file_path} (type: {file_extension})")
        metadata = self._build_file_metadata(
            file_path=file_path,
            filename=path.name,
            file_extension=file_extension,
            scope=scope,
            session_id=session_id,
        )

        try:
            if file_extension == ".pdf":
                from langchain_community.document_loaders import PyPDFLoader

                loader = PyPDFLoader(file_path)
            elif file_extension == ".txt":
                from langchain_community.document_loaders import TextLoader

                loader = TextLoader(file_path, encoding="utf-8")
            elif file_extension in [".docx", ".doc"]:
                from langchain_community.document_loaders import Docx2txtLoader

                loader = Docx2txtLoader(file_path)
            elif file_extension in [".md", ".markdown"]:
                from langchain_community.document_loaders import (
                    UnstructuredMarkdownLoader,
                )

                loader = UnstructuredMarkdownLoader(file_path)
            else:
                raise ValueError(
                    f"Unsupported file type: {file_extension}. "
                    f"Supported types: .pdf, .txt, .docx, .doc, .md"
                )

            documents = loader.load()

            # Add metadata
            for doc in documents:
                doc.metadata.update(metadata)  # pylint: disable=no-member

            logger.info(f"Loaded {len(documents)} pages/sections from {path.name}")
            return documents

        except Exception as e:
            logger.error(f"Error loading document {file_path}: {e}")
            raise

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks

        Args:
            documents: List of Document objects to chunk

        Returns:
            List of chunked Document objects
        """
        logger.info(f"Chunking {len(documents)} documents")

        chunked_docs = self.text_splitter.split_documents(documents)

        # Add chunk metadata and carry forward section/page hints when available.
        for i, doc in enumerate(chunked_docs):
            doc.metadata["chunk_id"] = i
            if "page" in doc.metadata and "page_label" not in doc.metadata:
                doc.metadata["page_label"] = str(doc.metadata.get("page"))
            if "title" in doc.metadata and "section" not in doc.metadata:
                doc.metadata["section"] = doc.metadata.get("title")

        logger.info(f"Created {len(chunked_docs)} chunks")
        return chunked_docs

    def process_file(
        self,
        file_path: str,
        scope: str = "global_rag",
        session_id: Optional[str] = None,
    ) -> List[Document]:
        """Load and chunk a document file

        Args:
            file_path: Path to the document file

        Returns:
            List of chunked Document objects
        """
        documents = self.load_document(file_path, scope=scope, session_id=session_id)
        chunked_docs = self.chunk_documents(documents)
        return chunked_docs

    def process_text(
        self, text: str, metadata: Optional[dict] = None
    ) -> List[Document]:
        """Process raw text into chunks

        Args:
            text: Text content to process
            metadata: Optional metadata for the document

        Returns:
            List of chunked Document objects
        """
        logger.info(f"Processing raw text ({len(text)} characters)")

        from langchain_core.documents import Document as LangChainDocument

        # Create a document from the text
        doc = LangChainDocument(page_content=text, metadata=metadata or {})

        # Chunk the document
        chunked_docs = self.text_splitter.split_documents([doc])

        logger.info(f"Created {len(chunked_docs)} chunks from text")
        return chunked_docs

    @staticmethod
    def _calculate_file_hash(file_path: str) -> str:
        """Calculate MD5 hash of a file

        Args:
            file_path: Path to the file

        Returns:
            MD5 hash string
        """
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _build_file_metadata(
        self,
        file_path: str,
        filename: str,
        file_extension: str,
        scope: str,
        session_id: Optional[str],
    ) -> dict:
        """Build Chroma-safe metadata (no None values)."""
        metadata = {
            "source": str(file_path),
            "filename": str(filename),
            "file_type": str(file_extension[1:]),
            "file_hash": self._calculate_file_hash(file_path),
            "scope": str(scope),
        }

        # Chroma metadata values must be scalar and cannot be None.
        if session_id:
            metadata["session_id"] = str(session_id)

        return metadata

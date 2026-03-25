"""Vector store management using ChromaDB"""

from typing import Any, Dict, List, Optional, TYPE_CHECKING
from pathlib import Path

from src.core.config import settings
from src.core.logger import get_logger

if TYPE_CHECKING:
    from langchain.schema import Document
else:
    Document = Any

logger = get_logger()


class VectorStore:
    """Vector store manager using ChromaDB"""
    
    def __init__(
        self,
        collection_name: str = "chatbot_documents",
        persist_directory: Optional[str] = None,
        embedding_model: Optional[str] = None
    ):
        """Initialize vector store
        
        Args:
            collection_name: Name of the collection to use
            persist_directory: Directory to persist the database
            embedding_model: Name of the embedding model to use
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory or settings.chroma_persist_dir
        self.embedding_model_name = embedding_model or settings.embedding_model
        self._collection_count_cache: Optional[int] = None
        
        # Create persist directory if it doesn't exist
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        import chromadb
        from chromadb.config import Settings as ChromaSettings
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import Chroma
        
        # Initialize embeddings
        logger.info(f"Loading embedding model: {self.embedding_model_name}")
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
        except Exception as e:
            logger.error(f"Failed to initialize embeddings model '{self.embedding_model_name}': {e}")
            raise RuntimeError(
                "Embedding initialization failed. Check torch/transformers compatibility "
                "and rebuild the Docker image with updated dependencies."
            ) from e
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        
        # Initialize or get collection
        self.vectorstore = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
        )
        
        logger.info(f"Vector store initialized with collection: {self.collection_name}")

    def _rebuild_vectorstore_wrapper(self) -> None:
        """Recreate the LangChain Chroma wrapper after collection resets."""
        from langchain_community.vectorstores import Chroma

        self.vectorstore = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
        )
    
    def add_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """Add documents to the vector store
        
        Args:
            documents: List of Document objects to add
            ids: Optional list of IDs for the documents
            
        Returns:
            List of document IDs
        """
        logger.info(f"Adding {len(documents)} documents to vector store")
        
        if ids:
            document_ids = self.vectorstore.add_documents(documents, ids=ids)
        else:
            document_ids = self.vectorstore.add_documents(documents)

        self._collection_count_cache = None
        
        logger.info(f"Successfully added {len(document_ids)} documents")
        return document_ids
    
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Search for similar documents
        
        Args:
            query: Query string
            k: Number of results to return
            filter: Optional metadata filter
            
        Returns:
            List of similar documents
        """
        logger.info(f"Searching for similar documents: query='{query}', k={k}")
        
        results = self.vectorstore.similarity_search(
            query=query,
            k=k,
            filter=filter
        )
        
        logger.info(f"Found {len(results)} similar documents")
        return results
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[tuple[Document, float]]:
        """Search for similar documents with relevance scores
        
        Args:
            query: Query string
            k: Number of results to return
            filter: Optional metadata filter
            
        Returns:
            List of tuples (document, score)
        """
        logger.info(f"Searching with scores: query='{query}', k={k}")
        
        results = self.vectorstore.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter
        )
        
        logger.info(f"Found {len(results)} documents with scores")
        return results
    
    def delete_documents(self, ids: List[str]) -> bool:
        """Delete documents by IDs
        
        Args:
            ids: List of document IDs to delete
            
        Returns:
            True if successful
        """
        logger.info(f"Deleting {len(ids)} documents")
        
        try:
            self.vectorstore.delete(ids=ids)
            self._collection_count_cache = None
            logger.info("Documents deleted successfully")
            return True
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            return False
    
    def get_collection_count(self, force_refresh: bool = False) -> int:
        """Get the number of documents in the collection
        
        Returns:
            Number of documents
        """
        if self._collection_count_cache is not None and not force_refresh:
            return self._collection_count_cache

        try:
            collection = self.client.get_collection(self.collection_name)
            count = collection.count()
            self._collection_count_cache = count
            logger.info(f"Collection '{self.collection_name}' has {count} documents")
            return count
        except Exception as e:
            logger.error(f"Error getting collection count: {e}")
            return 0

    def count_documents(self, metadata_filter: Optional[Dict[str, Any]] = None) -> int:
        """Count documents by metadata filter (chunk-level)."""
        if not metadata_filter:
            return self.get_collection_count()

        try:
            collection = self.client.get_collection(self.collection_name)
            rows = collection.get(where=metadata_filter)
            ids = rows.get("ids") or []
            return len(ids)
        except Exception as e:
            logger.error(f"Error counting filtered documents: {e}")
            return 0

    def list_indexed_documents(self, metadata_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, str]]:
        """Return unique document metadata entries stored in the collection."""
        try:
            collection = self.client.get_collection(self.collection_name)
            count = collection.count() if metadata_filter is None else self.count_documents(metadata_filter)
            if count == 0:
                return []

            rows = collection.get(limit=count, where=metadata_filter)
            metadatas = rows.get("metadatas") or []
            ids = rows.get("ids") or []

            unique_documents: Dict[str, Dict[str, str]] = {}
            for index, metadata in enumerate(metadatas):
                item = metadata or {}

                source = str(item.get("source") or "").strip()
                filename = str(item.get("filename") or "").strip()
                file_type = str(item.get("file_type") or "").strip().lower()
                file_hash = str(item.get("file_hash") or "").strip()

                if not filename and source:
                    filename = Path(source).name
                if not file_type and filename and "." in filename:
                    file_type = filename.rsplit(".", 1)[-1].lower()

                if file_hash:
                    key = f"hash:{file_hash}"
                elif source:
                    key = f"source:{source}"
                elif filename:
                    key = f"name:{filename}"
                elif index < len(ids) and ids[index]:
                    key = f"id:{ids[index]}"
                else:
                    # Skip chunks with no usable metadata.
                    continue

                if key in unique_documents:
                    continue

                fallback_id = key.split(":", 1)[-1]
                unique_documents[key] = {
                    "document_id": file_hash or fallback_id,
                    "filename": filename or "Documento sin nombre",
                    "file_type": file_type or "unknown",
                    "source": source or "unknown",
                    "file_hash": file_hash or fallback_id,
                    "scope": str(item.get("scope") or "global_rag"),
                    "session_id": item.get("session_id"),
                }

            return sorted(unique_documents.values(), key=lambda doc: doc["filename"].lower())
        except Exception as e:
            logger.error(f"Error listing indexed documents: {e}")
            return []

    def delete_by_metadata(self, metadata_filter: Dict[str, Any]) -> bool:
        """Delete documents that match a metadata filter."""
        logger.info(f"Deleting documents by metadata filter: {metadata_filter}")
        try:
            collection = self.client.get_collection(self.collection_name)
            collection.delete(where=metadata_filter)
            self._collection_count_cache = None
            return True
        except Exception as e:
            logger.error(f"Error deleting documents by metadata: {e}")
            return False
    
    def clear_collection(self) -> bool:
        """Clear all documents from the collection
        
        Returns:
            True if successful
        """
        logger.warning(f"Clearing collection: {self.collection_name}")
        
        try:
            self.client.delete_collection(self.collection_name)
            self._rebuild_vectorstore_wrapper()
            self._collection_count_cache = 0
            logger.info("Collection cleared successfully")
            return True
        except Exception as e:
            error_text = str(e)
            if "does not exist" in error_text.lower():
                # Treat idempotent clears as successful and ensure wrapper exists.
                self._rebuild_vectorstore_wrapper()
                self._collection_count_cache = 0
                logger.info("Collection did not exist; considered cleared")
                return True

            logger.error(f"Error clearing collection: {e}")
            return False

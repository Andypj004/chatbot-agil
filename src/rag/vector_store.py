"""Vector store management using ChromaDB"""

from typing import List, Optional, Dict, Any
from pathlib import Path
import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

from src.core.config import settings
from src.core.logger import get_logger

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
        
        # Create persist directory if it doesn't exist
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize embeddings
        logger.info(f"Loading embedding model: {self.embedding_model_name}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
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
            logger.info("Documents deleted successfully")
            return True
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            return False
    
    def get_collection_count(self) -> int:
        """Get the number of documents in the collection
        
        Returns:
            Number of documents
        """
        try:
            collection = self.client.get_collection(self.collection_name)
            count = collection.count()
            logger.info(f"Collection '{self.collection_name}' has {count} documents")
            return count
        except Exception as e:
            logger.error(f"Error getting collection count: {e}")
            return 0
    
    def clear_collection(self) -> bool:
        """Clear all documents from the collection
        
        Returns:
            True if successful
        """
        logger.warning(f"Clearing collection: {self.collection_name}")
        
        try:
            self.client.delete_collection(self.collection_name)
            # Recreate the collection
            self.vectorstore = Chroma(
                client=self.client,
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
            )
            logger.info("Collection cleared successfully")
            return True
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            return False

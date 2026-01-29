"""Dependency injection for FastAPI"""

from typing import Optional
from fastapi import Depends, HTTPException, status

from src.llm.factory import LLMFactory
from src.llm.base import BaseLLMProvider
from src.rag.vector_store import VectorStore
from src.rag.document_processor import DocumentProcessor
from src.rag.retriever import RAGRetriever
from src.tools.search_tool import SearchTool
from src.agents.chatbot_agent import ChatbotAgent
from src.core.config import settings
from src.core.logger import get_logger

logger = get_logger()

# Global instances (singleton pattern)
_vector_store: Optional[VectorStore] = None
_document_processor: Optional[DocumentProcessor] = None
_search_tool: Optional[SearchTool] = None


def get_vector_store() -> VectorStore:
    """Get or create vector store instance"""
    global _vector_store
    if _vector_store is None:
        logger.info("Initializing vector store")
        _vector_store = VectorStore()
    return _vector_store


def get_document_processor() -> DocumentProcessor:
    """Get or create document processor instance"""
    global _document_processor
    if _document_processor is None:
        logger.info("Initializing document processor")
        _document_processor = DocumentProcessor()
    return _document_processor


def get_search_tool() -> Optional[SearchTool]:
    """Get or create search tool instance"""
    global _search_tool
    if _search_tool is None and settings.has_search_capability():
        try:
            logger.info("Initializing search tool")
            _search_tool = SearchTool()
        except Exception as e:
            logger.warning(f"Failed to initialize search tool: {e}")
            _search_tool = None
    return _search_tool


def get_llm_provider(
    provider_name: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None
) -> BaseLLMProvider:
    """Get LLM provider instance
    
    Args:
        provider_name: LLM provider name
        model_name: Model name
        temperature: Sampling temperature
        max_tokens: Maximum tokens
        
    Returns:
        Configured LLM provider
        
    Raises:
        HTTPException: If provider creation fails
    """
    try:
        return LLMFactory.create_provider(
            provider_name=provider_name,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
    except Exception as e:
        logger.error(f"Failed to create LLM provider: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initialize LLM provider: {str(e)}"
        )


def get_rag_retriever(
    vector_store: VectorStore = Depends(get_vector_store),
    provider_name: Optional[str] = None
) -> RAGRetriever:
    """Get RAG retriever instance
    
    Args:
        vector_store: Vector store instance
        provider_name: LLM provider name
        
    Returns:
        Configured RAG retriever
    """
    llm_provider = get_llm_provider(provider_name=provider_name)
    return RAGRetriever(
        vector_store=vector_store,
        llm_provider=llm_provider
    )


def get_chatbot_agent(
    llm_provider: Optional[BaseLLMProvider] = None,
    use_rag: bool = True,
    use_search: bool = True
) -> ChatbotAgent:
    """Get chatbot agent instance
    
    Args:
        llm_provider: LLM provider instance
        use_rag: Whether to enable RAG
        use_search: Whether to enable search
        
    Returns:
        Configured chatbot agent
    """
    if llm_provider is None:
        llm_provider = get_llm_provider()
    
    # Get RAG retriever if enabled
    rag_retriever = None
    if use_rag:
        try:
            vector_store = get_vector_store()
            rag_retriever = RAGRetriever(
                vector_store=vector_store,
                llm_provider=llm_provider
            )
        except Exception as e:
            logger.warning(f"Failed to initialize RAG retriever: {e}")
    
    # Get search tool if enabled
    search_tool = None
    if use_search:
        search_tool = get_search_tool()
    
    return ChatbotAgent(
        llm_provider=llm_provider,
        rag_retriever=rag_retriever,
        search_tool=search_tool,
        enable_memory=True
    )

"""Dependency injection for FastAPI"""

from functools import lru_cache
from typing import Optional, Dict, Any

from fastapi import Depends, HTTPException, status, Header

from src.llm.factory import LLMFactory
from src.llm.base import BaseLLMProvider
from src.rag.vector_store import VectorStore
from src.rag.document_processor import DocumentProcessor
from src.rag.retriever import RAGRetriever
from src.agents.chatbot_agent import ChatbotAgent
from src.memory.session_manager import SessionManager
from src.core.config import settings
from src.core.logger import get_logger

logger = get_logger()

# Global instances (singleton pattern)
_vector_store: Optional[VectorStore] = None
_document_processor: Optional[DocumentProcessor] = None
_session_manager: Optional[SessionManager] = None


def _provider_cache_key(
    provider_name: Optional[str],
    model_name: Optional[str],
    temperature: Optional[float],
    max_tokens: Optional[int],
) -> tuple[Optional[str], Optional[str], Optional[float], Optional[int]]:
    normalized_provider = (
        LLMFactory.normalize_provider_name(provider_name) if provider_name else None
    )
    return normalized_provider, model_name, temperature, max_tokens


@lru_cache(maxsize=32)
def _create_cached_llm_provider(
    provider_name: Optional[str],
    model_name: Optional[str],
    temperature: Optional[float],
    max_tokens: Optional[int],
) -> BaseLLMProvider:
    return LLMFactory.create_provider(
        provider_name=provider_name,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )


@lru_cache(maxsize=32)
def _create_cached_rag_retriever(
    provider_name: Optional[str],
    model_name: Optional[str],
    temperature: Optional[float],
    max_tokens: Optional[int],
) -> RAGRetriever:
    vector_store = get_vector_store()
    llm_provider = get_llm_provider(
        provider_name=provider_name,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return RAGRetriever(vector_store=vector_store, llm_provider=llm_provider)


def reset_runtime_caches() -> None:
    """Clear cached providers/retrievers after runtime config changes."""
    _create_cached_llm_provider.cache_clear()
    _create_cached_rag_retriever.cache_clear()


def is_vector_store_initialized() -> bool:
    """Return whether the vector store singleton has already been created."""
    return _vector_store is not None


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


def get_session_manager() -> SessionManager:
    """Get or create persistent session manager instance."""
    global _session_manager
    if _session_manager is None:
        logger.info("Initializing session manager")
        _session_manager = SessionManager(db_path=settings.conversation_db_path)
    return _session_manager


def _extract_auth_token(
    authorization: Optional[str] = None,
    x_auth_token: Optional[str] = None,
) -> Optional[str]:
    token = x_auth_token or authorization
    if not token:
        return None

    token = token.strip()
    if token.lower().startswith("bearer "):
        token = token[7:].strip()
    return token or None


def get_current_user_optional(
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
    x_auth_token: Optional[str] = Header(default=None, alias="X-Auth-Token"),
    session_manager: SessionManager = Depends(get_session_manager),
) -> Optional[Dict[str, Any]]:
    """Resolve the current user from an auth token if one is provided."""
    token = _extract_auth_token(authorization=authorization, x_auth_token=x_auth_token)
    if not token:
        return None

    user = session_manager.get_user_by_token(token)
    if user is None:
        return None
    return user


def get_current_user(
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional),
) -> Dict[str, Any]:
    """Require a valid authenticated user."""
    if current_user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )
    return current_user


def get_llm_provider(
    provider_name: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
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
        cache_key = _provider_cache_key(
            provider_name=provider_name,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return _create_cached_llm_provider(*cache_key)
    except ValueError as e:
        logger.error(f"Invalid LLM provider configuration: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create LLM provider: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initialize LLM provider: {str(e)}",
        )


def get_rag_retriever(
    vector_store: VectorStore = Depends(get_vector_store),
    provider_name: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> RAGRetriever:
    """Get RAG retriever instance

    Args:
        vector_store: Vector store instance
        provider_name: LLM provider name

    Returns:
        Configured RAG retriever
    """
    _ = vector_store
    cache_key = _provider_cache_key(
        provider_name=provider_name,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return _create_cached_rag_retriever(*cache_key)


def get_chatbot_agent(
    llm_provider: Optional[BaseLLMProvider] = None,
    provider_name: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    use_rag: bool = True,
) -> ChatbotAgent:
    """Get chatbot agent instance

    Args:
        llm_provider: LLM provider instance
        use_rag: Whether to enable RAG

    Returns:
        Configured chatbot agent
    """
    if llm_provider is None:
        llm_provider = get_llm_provider(
            provider_name=provider_name,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    # Get RAG retriever if enabled
    rag_retriever = None
    if use_rag:
        try:
            rag_retriever = get_rag_retriever(
                provider_name=provider_name or llm_provider.get_provider_name(),
                model_name=model_name or llm_provider.model_name,
                temperature=(
                    temperature if temperature is not None else llm_provider.temperature
                ),
                max_tokens=(
                    max_tokens if max_tokens is not None else llm_provider.max_tokens
                ),
            )
        except Exception as e:
            logger.warning(f"Failed to initialize RAG retriever: {e}")

    return ChatbotAgent(
        llm_provider=llm_provider, rag_retriever=rag_retriever, enable_memory=False
    )

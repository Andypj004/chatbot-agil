"""Health check endpoint"""

from fastapi import APIRouter, Depends

from src.api.models import HealthResponse
from src.api.dependencies import get_vector_store, get_search_tool
from src.llm.factory import LLMFactory
from src.rag.vector_store import VectorStore
from src.core.logger import get_logger
from src import __version__

logger = get_logger()

router = APIRouter(prefix="/health", tags=["health"])


@router.get("", response_model=HealthResponse, summary="Health check")
async def health_check(
    vector_store: VectorStore = Depends(get_vector_store)
):
    """Check the health status of the chatbot system
    
    Returns:
        System health status and component availability
    """
    logger.info("Performing health check")
    
    try:
        # Check vector store
        try:
            doc_count = vector_store.get_collection_count()
            rag_status = "available"
        except Exception as e:
            logger.warning(f"RAG system check failed: {e}")
            doc_count = 0
            rag_status = "unavailable"
        
        # Check search tool
        try:
            search_tool = get_search_tool()
            search_status = "available" if search_tool else "not_configured"
        except Exception:
            search_status = "unavailable"
        
        # Get available providers
        providers = LLMFactory.get_available_providers()
        
        return HealthResponse(
            status="healthy",
            version=__version__,
            llm_providers=providers,
            rag_status=rag_status,
            search_status=search_status,
            vector_store_documents=doc_count
        )
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return HealthResponse(
            status="unhealthy",
            version=__version__,
            llm_providers=[],
            rag_status="error",
            search_status="error",
            vector_store_documents=0
        )

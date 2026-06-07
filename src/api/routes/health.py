"""Health check endpoint"""

from fastapi import APIRouter

from src.api.models import HealthResponse
from src.api.dependencies import get_vector_store, is_vector_store_initialized
from src.llm.factory import LLMFactory
from src.core.logger import get_logger
from src import __version__

logger = get_logger()

router = APIRouter(prefix="/health", tags=["health"])


@router.get("", response_model=HealthResponse, summary="Health check")
async def health_check():
    """Check the health status of the chatbot system

    Returns:
        System health status and component availability
    """
    logger.info("Performing health check")

    try:
        doc_count = 0
        rag_status = "not_initialized"

        # Keep health checks passive so the home page does not pull embeddings/Chroma.
        if is_vector_store_initialized():
            try:
                vector_store = get_vector_store()
                doc_count = vector_store.get_collection_count()
                rag_status = "available"
            except Exception as e:
                logger.warning(f"RAG system check failed: {e}")
                doc_count = 0
                rag_status = "unavailable"

        # Get available providers
        providers = LLMFactory.get_available_providers()

        return HealthResponse(
            status="healthy",
            version=__version__,
            llm_providers=providers,
            rag_status=rag_status,
            vector_store_documents=doc_count,
        )

    except Exception as e:
        logger.error(f"Health check error: {e}")
        return HealthResponse(
            status="unhealthy",
            version=__version__,
            llm_providers=[],
            rag_status="error",
            vector_store_documents=0,
        )

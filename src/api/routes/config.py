"""Configuration management endpoints"""

from fastapi import APIRouter, HTTPException, status

from src.api.models import ConfigUpdateRequest, ConfigResponse
from src.llm.factory import LLMFactory
from src.core.config import settings
from src.core.logger import get_logger

logger = get_logger()

router = APIRouter(prefix="/config", tags=["configuration"])


@router.get("", response_model=ConfigResponse, summary="Get current configuration")
async def get_config():
    """Get the current chatbot configuration
    
    Returns:
        Current configuration settings
    """
    logger.info("Getting current configuration")
    
    try:
        return ConfigResponse(
            llm_provider=settings.default_llm_provider,
            model_name=settings.default_model,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            available_providers=LLMFactory.get_available_providers(),
            rag_enabled=True,  # RAG is always available
            search_enabled=settings.has_search_capability()
        )
    except Exception as e:
        logger.error(f"Error getting configuration: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving configuration: {str(e)}"
        )


@router.post("", response_model=ConfigResponse, summary="Update configuration")
async def update_config(request: ConfigUpdateRequest):
    """Update chatbot configuration
    
    Note: This updates the in-memory configuration only.
    To persist changes, update the .env file.
    
    Args:
        request: Configuration update request
        
    Returns:
        Updated configuration
    """
    logger.info("Updating configuration")
    
    try:
        # Validate provider if specified
        if request.llm_provider:
            available_providers = LLMFactory.get_available_providers()
            if request.llm_provider not in available_providers:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid provider. Available: {', '.join(available_providers)}"
                )
            settings.default_llm_provider = request.llm_provider
        
        # Update settings
        if request.model_name:
            settings.default_model = request.model_name
        if request.temperature is not None:
            settings.temperature = request.temperature
        if request.max_tokens:
            settings.max_tokens = request.max_tokens
        
        logger.info("Configuration updated successfully")
        
        return ConfigResponse(
            llm_provider=settings.default_llm_provider,
            model_name=settings.default_model,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            available_providers=LLMFactory.get_available_providers(),
            rag_enabled=True,
            search_enabled=settings.has_search_capability()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating configuration: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating configuration: {str(e)}"
        )

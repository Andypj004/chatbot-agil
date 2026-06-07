"""Configuration management endpoints"""

from fastapi import APIRouter, HTTPException, status

from src.api.models import ConfigUpdateRequest, ConfigResponse
from src.api.dependencies import reset_runtime_caches
from src.llm.factory import LLMFactory
from src.core.config import settings
from src.core.logger import get_logger

logger = get_logger()

router = APIRouter(prefix="/config", tags=["configuration"])


def _build_config_response() -> ConfigResponse:
    """Build a normalized configuration response payload."""
    return ConfigResponse(
        llm_provider=LLMFactory.normalize_provider_name(settings.default_llm_provider),
        model_name=settings.default_model,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        available_providers=LLMFactory.get_available_providers(),
        available_models=LLMFactory.get_available_models(),
        rag_enabled=True,
    )


@router.get("", response_model=ConfigResponse, summary="Get current configuration")
async def get_config():
    """Get the current chatbot configuration

    Returns:
        Current configuration settings
    """
    logger.info("Getting current configuration")

    try:
        return _build_config_response()
    except Exception as e:
        logger.error(f"Error getting configuration: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving configuration: {str(e)}",
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
        normalized_provider = LLMFactory.normalize_provider_name(
            request.llm_provider or settings.default_llm_provider
        )
        available_providers = LLMFactory.get_available_providers()

        # Validate provider if specified
        if request.llm_provider:
            if normalized_provider not in available_providers:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid provider. Available: {', '.join(available_providers)}",
                )

        available_models = LLMFactory.get_available_models()
        if request.model_name:
            provider_models = available_models.get(normalized_provider, [])
            if provider_models and request.model_name not in provider_models:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=(
                        f"Invalid model for provider '{normalized_provider}'. Available: "
                        f"{', '.join(provider_models)}"
                    ),
                )

        settings.default_llm_provider = normalized_provider

        # Update settings
        if request.model_name:
            settings.default_model = request.model_name
        elif request.llm_provider:
            settings.default_model = LLMFactory._get_default_model_for_provider(
                normalized_provider
            )
        if request.temperature is not None:
            settings.temperature = request.temperature
        if request.max_tokens:
            settings.max_tokens = request.max_tokens

        reset_runtime_caches()

        logger.info("Configuration updated successfully")

        return _build_config_response()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating configuration: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating configuration: {str(e)}",
        )

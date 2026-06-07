"""Google (Gemini) LLM provider implementation"""

from typing import List
import httpx

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chat_models.base import BaseChatModel

from src.llm.base import BaseLLMProvider
from src.core.logger import get_logger

logger = get_logger()


def _normalize_model_name(model_name: str) -> str:
    """Normalize Gemini model names with or without 'models/' prefix."""
    if model_name.startswith("models/"):
        return model_name.split("models/", 1)[1]
    return model_name


class GoogleProvider(BaseLLMProvider):
    """Google LLM provider using Gemini models"""

    def get_llm(self) -> BaseChatModel:
        """Get Google Gemini chat model instance

        Returns:
            Configured ChatGoogleGenerativeAI instance
        """
        if self._llm is None:
            resolved_model = self._resolve_supported_model(self.model_name)
            logger.info(f"Initializing Google provider with model: {resolved_model}")
            self._llm = ChatGoogleGenerativeAI(
                google_api_key=self.api_key,
                model=resolved_model,
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
                **self.kwargs,
            )
            self.model_name = resolved_model
        return self._llm

    def _resolve_supported_model(self, requested_model: str) -> str:
        """Validate requested model against models enabled for current API key/project."""
        normalized_requested = _normalize_model_name(requested_model)
        available_models = self._list_generate_content_models()

        if not available_models:
            logger.warning(
                "Could not determine available Google models; using requested model as-is"
            )
            return normalized_requested

        if normalized_requested in available_models:
            return normalized_requested
        raise ValueError(
            f"Google model '{normalized_requested}' is not enabled for this API key/project. "
            f"Available models: {', '.join(available_models)}"
        )

    def _list_generate_content_models(self) -> List[str]:
        """List models available for generateContent using current API key."""
        url = "https://generativelanguage.googleapis.com/v1beta/models"
        params = {"key": self.api_key}

        try:
            response = httpx.get(url, params=params, timeout=10.0)
            response.raise_for_status()
            payload = response.json()
        except Exception as e:
            logger.warning(f"Failed to fetch Google model list: {e}")
            return []

        models = []
        for entry in payload.get("models", []):
            methods = entry.get("supportedGenerationMethods", [])
            name = entry.get("name", "")
            if "generateContent" in methods and name:
                models.append(_normalize_model_name(name))

        return sorted(set(models))

    def get_provider_name(self) -> str:
        """Get provider name

        Returns:
            Provider name string
        """
        return "google"

    def get_default_model(self) -> str:
        """Get default model for Google

        Returns:
            Default model name
        """
        return "gemini-1.5-flash"

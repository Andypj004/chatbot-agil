"""Base class for LLM providers"""

from abc import ABC, abstractmethod
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    # Accept both chat models (BaseChatModel) and completion LLMs (BaseLLM).
    from langchain_core.language_models import BaseLanguageModel
else:
    BaseLanguageModel = Any


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers"""

    def __init__(
        self,
        api_key: str,
        model_name: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ):
        """Initialize LLM provider

        Args:
            api_key: API key for the provider
            model_name: Name of the model to use
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters
        """
        self.validate_generation_params(temperature=temperature, max_tokens=max_tokens)
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.kwargs = kwargs
        self._llm: Optional[BaseLanguageModel] = None

    @abstractmethod
    def get_llm(self) -> BaseLanguageModel:
        """Get the LangChain language model instance.

        Returns a ``BaseChatModel`` for cloud providers (OpenAI, Anthropic, …)
        or a ``BaseLLM`` for local providers (Ollama). Both support ``invoke()``.
        """
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """Get the provider name

        Returns:
            Provider name string
        """
        pass

    def get_default_model(self) -> str:
        """Get the default model for this provider

        Returns:
            Default model name
        """
        return self.model_name

    def validate_api_key(self) -> bool:
        """Validate that the API key is set

        Returns:
            True if API key is valid, False otherwise
        """
        return bool(self.api_key and self.api_key.strip())

    @staticmethod
    def validate_generation_params(temperature: float, max_tokens: int) -> None:
        """Validate common generation parameters shared across providers."""
        if not 0.0 <= temperature <= 1.0:
            raise ValueError("temperature must be between 0.0 and 1.0")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than 0")

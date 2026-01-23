"""Base class for LLM providers"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from langchain.schema import BaseMessage
from langchain.chat_models.base import BaseChatModel


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
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.kwargs = kwargs
        self._llm: Optional[BaseChatModel] = None
    
    @abstractmethod
    def get_llm(self) -> BaseChatModel:
        """Get the LangChain chat model instance
        
        Returns:
            Configured BaseChatModel instance
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

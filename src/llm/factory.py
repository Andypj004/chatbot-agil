"""Factory for creating LLM provider instances"""

from typing import Optional
from src.core.config import settings
from src.core.logger import get_logger
from src.llm.base import BaseLLMProvider

logger = get_logger()


class LLMFactory:
    """Factory class for creating LLM provider instances"""
    
    _providers = {}
    
    @classmethod
    def register_provider(cls, name: str, provider_class: type):
        """Register a new LLM provider
        
        Args:
            name: Provider name (e.g., 'openai', 'anthropic')
            provider_class: Provider class to register
        """
        cls._providers[name.lower()] = provider_class
        logger.info(f"Registered LLM provider: {name}")
    
    @classmethod
    def create_provider(
        cls,
        provider_name: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> BaseLLMProvider:
        """Create an LLM provider instance
        
        Args:
            provider_name: Name of the provider (defaults to settings)
            model_name: Model name (defaults to provider default)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Configured BaseLLMProvider instance
            
        Raises:
            ValueError: If provider is not registered or API key is missing
        """
        # Use defaults from settings if not provided
        provider_name = provider_name or settings.default_llm_provider
        provider_name = provider_name.lower()
        
        if provider_name not in cls._providers:
            available = ", ".join(cls._providers.keys())
            raise ValueError(
                f"Provider '{provider_name}' is not registered. "
                f"Available providers: {available}"
            )
        
        # Get API key for the provider
        api_key = settings.get_api_key(provider_name)
        if not api_key:
            raise ValueError(
                f"API key for provider '{provider_name}' is not configured. "
                f"Please set it in the .env file."
            )
        
        # Get provider class and create instance
        provider_class = cls._providers[provider_name]
        
        # Use settings defaults if not provided
        temperature = temperature if temperature is not None else settings.temperature
        max_tokens = max_tokens if max_tokens is not None else settings.max_tokens
        
        # Determine model name
        if model_name is None:
            # Use default model from settings or provider default
            model_name = cls._get_default_model_for_provider(provider_name)
        
        logger.info(
            f"Creating {provider_name} provider with model: {model_name}, "
            f"temperature: {temperature}, max_tokens: {max_tokens}"
        )
        
        return provider_class(
            api_key=api_key,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
    
    @classmethod
    def _get_default_model_for_provider(cls, provider_name: str) -> str:
        """Get default model for a provider
        
        Args:
            provider_name: Name of the provider
            
        Returns:
            Default model name
        """
        default_models = {
            "openai": "gpt-4-turbo-preview",
            "anthropic": "claude-3-opus-20240229",
            "claude": "claude-3-opus-20240229",
            "google": "gemini-pro",
            "gemini": "gemini-pro",
            "deepseek": "deepseek-chat",
        }
        return default_models.get(provider_name, settings.default_model)
    
    @classmethod
    def get_available_providers(cls) -> list:
        """Get list of available providers
        
        Returns:
            List of registered provider names
        """
        return list(cls._providers.keys())


# Auto-register providers on import
def _register_providers():
    """Register all available providers"""
    try:
        from src.llm.providers.openai_provider import OpenAIProvider
        LLMFactory.register_provider("openai", OpenAIProvider)
    except ImportError:
        logger.warning("OpenAI provider not available")
    
    try:
        from src.llm.providers.anthropic_provider import AnthropicProvider
        LLMFactory.register_provider("anthropic", AnthropicProvider)
        LLMFactory.register_provider("claude", AnthropicProvider)
    except ImportError:
        logger.warning("Anthropic provider not available")
    
    try:
        from src.llm.providers.google_provider import GoogleProvider
        LLMFactory.register_provider("google", GoogleProvider)
        LLMFactory.register_provider("gemini", GoogleProvider)
    except ImportError:
        logger.warning("Google provider not available")
    
    try:
        from src.llm.providers.deepseek_provider import DeepseekProvider
        LLMFactory.register_provider("deepseek", DeepseekProvider)
    except ImportError:
        logger.warning("Deepseek provider not available")


_register_providers()

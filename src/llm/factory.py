"""Factory for creating LLM provider instances"""

from typing import Optional

from src.core.config import settings
from src.core.logger import get_logger
from src.llm.base import BaseLLMProvider

logger = get_logger()


class LLMFactory:
    """Factory class for creating LLM provider instances"""
    
    _providers = {}
    _provider_aliases = {
        "claude": "anthropic",
        "gemini": "google",
    }
    _provider_models = {
        "openai": ["gpt-4o-mini", "gpt-4-turbo-preview", "gpt-3.5-turbo"],
        "anthropic": ["claude-3-5-haiku-latest", "claude-3-5-sonnet-latest"],
        "google": [
            "gemini-1.5-flash",
            "gemini-1.5-flash-8b",
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
            "gemini-pro",
        ],
        "deepseek": ["deepseek-chat", "deepseek-reasoner"],
        "ollama": ["llama3.2:3b", "qwen2.5:3b", "phi3:mini"],
    }
    
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
        provider_name = cls.normalize_provider_name(provider_name)
        
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
        BaseLLMProvider.validate_generation_params(
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        # Determine model name
        if model_name is None:
            # Use default model from settings or provider default
            model_name = cls._get_default_model_for_provider(provider_name)
        else:
            provider_models = cls._provider_models.get(provider_name, [])
            if provider_models and model_name not in provider_models:
                raise ValueError(
                    f"Model '{model_name}' is not supported for provider "
                    f"'{provider_name}'. Available models: {', '.join(provider_models)}"
                )
        
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
        provider_name = cls.normalize_provider_name(provider_name)
        default_models = {
            "openai": "gpt-4-turbo-preview",
            "anthropic": "claude-3-5-sonnet-latest",
            "google": "gemini-1.5-flash",
            "deepseek": "deepseek-chat",
            "ollama": "llama3.2:3b",
        }
        return default_models.get(provider_name, settings.default_model)

    @classmethod
    def normalize_provider_name(cls, provider_name: str) -> str:
        """Normalize provider aliases to a canonical provider name."""
        name = provider_name.lower()
        return cls._provider_aliases.get(name, name)
    
    @classmethod
    def get_available_providers(cls) -> list:
        """Get list of available providers
        
        Returns:
            List of registered provider names
        """
        canonical_providers = {
            cls.normalize_provider_name(provider_name)
            for provider_name in cls._providers.keys()
        }
        preferred_order = ["openai", "anthropic", "google", "deepseek", "ollama"]
        ordered = [p for p in preferred_order if p in canonical_providers]
        extras = sorted([p for p in canonical_providers if p not in preferred_order])
        return ordered + extras

    @classmethod
    def get_available_models(cls) -> dict:
        """Get model catalog for currently available canonical providers."""
        providers = cls.get_available_providers()
        return {provider: cls._provider_models.get(provider, []) for provider in providers}


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

    try:
        from src.llm.providers.ollama_provider import OllamaProvider
        LLMFactory.register_provider("ollama", OllamaProvider)
    except ImportError:
        logger.warning("Ollama provider not available")


_register_providers()

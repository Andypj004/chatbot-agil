"""Factory for creating LLM provider instances"""

from typing import Optional
import json
import os
import time

from src.core.config import settings
from src.core.logger import get_logger
from src.llm.base import BaseLLMProvider

logger = get_logger()

# Cached result of the Ollama reachability probe: (timestamp, base_url or None).
# Probing costs a network timeout, and `get_available_providers()` runs on every
# GET /config and /health call, so the result is reused for a short window.
_OLLAMA_PROBE_TTL_SECONDS = 60
_ollama_probe_cache: tuple[float, Optional[str]] = (0.0, None)


def _reachable_ollama_base_url(force: bool = False) -> Optional[str]:
    """Return a reachable Ollama base URL, or None if the server is not up."""
    global _ollama_probe_cache

    probed_at, cached_url = _ollama_probe_cache
    if not force and (time.monotonic() - probed_at) < _OLLAMA_PROBE_TTL_SECONDS:
        return cached_url

    base_url = None
    try:
        # Imported lazily: langchain_ollama may not be installed.
        from src.llm.providers.ollama_provider import (
            _candidate_base_urls,
            _is_ollama_reachable,
        )

        for candidate in _candidate_base_urls(settings.ollama_base_url):
            if _is_ollama_reachable(candidate):
                base_url = candidate
                break
    except Exception as e:
        logger.debug(f"Ollama availability probe failed: {e}")

    _ollama_probe_cache = (time.monotonic(), base_url)
    return base_url


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
            "gemini-3.1-flash-lite",
            "gemini-1.5-flash",
            "gemini-1.5-flash-8b",
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
            "gemini-pro",
        ],
        "deepseek": ["deepseek-chat", "deepseek-reasoner"],
        "ollama": [
            "llama3.2:3b",
            "qwen2.5:3b",
            "phi3:mini",
            "llava:7b",
            "llava:13b",
            "gemma3:4b",
        ],
    }

    # Load external models catalog if available (src/llm/models.json)
    try:
        _models_path = os.path.join(os.path.dirname(__file__), "models.json")
        if os.path.exists(_models_path):
            with open(_models_path, "r", encoding="utf-8") as _f:
                _external = json.load(_f)
                if isinstance(_external, dict):
                    # sanitize: only lists
                    cleaned = {
                        k: v for k, v in _external.items() if isinstance(v, list)
                    }
                    if cleaned:
                        _provider_models.clear()
                        _provider_models.update(cleaned)
                        logger.info(
                            "Loaded external LLM model catalog from src/llm/models.json"
                        )
    except Exception:
        logger.warning(
            "Failed to load external models catalog; using builtin provider list"
        )

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
        **kwargs,
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
        default_provider = cls.normalize_provider_name(settings.default_llm_provider)

        if settings.pilot_lock:
            # Pilot lock: only the configured default provider may be used. Any other
            # requested provider (registered or not) is ignored in favor of the
            # default, and its accompanying model request is ignored too.
            if (
                provider_name
                and cls.normalize_provider_name(provider_name) != default_provider
            ):
                provider_name = default_provider
                model_name = None
            else:
                provider_name = default_provider
        else:
            provider_name = (
                cls.normalize_provider_name(provider_name)
                if provider_name
                else default_provider
            )
            if not cls.is_provider_available(provider_name):
                available = ", ".join(cls.get_available_providers())
                raise ValueError(
                    f"Provider '{provider_name}' is not available. "
                    f"Available providers: {available}"
                )

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
                # Unsupported model requests fall back to the provider default
                # instead of erroring.
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
            **kwargs,
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
        if settings.default_model:
            provider_models = cls._provider_models.get(provider_name, [])
            if not provider_models or settings.default_model in provider_models:
                return settings.default_model

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
    def get_registered_providers(cls) -> list:
        """Get every provider backend implemented in this codebase.

        Unlike `get_available_providers`, this is not subject to the pilot lock —
        it reflects the factory's full registration, used for introspection
        (e.g. confirming the multi-provider architecture is in place).
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
    def is_provider_available(cls, provider_name: str) -> bool:
        """Return whether a provider is registered and actually usable.

        A provider is usable when its backend is registered and it has a real API
        key configured. Ollama needs no key, so it counts as available only when
        its server responds.
        """
        provider_name = cls.normalize_provider_name(provider_name)
        if provider_name not in cls._providers:
            return False

        if provider_name == "ollama":
            return _reachable_ollama_base_url() is not None

        return settings.has_provider_api_key(provider_name)

    @classmethod
    def get_available_providers(cls) -> list:
        """Get list of providers that can be selected at runtime.

        With the pilot lock enabled only the configured default provider is
        exposed. Otherwise every registered provider with a usable configuration
        is exposed, with the default provider first.

        Returns:
            List of canonical provider names.
        """
        default_provider = cls.normalize_provider_name(settings.default_llm_provider)

        if settings.pilot_lock:
            return [default_provider]

        available = [
            provider_name
            for provider_name in cls.get_registered_providers()
            if cls.is_provider_available(provider_name)
        ]

        # The default provider always leads the list so it stays in sync with the
        # `llm_provider` field the frontend preselects, even if it is misconfigured.
        if default_provider in available:
            available.remove(default_provider)
        available.insert(0, default_provider)
        return available

    @classmethod
    def get_available_models(cls) -> dict:
        """Get the model catalog for every currently available provider.

        With the pilot lock enabled only the configured default model is exposed.
        """
        providers = cls.get_available_providers()

        if settings.pilot_lock:
            provider = providers[0]
            return {provider: [cls._get_default_model_for_provider(provider)]}

        catalog = {}
        for provider_name in providers:
            if provider_name == "ollama":
                catalog[provider_name] = cls._get_installed_ollama_models()
            else:
                catalog[provider_name] = list(
                    cls._provider_models.get(provider_name, [])
                )
        return catalog

    @classmethod
    def _get_installed_ollama_models(cls) -> list:
        """Return the model tags actually pulled in the local Ollama server."""
        fallback = list(cls._provider_models.get("ollama", []))

        base_url = _reachable_ollama_base_url()
        if not base_url:
            return fallback

        try:
            from src.llm.providers.ollama_provider import _list_ollama_models

            return _list_ollama_models(base_url) or fallback
        except Exception as e:
            logger.debug(f"Could not list installed Ollama models: {e}")
            return fallback


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

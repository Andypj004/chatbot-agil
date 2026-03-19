"""LLM Provider abstraction layer"""

from src.llm.base import BaseLLMProvider
from src.llm.factory import LLMFactory

try:
    from src.llm.providers.openai_provider import OpenAIProvider
except ImportError:  # pragma: no cover - optional dependency
    OpenAIProvider = None

try:
    from src.llm.providers.anthropic_provider import AnthropicProvider
except ImportError:  # pragma: no cover - optional dependency
    AnthropicProvider = None

try:
    from src.llm.providers.google_provider import GoogleProvider
except ImportError:  # pragma: no cover - optional dependency
    GoogleProvider = None

try:
    from src.llm.providers.deepseek_provider import DeepseekProvider
except ImportError:  # pragma: no cover - optional dependency
    DeepseekProvider = None

try:
    from src.llm.providers.ollama_provider import OllamaProvider
except ImportError:  # pragma: no cover - optional dependency
    OllamaProvider = None

__all__ = [
    "BaseLLMProvider",
    "LLMFactory",
    "OpenAIProvider",
    "AnthropicProvider",
    "GoogleProvider",
    "DeepseekProvider",
    "OllamaProvider",
]

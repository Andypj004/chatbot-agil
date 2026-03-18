"""Ollama LLM provider implementation."""

import httpx
from langchain_community.llms import Ollama
from langchain.llms.base import BaseLLM

from src.llm.base import BaseLLMProvider
from src.core.config import settings
from src.core.logger import get_logger

logger = get_logger()


def _normalize_url(url: str) -> str:
    """Normalize base URL by trimming whitespace and trailing slash."""
    return url.strip().rstrip("/")


def _is_ollama_reachable(base_url: str) -> bool:
    """Check if Ollama API is reachable on the given URL."""
    try:
        response = httpx.get(f"{base_url}/api/tags", timeout=3.0)
        return response.status_code == 200
    except Exception:
        return False


def _list_ollama_models(base_url: str) -> list[str]:
    """Return locally available Ollama model tags."""
    try:
        response = httpx.get(f"{base_url}/api/tags", timeout=3.0)
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return []

    return [model.get("name", "") for model in payload.get("models", []) if model.get("name")]


def _candidate_base_urls(configured_url: str) -> list:
    """Build candidate Ollama URLs for common Docker/WSL setups."""
    configured = _normalize_url(configured_url)
    candidates = [configured]

    if "localhost" in configured:
        candidates.append(configured.replace("localhost", "host.docker.internal"))
        candidates.append(configured.replace("localhost", "ollama"))
    if "127.0.0.1" in configured:
        candidates.append(configured.replace("127.0.0.1", "host.docker.internal"))
        candidates.append(configured.replace("127.0.0.1", "ollama"))

    # Common direct alternatives if configured URL is custom.
    candidates.extend(["http://ollama:11434", "http://host.docker.internal:11434"])

    # Preserve order while removing duplicates.
    unique_candidates = []
    for url in candidates:
        if url not in unique_candidates:
            unique_candidates.append(url)
    return unique_candidates


class OllamaProvider(BaseLLMProvider):
    """Ollama provider for local/offline models."""

    def get_llm(self) -> BaseLLM:
        """Get Ollama LLM instance (uses /api/generate)."""
        if self._llm is None:
            resolved_base_url = None
            available_models = []
            for candidate in _candidate_base_urls(settings.ollama_base_url):
                if _is_ollama_reachable(candidate):
                    resolved_base_url = candidate
                    available_models = _list_ollama_models(candidate)
                    break

            if not resolved_base_url:
                candidates = ", ".join(_candidate_base_urls(settings.ollama_base_url))
                raise ConnectionError(
                    "Could not reach Ollama API from chatbot container. "
                    f"Tried: {candidates}. "
                    "If using docker compose with Ollama service, run with profile 'ollama'. "
                    "If using host Ollama, set OLLAMA_BASE_URL=http://host.docker.internal:11434."
                )

            if available_models and self.model_name not in available_models:
                raise ValueError(
                    f"Ollama model '{self.model_name}' is not available locally. "
                    f"Run 'ollama pull {self.model_name}' first. Available models: "
                    f"{', '.join(available_models)}"
                )

            logger.info(
                f"Initializing Ollama provider with model: {self.model_name} "
                f"at {resolved_base_url}"
            )
            self._llm = Ollama(
                base_url=resolved_base_url,
                model=self.model_name,
                temperature=self.temperature,
                **self.kwargs,
            )
        return self._llm

    def get_provider_name(self) -> str:
        """Get provider name."""
        return "ollama"

    def get_default_model(self) -> str:
        """Get default model for Ollama."""
        return "llama3.2:3b"

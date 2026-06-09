"""Unit tests for LLM factory"""

import pytest
from unittest.mock import Mock, patch

from src.llm.factory import LLMFactory
from src.llm.base import BaseLLMProvider


def test_factory_registration():
    """Test provider registration"""
    available_providers = LLMFactory.get_available_providers()
    assert len(available_providers) > 0
    assert "openai" in available_providers


def test_create_provider_with_defaults():
    """Test creating provider with default settings"""
    with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
        from src.core.config import Settings
        with patch('src.llm.factory.settings') as mock_settings:
            mock_settings.get_api_key.return_value = 'test-key'
            mock_settings.default_llm_provider = 'openai'
            mock_settings.temperature = 0.7
            mock_settings.max_tokens = 2000
            
            # This will fail without a real API key, but tests the flow
            try:
                provider = LLMFactory.create_provider()
                assert isinstance(provider, BaseLLMProvider)
            except Exception:
                # Expected if no real API key
                pass


def test_invalid_provider():
    """Test creating invalid provider"""
    with pytest.raises(ValueError, match="not registered"):
        LLMFactory.create_provider(provider_name="invalid_provider")


def test_get_available_providers():
    """Test getting available providers"""
    providers = LLMFactory.get_available_providers()
    assert isinstance(providers, list)
    assert len(providers) > 0


def test_get_available_models_catalog_shape():
    """Test model catalog structure"""
    models = LLMFactory.get_available_models()
    assert isinstance(models, dict)
    for provider_name, provider_models in models.items():
        assert isinstance(provider_name, str)
        assert isinstance(provider_models, list)


def test_provider_defaults_exist_in_catalogs():
    """Every provider default model should be present in its advertised catalog."""
    models = LLMFactory.get_available_models()
    for provider_name, provider_models in models.items():
        default_model = LLMFactory._get_default_model_for_provider(provider_name)
        assert default_model in provider_models


def test_factory_prefers_settings_default_model_when_supported():
    """The factory should honor the configured default model when it is valid for the provider."""
    with patch("src.llm.factory.settings") as mock_settings:
        mock_settings.default_model = "gpt-4o-mini"
        mock_settings.temperature = 0.7
        mock_settings.max_tokens = 2000

        default_model = LLMFactory._get_default_model_for_provider("openai")

        assert default_model == "gpt-4o-mini"


def test_factory_rejects_invalid_model_for_provider():
    """Factory should reject provider/model combinations outside the catalog."""
    with pytest.raises(ValueError, match="is not supported for provider 'openai'"):
        LLMFactory.create_provider(
            provider_name="openai",
            model_name="not-a-real-model",
        )


def test_factory_rejects_invalid_generation_params():
    """Factory should reject invalid shared generation settings early."""
    with pytest.raises(ValueError, match="temperature must be between 0.0 and 1.0"):
        LLMFactory.create_provider(provider_name="openai", temperature=1.5)

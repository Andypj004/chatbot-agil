"""Integration tests for API endpoints"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

from src.api import dependencies
from src.main import app

client = TestClient(app)


def test_root_endpoint():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data


def test_frontend_endpoint():
    """Test frontend endpoint"""
    response = client.get("/app")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")


def test_health_endpoint():
    """Test health check endpoint"""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "version" in data
    assert "llm_providers" in data


def test_get_config():
    """Test get configuration endpoint"""
    response = client.get("/api/v1/config")
    assert response.status_code == 200
    data = response.json()
    assert "llm_provider" in data
    assert "model_name" in data
    assert "temperature" in data
    assert "available_models" in data
    assert "rag_enabled" in data


@patch('src.api.routes.chat.get_llm_provider')
@patch('src.api.routes.chat.get_chatbot_agent')
def test_chat_endpoint(mock_get_agent, mock_get_provider):
    """Test chat endpoint"""
    mock_get_provider.return_value = Mock()

    # Mock the agent
    mock_agent = Mock()
    mock_agent.chat.return_value = {
        "response": "Test response",
        "provider": "openai",
        "model": "gpt-4",
        "used_rag": False,
    }
    mock_get_agent.return_value = mock_agent
    
    response = client.post(
        "/api/v1/chat",
        json={
            "message": "Hello, world!",
            "use_rag": False
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert "provider" in data
    assert "session_id" in data


def test_chat_endpoint_invalid_request():
    """Test chat endpoint with invalid request"""
    response = client.post(
        "/api/v1/chat",
        json={}  # Missing required 'message' field
    )
    
    assert response.status_code == 422  # Unprocessable Entity


def test_update_config_rejects_invalid_model_for_provider():
    """Config update should reject models outside the selected provider catalog."""
    response = client.post(
        "/api/v1/config",
        json={
            "llm_provider": "anthropic",
            "model_name": "gpt-4-turbo-preview"
        }
    )

    assert response.status_code == 400
    assert "Invalid model for provider 'anthropic'" in response.json()["detail"]


def test_update_config_sets_provider_default_model_when_model_omitted():
    """Switching providers without a model should apply the provider default."""
    response = client.post(
        "/api/v1/config",
        json={
            "llm_provider": "anthropic"
        }
    )

    assert response.status_code == 200
    data = response.json()
    assert data["llm_provider"] == "anthropic"
    assert data["model_name"] == "claude-3-5-sonnet-latest"


@patch('src.api.dependencies.LLMFactory.create_provider')
def test_get_llm_provider_reuses_cached_instance(mock_create_provider):
    """Dependency helper should reuse identical provider instances across requests."""
    dependencies.reset_runtime_caches()
    mock_provider = Mock()
    mock_create_provider.return_value = mock_provider

    first = dependencies.get_llm_provider(provider_name="openai", model_name="gpt-4o-mini")
    second = dependencies.get_llm_provider(provider_name="openai", model_name="gpt-4o-mini")

    assert first is second
    mock_create_provider.assert_called_once()


@patch('src.api.dependencies._create_cached_rag_retriever')
def test_get_chatbot_agent_builds_rag_only_when_enabled(mock_cached_retriever):
    """Agent factory should avoid retriever work when RAG is disabled."""
    mock_provider = Mock()
    mock_provider.get_provider_name.return_value = "openai"
    mock_provider.model_name = "gpt-4o-mini"
    mock_provider.temperature = 0.7
    mock_provider.max_tokens = 2000

    agent = dependencies.get_chatbot_agent(llm_provider=mock_provider, use_rag=False)

    assert agent.rag_retriever is None
    mock_cached_retriever.assert_not_called()


def test_chat_endpoint_rejects_invalid_model_for_provider_request():
    """Chat should surface invalid provider/model combinations as client errors."""
    response = client.post(
        "/api/v1/chat",
        json={
            "message": "Hello",
            "llm_provider": "anthropic",
            "model_name": "gpt-4-turbo-preview",
            "use_rag": False,
        }
    )

    assert response.status_code == 400
    assert "not supported for provider 'anthropic'" in response.json()["detail"]


@patch('src.api.routes.chat.get_llm_provider')
@patch('src.api.routes.chat.get_chatbot_agent')
def test_chat_history_endpoints(mock_get_agent, mock_get_provider):
    """Session list/history endpoints should return persisted chat turns."""
    mock_get_provider.return_value = Mock()

    mock_agent = Mock()
    mock_agent.chat.return_value = {
        "response": "Respuesta persistida",
        "provider": "openai",
        "model": "gpt-4o-mini",
        "used_rag": False,
        "sources": [],
    }
    mock_get_agent.return_value = mock_agent

    session_id = "test-session-history"
    chat_response = client.post(
        "/api/v1/chat",
        json={
            "session_id": session_id,
            "message": "Hola historial",
            "use_rag": False,
        },
    )
    assert chat_response.status_code == 200

    sessions_response = client.get("/api/v1/sessions")
    assert sessions_response.status_code == 200
    sessions_payload = sessions_response.json()
    assert "sessions" in sessions_payload
    assert any(item["session_id"] == session_id for item in sessions_payload["sessions"])

    history_response = client.get(f"/api/v1/sessions/{session_id}/history")
    assert history_response.status_code == 200
    history_payload = history_response.json()
    assert history_payload["session_id"] == session_id
    assert history_payload["total_messages"] >= 2


@patch('src.api.routes.chat.get_llm_provider')
@patch('src.api.routes.chat.get_chatbot_agent')
def test_chat_streaming_endpoint(mock_get_agent, mock_get_provider):
    """Streaming chat should emit SSE response payload."""
    mock_get_provider_instance = Mock()
    mock_get_provider_instance.get_provider_name.return_value = "openai"
    mock_get_provider_instance.model_name = "gpt-4o-mini"
    mock_get_provider.return_value = mock_get_provider_instance

    mock_agent = Mock()

    def _stream():
        yield {"type": "delta", "content": "Hola "}
        yield {"type": "delta", "content": "mundo"}
        yield {
            "type": "final",
            "provider": "openai",
            "model": "gpt-4o-mini",
            "used_rag": False,
            "sources": [],
        }

    mock_agent.chat_stream.return_value = _stream()
    mock_get_agent.return_value = mock_agent

    response = client.post(
        "/api/v1/chat",
        json={
            "message": "Hola",
            "use_rag": False,
            "stream": True,
        },
    )

    assert response.status_code == 200
    assert response.headers.get("content-type", "").startswith("text/event-stream")
    body = response.text
    assert '"type": "delta"' in body
    assert '"type": "final"' in body

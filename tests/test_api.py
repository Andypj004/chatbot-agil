"""Integration tests for API endpoints"""

import pytest
from pathlib import Path
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
from langchain_core.documents import Document

from src.api import dependencies
from src.core import config as cfg_module
from src.main import app
from src.memory.session_manager import SessionManager

client = TestClient(app)

ADMIN_EMAIL = "admin@configtest.com"
ADMIN_PASSWORD = "adminpass123"


@pytest.fixture(autouse=True)
def isolate_runtime_state(tmp_path, monkeypatch):
    """Ensure each test runs with a writable isolated session database."""
    test_db_path = tmp_path / "conversations-test.db"
    monkeypatch.setattr(
        dependencies, "_session_manager", SessionManager(db_path=str(test_db_path))
    )
    monkeypatch.setattr(cfg_module.settings, "admin_emails", ADMIN_EMAIL)
    app.dependency_overrides.clear()
    yield
    app.dependency_overrides.clear()
    monkeypatch.setattr(dependencies, "_session_manager", None)


def _register_user(email, password="secret123", account_type="Estudiante"):
    response = client.post(
        "/api/v1/auth/register",
        json={
            "email": email,
            "password": password,
            "full_name": "Test User",
            "account_type": account_type,
            "knowledge_level": 2,
            "questionnaire_answers": [],
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def _auth_header(token):
    return {"Authorization": f"Bearer {token}"}


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


def test_user_registration_and_login_flow():
    """Users should be able to register, log in, and receive a scoped profile."""
    registration_response = client.post(
        "/api/v1/auth/register",
        json={
            "email": "profesor@example.com",
            "password": "secret123",
            "full_name": "Ada Lovelace",
            "account_type": "Profesor",
            "knowledge_level": 2,
            "questionnaire_answers": [
                {"question_number": 1, "answer": "c"},
                {"question_number": 2, "answer": "c"},
                {"question_number": 3, "answer": "c"},
                {"question_number": 4, "answer": "c"},
                {"question_number": 5, "answer": "c"},
            ],
        },
    )

    assert registration_response.status_code == 200
    registration_payload = registration_response.json()
    assert registration_payload["token_type"] == "bearer"
    assert registration_payload["user"]["agile_adoption_level"] == 4

    login_response = client.post(
        "/api/v1/auth/login",
        json={
            "email": "profesor@example.com",
            "password": "secret123",
        },
    )

    assert login_response.status_code == 200
    login_payload = login_response.json()
    assert login_payload["user"]["email"] == "profesor@example.com"
    assert login_payload["access_token"]


def test_authenticated_users_get_isolated_sessions():
    """Each authenticated user should see only their own sessions."""
    first_user = client.post(
        "/api/v1/auth/register",
        json={
            "email": "student1@example.com",
            "password": "secret123",
            "full_name": "Student One",
            "account_type": "Estudiante",
            "knowledge_level": 1,
            "questionnaire_answers": [
                {"question_number": 1, "answer": "a"},
                {"question_number": 2, "answer": "a"},
                {"question_number": 3, "answer": "a"},
                {"question_number": 4, "answer": "a"},
                {"question_number": 5, "answer": "a"},
            ],
        },
    ).json()

    second_user = client.post(
        "/api/v1/auth/register",
        json={
            "email": "student2@example.com",
            "password": "secret123",
            "full_name": "Student Two",
            "account_type": "Estudiante",
            "knowledge_level": 3,
            "questionnaire_answers": [
                {"question_number": 1, "answer": "b"},
                {"question_number": 2, "answer": "b"},
                {"question_number": 3, "answer": "b"},
                {"question_number": 4, "answer": "b"},
                {"question_number": 5, "answer": "b"},
            ],
        },
    ).json()

    mock_provider = Mock()
    mock_agent = Mock()
    mock_agent.chat.return_value = {
        "response": "Respuesta privada",
        "provider": "openai",
        "model": "gpt-4",
        "used_rag": False,
        "sources": [],
    }

    with patch(
        "src.api.routes.chat.get_llm_provider", return_value=mock_provider
    ), patch("src.api.routes.chat.get_chatbot_agent", return_value=mock_agent):
        first_chat = client.post(
            "/api/v1/chat",
            headers={"Authorization": f"Bearer {first_user['access_token']}"},
            json={
                "session_id": "user-one-session",
                "message": "Hola",
                "use_rag": False,
            },
        )

    assert first_chat.status_code == 200

    first_sessions = client.get(
        "/api/v1/sessions",
        headers={"Authorization": f"Bearer {first_user['access_token']}"},
    )
    assert first_sessions.status_code == 200
    assert any(
        item["session_id"] == "user-one-session"
        for item in first_sessions.json()["sessions"]
    )

    second_sessions = client.get(
        "/api/v1/sessions",
        headers={"Authorization": f"Bearer {second_user['access_token']}"},
    )
    assert second_sessions.status_code == 200
    assert all(
        item["session_id"] != "user-one-session"
        for item in second_sessions.json()["sessions"]
    )


@patch("src.api.routes.chat.get_llm_provider")
@patch("src.api.routes.chat.get_chatbot_agent")
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
        "/api/v1/chat", json={"message": "Hello, world!", "use_rag": False}
    )

    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert "provider" in data
    assert "session_id" in data


def test_chat_endpoint_invalid_request():
    """Test chat endpoint with invalid request"""
    response = client.post("/api/v1/chat", json={})  # Missing required 'message' field

    assert response.status_code == 422  # Unprocessable Entity


def test_update_config_rejects_invalid_model_for_provider():
    """Config update should reject models outside the selected provider catalog."""
    admin = _register_user(ADMIN_EMAIL, ADMIN_PASSWORD, "Profesor")

    response = client.post(
        "/api/v1/config",
        headers=_auth_header(admin["access_token"]),
        json={"llm_provider": "anthropic", "model_name": "gpt-4-turbo-preview"},
    )

    assert response.status_code == 400
    assert "Invalid model for provider" in response.json()["detail"]


def test_update_config_sets_provider_default_model_when_model_omitted():
    """Switching providers without a model should apply the provider default."""
    admin = _register_user(ADMIN_EMAIL, ADMIN_PASSWORD, "Profesor")

    response = client.post(
        "/api/v1/config",
        headers=_auth_header(admin["access_token"]),
        json={"llm_provider": "openai"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["llm_provider"] == "openai"
    assert data["model_name"]


def test_update_config_requires_authentication():
    """Anonymous users cannot modify the global LLM configuration."""
    response = client.post("/api/v1/config", json={"llm_provider": "openai"})

    assert response.status_code == 401


def test_update_config_requires_admin():
    """Authenticated non-admin users cannot modify the global LLM configuration."""
    user = _register_user("student-config@example.com")

    response = client.post(
        "/api/v1/config",
        headers=_auth_header(user["access_token"]),
        json={"llm_provider": "openai"},
    )

    assert response.status_code == 403


@patch("src.api.dependencies.LLMFactory.create_provider")
def test_get_llm_provider_reuses_cached_instance(mock_create_provider):
    """Dependency helper should reuse identical provider instances across requests."""
    dependencies.reset_runtime_caches()
    mock_provider = Mock()
    mock_create_provider.return_value = mock_provider

    first = dependencies.get_llm_provider(
        provider_name="openai", model_name="gpt-4o-mini"
    )
    second = dependencies.get_llm_provider(
        provider_name="openai", model_name="gpt-4o-mini"
    )

    assert first is second
    mock_create_provider.assert_called_once()


@patch("src.api.dependencies._create_cached_rag_retriever")
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
            "llm_provider": "openai",
            "model_name": "gpt-4-turbo-preview",
            "use_rag": False,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert "response" in payload
    assert payload.get("used_rag") is False


@patch("src.api.routes.chat.get_llm_provider")
@patch("src.api.routes.chat.get_chatbot_agent")
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
    assert any(
        item["session_id"] == session_id for item in sessions_payload["sessions"]
    )

    history_response = client.get(f"/api/v1/sessions/{session_id}/history")
    assert history_response.status_code == 200
    history_payload = history_response.json()
    assert history_payload["session_id"] == session_id
    assert history_payload["total_messages"] >= 2


@patch("src.api.routes.chat.get_llm_provider")
@patch("src.api.routes.chat.get_chatbot_agent")
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


def test_session_document_upload_and_list_flow(tmp_path):
    """Session-scoped uploads should be listed only under the target session endpoint."""
    dependencies.reset_runtime_caches()

    mock_vector_store = Mock()
    mock_vector_store.add_documents.return_value = ["session_chat:s1:sample:0"]
    mock_doc_processor = Mock()
    mock_doc_processor.process_file.return_value = [
        Document(
            page_content="contenido",
            metadata={
                "file_hash": "sample",
                "file_type": "pdf",
                "scope": "session_chat",
                "session_id": "s1",
            },
        )
    ]

    sample_path = tmp_path / "sample.pdf"
    sample_path.write_bytes(b"%PDF-1.4 test")

    app.dependency_overrides[dependencies.get_vector_store] = lambda: mock_vector_store
    app.dependency_overrides[dependencies.get_document_processor] = (
        lambda: mock_doc_processor
    )

    with patch(
        "src.api.routes.documents.save_uploaded_file", return_value=str(sample_path)
    ):
        upload_response = client.post(
            "/api/v1/documents/sessions/s1/upload",
            files={"file": ("sample.pdf", b"dummy-pdf", "application/pdf")},
        )

    app.dependency_overrides.pop(dependencies.get_vector_store, None)
    app.dependency_overrides.pop(dependencies.get_document_processor, None)

    assert upload_response.status_code == 200
    payload = upload_response.json()
    assert payload["scope"] == "session_chat"
    assert payload["session_id"] == "s1"

    list_response = client.get("/api/v1/documents/sessions/s1")
    assert list_response.status_code == 200
    list_payload = list_response.json()
    assert list_payload["total_documents"] >= 1
    assert any(item.get("session_id") == "s1" for item in list_payload["documents"])


def test_session_document_delete_endpoint_with_metadata_filter():
    """Deleting a session-scoped document should call metadata-based deletion."""
    dependencies.reset_runtime_caches()

    manager = dependencies.get_session_manager()
    session_id = "test-session-doc-delete"
    document_id = "session_chat:test-session-doc-delete:to-delete"
    source_path = Path("data/uploads/sessions/test-session-doc-delete/to-delete.pdf")
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text("dummy", encoding="utf-8")

    manager.add_session_document(
        session_id=session_id,
        document_id=document_id,
        filename="to-delete.pdf",
        source=str(source_path),
        file_type="pdf",
        file_hash="to-delete",
    )

    mock_vector_store = Mock()
    mock_vector_store.delete_by_metadata.return_value = True
    app.dependency_overrides[dependencies.get_vector_store] = lambda: mock_vector_store

    response = client.delete(f"/api/v1/documents/sessions/{session_id}/{document_id}")

    app.dependency_overrides.pop(dependencies.get_vector_store, None)

    assert response.status_code == 200
    mock_vector_store.delete_by_metadata.assert_called_once()


def test_list_global_documents_requires_authentication():
    """Anonymous users cannot list the global RAG knowledge base."""
    response = client.get("/api/v1/documents")

    assert response.status_code == 401


def test_list_global_documents_requires_admin():
    """Authenticated non-admin users cannot list the global RAG knowledge base."""
    user = _register_user("student-list-docs@example.com")

    response = client.get(
        "/api/v1/documents", headers=_auth_header(user["access_token"])
    )

    assert response.status_code == 403


def test_admin_can_list_global_documents():
    """Admins can list the global RAG knowledge base."""
    admin = _register_user(ADMIN_EMAIL, ADMIN_PASSWORD, "Profesor")

    mock_vector_store = Mock()
    mock_vector_store.list_indexed_documents.return_value = []
    app.dependency_overrides[dependencies.get_vector_store] = lambda: mock_vector_store

    response = client.get(
        "/api/v1/documents", headers=_auth_header(admin["access_token"])
    )

    app.dependency_overrides.pop(dependencies.get_vector_store, None)

    assert response.status_code == 200


def test_upload_global_document_requires_admin():
    """Authenticated non-admin users cannot upload to the global RAG knowledge base."""
    user = _register_user("student-upload-docs@example.com")

    mock_vector_store = Mock()
    mock_doc_processor = Mock()
    app.dependency_overrides[dependencies.get_vector_store] = lambda: mock_vector_store
    app.dependency_overrides[dependencies.get_document_processor] = (
        lambda: mock_doc_processor
    )

    response = client.post(
        "/api/v1/documents/upload",
        headers=_auth_header(user["access_token"]),
        files={"file": ("sample.txt", b"contenido", "text/plain")},
    )

    app.dependency_overrides.pop(dependencies.get_vector_store, None)
    app.dependency_overrides.pop(dependencies.get_document_processor, None)

    assert response.status_code == 403


def test_delete_global_document_requires_admin():
    """Authenticated non-admin users cannot delete documents from the global RAG knowledge base."""
    user = _register_user("student-delete-doc@example.com")

    mock_vector_store = Mock()
    app.dependency_overrides[dependencies.get_vector_store] = lambda: mock_vector_store

    response = client.delete(
        "/api/v1/documents/some-doc-id",
        headers=_auth_header(user["access_token"]),
    )

    app.dependency_overrides.pop(dependencies.get_vector_store, None)

    assert response.status_code == 403


def test_clear_global_documents_requires_authentication():
    """Anonymous users cannot wipe the global RAG knowledge base."""
    response = client.delete("/api/v1/documents")

    assert response.status_code == 401


def test_clear_global_documents_requires_admin():
    """Authenticated non-admin users cannot wipe the global RAG knowledge base."""
    user = _register_user("student-clear-docs@example.com")

    mock_vector_store = Mock()
    app.dependency_overrides[dependencies.get_vector_store] = lambda: mock_vector_store

    response = client.delete(
        "/api/v1/documents", headers=_auth_header(user["access_token"])
    )

    app.dependency_overrides.pop(dependencies.get_vector_store, None)

    assert response.status_code == 403

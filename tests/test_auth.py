"""Tests for authentication: security utilities, user CRUD, auth endpoints, and prompt personalization."""

import pytest

from src.core.security import (
    hash_password,
    verify_password,
    generate_token,
    hash_token,
    assess_agile_level,
    build_user_profile_context,
)


# ---------------------------------------------------------------------------
# Security utilities
# ---------------------------------------------------------------------------


def test_hash_password_returns_hex_strings():
    pw_hash, salt = hash_password("mypassword")
    assert isinstance(pw_hash, str) and len(pw_hash) > 0
    assert isinstance(salt, str) and len(salt) > 0


def test_hash_password_is_not_plaintext():
    pw_hash, _ = hash_password("mypassword")
    assert pw_hash != "mypassword"


def test_verify_password_correct():
    pw_hash, salt = hash_password("correct_password")
    assert verify_password("correct_password", pw_hash, salt) is True


def test_verify_password_wrong():
    pw_hash, salt = hash_password("correct_password")
    assert verify_password("wrong_password", pw_hash, salt) is False


def test_verify_password_different_salts_do_not_collide():
    pw_hash1, salt1 = hash_password("same_password")
    pw_hash2, salt2 = hash_password("same_password")
    # Salts should differ, making hashes differ too
    assert salt1 != salt2
    assert pw_hash1 != pw_hash2


def test_generate_token_is_unique():
    tokens = {generate_token() for _ in range(20)}
    assert len(tokens) == 20


def test_generate_token_has_sufficient_length():
    token = generate_token()
    assert len(token) >= 32


def test_hash_token_is_deterministic():
    token = generate_token()
    assert hash_token(token) == hash_token(token)


def test_hash_token_different_inputs():
    assert hash_token("abc") != hash_token("xyz")


# ---------------------------------------------------------------------------
# assess_agile_level
# ---------------------------------------------------------------------------


def test_assess_agile_level_all_c_is_avanzado():
    result = assess_agile_level([
        {"question_number": i, "answer": "c"} for i in range(1, 6)
    ])
    assert result["level"] == 4
    assert result["label"] == "Avanzado"


def test_assess_agile_level_all_a_is_inicial():
    result = assess_agile_level([
        {"question_number": i, "answer": "a"} for i in range(1, 6)
    ])
    assert result["level"] == 2
    assert result["label"] == "Inicial"


def test_assess_agile_level_all_d_is_ninguno():
    result = assess_agile_level([
        {"question_number": i, "answer": "d"} for i in range(1, 6)
    ])
    assert result["level"] == 1
    assert result["label"] == "Ninguno"


def test_assess_agile_level_mixed_returns_average_level():
    result = assess_agile_level([
        {"question_number": 1, "answer": "c"},
        {"question_number": 2, "answer": "b"},
        {"question_number": 3, "answer": "b"},
        {"question_number": 4, "answer": "a"},
        {"question_number": 5, "answer": "a"},
    ])
    assert 1 <= result["level"] <= 4


def test_assess_agile_level_empty_answers_defaults():
    result = assess_agile_level([])
    assert result["level"] == 1


def test_assess_agile_level_accepts_plain_strings():
    result = assess_agile_level(["c", "c", "c", "c", "c"])
    assert result["level"] == 4


# ---------------------------------------------------------------------------
# build_user_profile_context
# ---------------------------------------------------------------------------


def test_build_user_profile_context_includes_name():
    user = {"full_name": "Ana García", "account_type": "Estudiante",
            "knowledge_level": 2, "agile_adoption_level": 2, "agile_adoption_label": "Inicial"}
    block = build_user_profile_context(user)
    assert "Ana García" in block


def test_build_user_profile_context_includes_account_type():
    user = {"full_name": "Carlos", "account_type": "Profesor",
            "knowledge_level": 4, "agile_adoption_level": 4, "agile_adoption_label": "Avanzado"}
    block = build_user_profile_context(user)
    assert "Profesor" in block


def test_build_user_profile_context_none_returns_empty():
    assert build_user_profile_context(None) == ""


def test_build_user_profile_context_empty_dict_does_not_crash():
    result = build_user_profile_context({})
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# SessionManager — user CRUD
# ---------------------------------------------------------------------------


@pytest.fixture
def sm(tmp_path):
    from src.memory.session_manager import SessionManager
    return SessionManager(db_path=str(tmp_path / "test_auth.db"))


SAMPLE_ANSWERS = [
    {"question_number": i, "answer": "b"} for i in range(1, 6)
]


def _register(sm, email="test@example.com", password="pass1234", full_name="Test User"):
    return sm.create_user(
        email=email,
        password=password,
        full_name=full_name,
        account_type="Estudiante",
        knowledge_level=2,
        questionnaire_answers=SAMPLE_ANSWERS,
    )


def test_create_user_returns_profile(sm):
    profile = _register(sm)
    assert profile["email"] == "test@example.com"
    assert profile["full_name"] == "Test User"
    assert profile["account_type"] == "Estudiante"
    assert 1 <= profile["agile_adoption_level"] <= 4


def test_create_user_duplicate_email_raises(sm):
    _register(sm)
    with pytest.raises(ValueError, match="email already registered"):
        _register(sm)


def test_create_user_email_is_case_insensitive(sm):
    sm.create_user("Upper@Example.COM", "pass1234", "Alice", "Estudiante", 1, SAMPLE_ANSWERS)
    user = sm.get_user_by_email("upper@example.com")
    assert user is not None


def test_get_user_by_email_not_found(sm):
    assert sm.get_user_by_email("ghost@example.com") is None


def test_get_user_profile_by_id(sm):
    profile = _register(sm)
    fetched = sm.get_user_profile(profile["user_id"])
    assert fetched is not None
    assert fetched["email"] == "test@example.com"


def test_issue_user_token_returns_string(sm):
    profile = _register(sm)
    token = sm.issue_user_token(profile["user_id"])
    assert isinstance(token, str) and len(token) > 10


def test_get_user_by_token_returns_profile(sm):
    profile = _register(sm)
    token = sm.issue_user_token(profile["user_id"])
    result = sm.get_user_by_token(token)
    assert result is not None
    assert result["user_id"] == profile["user_id"]


def test_get_user_by_invalid_token_returns_none(sm):
    assert sm.get_user_by_token("invalid-token-xyz") is None


def test_authenticate_user_correct_password(sm):
    _register(sm, email="auth@example.com", password="mySecret99")
    result = sm.authenticate_user("auth@example.com", "mySecret99")
    assert result is not None
    assert "access_token" in result
    assert result["email"] == "auth@example.com"


def test_authenticate_user_wrong_password_returns_none(sm):
    _register(sm, email="auth2@example.com", password="rightPass")
    result = sm.authenticate_user("auth2@example.com", "wrongPass")
    assert result is None


def test_authenticate_user_unknown_email_returns_none(sm):
    assert sm.authenticate_user("nobody@example.com", "anypass") is None


def test_session_linked_to_user(sm):
    profile = _register(sm)
    sm.create_session("session-abc", user_id=profile["user_id"])
    sessions = sm.list_sessions(user_id=profile["user_id"])
    assert any(s["session_id"] == "session-abc" for s in sessions)


def test_list_sessions_filters_by_user(sm):
    u1 = _register(sm, email="u1@x.com")
    u2 = sm.create_user("u2@x.com", "pass1234", "U2", "Profesor", 3, SAMPLE_ANSWERS)
    sm.create_session("sess-u1", user_id=u1["user_id"])
    sm.create_session("sess-u2", user_id=u2["user_id"])
    sm.append_message("sess-u1", "user", "hola")
    sm.append_message("sess-u2", "user", "hola")
    u1_sessions = sm.list_sessions(user_id=u1["user_id"])
    assert all(s["session_id"] == "sess-u1" for s in u1_sessions)


# ---------------------------------------------------------------------------
# Auth API endpoints
# ---------------------------------------------------------------------------


@pytest.fixture
def auth_client(tmp_path, monkeypatch):
    """TestClient backed by an isolated in-memory database."""
    from fastapi.testclient import TestClient
    from src.api import dependencies
    from src.main import app
    from src.memory.session_manager import SessionManager

    db_path = tmp_path / "auth_endpoint_test.db"
    test_sm = SessionManager(db_path=str(db_path))
    monkeypatch.setattr(dependencies, "_session_manager", test_sm)
    app.dependency_overrides.clear()
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()
    monkeypatch.setattr(dependencies, "_session_manager", None)


VALID_REGISTER = {
    "email": "student@test.com",
    "password": "password123",
    "full_name": "Test Student",
    "account_type": "Estudiante",
    "knowledge_level": 2,
    "questionnaire_answers": [
        {"question_number": i, "answer": "b"} for i in range(1, 6)
    ],
}


def test_register_returns_token(auth_client):
    resp = auth_client.post("/api/v1/auth/register", json=VALID_REGISTER)
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert data["user"]["email"] == "student@test.com"


def test_register_computes_agile_level(auth_client):
    """All 'c' answers → level 4 (Avanzado)."""
    payload = {**VALID_REGISTER, "email": "advanced@test.com",
               "questionnaire_answers": [{"question_number": i, "answer": "c"} for i in range(1, 6)]}
    resp = auth_client.post("/api/v1/auth/register", json=payload)
    assert resp.status_code == 200
    assert resp.json()["user"]["agile_adoption_level"] == 4


def test_register_duplicate_email_returns_400(auth_client):
    auth_client.post("/api/v1/auth/register", json=VALID_REGISTER)
    resp = auth_client.post("/api/v1/auth/register", json=VALID_REGISTER)
    assert resp.status_code == 400


def test_register_missing_fields_returns_422(auth_client):
    resp = auth_client.post("/api/v1/auth/register", json={"email": "x@x.com"})
    assert resp.status_code == 422


def test_login_success(auth_client):
    auth_client.post("/api/v1/auth/register", json=VALID_REGISTER)
    resp = auth_client.post("/api/v1/auth/login", json={
        "email": "student@test.com", "password": "password123"
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert data["user"]["email"] == "student@test.com"


def test_login_wrong_password_returns_401(auth_client):
    auth_client.post("/api/v1/auth/register", json=VALID_REGISTER)
    resp = auth_client.post("/api/v1/auth/login", json={
        "email": "student@test.com", "password": "wrongpassword"
    })
    assert resp.status_code == 401


def test_login_unknown_email_returns_401(auth_client):
    resp = auth_client.post("/api/v1/auth/login", json={
        "email": "ghost@test.com", "password": "anypass"
    })
    assert resp.status_code == 401


def test_me_without_token_returns_401(auth_client):
    resp = auth_client.get("/api/v1/auth/me")
    assert resp.status_code == 401


def test_me_with_valid_token_returns_profile(auth_client):
    reg = auth_client.post("/api/v1/auth/register", json=VALID_REGISTER).json()
    token = reg["access_token"]
    resp = auth_client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    assert resp.json()["email"] == "student@test.com"


def test_me_with_invalid_token_returns_401(auth_client):
    resp = auth_client.get("/api/v1/auth/me", headers={"Authorization": "Bearer bogus-token"})
    assert resp.status_code == 401


def test_register_response_includes_is_admin(auth_client):
    response = auth_client.post(
        "/api/v1/auth/register",
        json={
            "email": "newuser@example.com",
            "password": "securepass",
            "full_name": "New User",
            "account_type": "Estudiante",
            "knowledge_level": 2,
            "questionnaire_answers": [],
        },
    )
    assert response.status_code == 200
    assert "is_admin" in response.json()["user"]
    assert response.json()["user"]["is_admin"] is False


def test_sessions_filtered_by_token(auth_client):
    """Sessions endpoint should return only the requesting user's sessions."""
    from unittest.mock import Mock, patch

    u1 = auth_client.post("/api/v1/auth/register", json=VALID_REGISTER).json()
    u2_payload = {**VALID_REGISTER, "email": "other@test.com", "full_name": "Other User"}
    u2 = auth_client.post("/api/v1/auth/register", json=u2_payload).json()

    mock_agent = Mock()
    mock_agent.chat.return_value = {
        "response": "ok", "provider": "openai", "model": "gpt-4", "used_rag": False, "sources": [],
    }

    with patch("src.api.routes.chat.get_llm_provider", return_value=Mock()), \
         patch("src.api.routes.chat.get_chatbot_agent", return_value=mock_agent):

        auth_client.post("/api/v1/chat",
                         headers={"Authorization": f"Bearer {u1['access_token']}"},
                         json={"message": "Hola", "use_rag": False, "session_id": "sess-u1"})

    sessions_u1 = auth_client.get("/api/v1/sessions",
                                   headers={"Authorization": f"Bearer {u1['access_token']}"}).json()
    sessions_u2 = auth_client.get("/api/v1/sessions",
                                   headers={"Authorization": f"Bearer {u2['access_token']}"}).json()

    assert any(s["session_id"] == "sess-u1" for s in sessions_u1["sessions"])
    assert all(s["session_id"] != "sess-u1" for s in sessions_u2["sessions"])


# ---------------------------------------------------------------------------
# Prompt personalization integration
# ---------------------------------------------------------------------------


def test_prompt_includes_user_profile():
    from src.core.prompt_manager import PromptManager

    pm = PromptManager()
    user_note = "Usuario: Demo; perfil: Estudiante; nivel declarado: 2 (Inicial)"
    prompt = pm.build_direct_prompt(
        message="¿Qué es Scrum?",
        user_profile_note=user_note,
    )
    assert "Demo" in prompt or "Estudiante" in prompt
    assert "¿Qué es Scrum?" in prompt


def test_prompt_without_user_profile_still_works():
    from src.core.prompt_manager import PromptManager

    pm = PromptManager()
    prompt = pm.build_direct_prompt(message="¿Qué es Kanban?")
    assert "¿Qué es Kanban?" in prompt


def test_socratic_prompt_includes_user_profile():
    from src.core.prompt_manager import PromptManager

    pm = PromptManager()
    user_note = "Usuario: Lucía; perfil: Profesor; nivel declarado: 4 (Avanzado)"
    prompt = pm.build_socratic_prompt(
        message="¿Cómo mejorar la velocidad de nuestro equipo?",
        user_profile_note=user_note,
    )
    assert "Lucía" in prompt or "Profesor" in prompt
    assert "velocidad" in prompt


# ---------------------------------------------------------------------------
# is_admin field
# ---------------------------------------------------------------------------


class TestIsAdminField:
    def test_create_user_default_not_admin(self, sm):
        profile = sm.create_user(
            email="regular@test.com",
            password="password123",
            full_name="Regular User",
            account_type="Estudiante",
            knowledge_level=2,
        )
        assert profile["is_admin"] is False

    def test_create_user_with_admin_flag(self, sm):
        profile = sm.create_user(
            email="admin@test.com",
            password="password123",
            full_name="Admin User",
            account_type="Profesor",
            knowledge_level=3,
            is_admin=True,
        )
        assert profile["is_admin"] is True

    def test_is_admin_persisted_across_lookup(self, sm):
        sm.create_user(
            email="boss@test.com",
            password="password123",
            full_name="Boss",
            account_type="Profesor",
            knowledge_level=4,
            is_admin=True,
        )
        profile = sm.get_user_by_email("boss@test.com")
        assert profile["is_admin"] is True

    def test_create_user_explicit_false_is_not_admin(self, sm):
        profile = sm.create_user(
            email="explicit_false@test.com",
            password="password123",
            full_name="Explicit False",
            account_type="Estudiante",
            knowledge_level=2,
            is_admin=False,
        )
        assert profile["is_admin"] is False


class TestDeleteUser:
    def test_delete_user_returns_session_ids(self, sm):
        profile = sm.create_user(
            email="todelete@test.com",
            password="password123",
            full_name="To Delete",
            account_type="Estudiante",
            knowledge_level=1,
        )
        uid = profile["user_id"]
        sid = "session-del-1"
        sm.create_session(session_id=sid, user_id=uid)
        sm.append_message(session_id=sid, role="user", text="hello")

        result = sm.delete_user(uid)
        assert isinstance(result, list)
        assert sid in result
        # Verify cascade: messages and session actually removed from DB
        assert sm.get_messages(sid) == []
        assert sm.get_session_record(sid) is None

    def test_delete_user_removes_user_from_db(self, sm):
        profile = sm.create_user(
            email="gone@test.com",
            password="password123",
            full_name="Gone",
            account_type="Estudiante",
            knowledge_level=1,
        )
        uid = profile["user_id"]
        sm.delete_user(uid)
        assert sm.get_user_profile(uid) is None

    def test_delete_user_not_found_returns_none(self, sm):
        result = sm.delete_user("nonexistent-id")
        assert result is None

    def test_list_users_returns_all(self, sm):
        sm.create_user(
            email="user1@test.com",
            password="password123",
            full_name="User One",
            account_type="Estudiante",
            knowledge_level=1,
        )
        sm.create_user(
            email="user2@test.com",
            password="password123",
            full_name="User Two",
            account_type="Profesor",
            knowledge_level=2,
        )
        users = sm.list_users()
        emails = [u["email"] for u in users]
        assert "user1@test.com" in emails
        assert "user2@test.com" in emails

    def test_count_users(self, sm):
        initial = sm.count_users()
        sm.create_user(
            email="count@test.com",
            password="password123",
            full_name="Count Me",
            account_type="Estudiante",
            knowledge_level=1,
        )
        assert sm.count_users() == initial + 1

    def test_delete_user_with_no_sessions_returns_empty_list(self, sm):
        profile = sm.create_user(
            email="nosessions@test.com",
            password="password123",
            full_name="No Sessions",
            account_type="Estudiante",
            knowledge_level=1,
        )
        uid = profile["user_id"]
        result = sm.delete_user(uid)
        assert result == []
        assert sm.get_user_profile(uid) is None


class TestDeleteMe:
    def _register(self, client, email="del@example.com"):
        resp = client.post(
            "/api/v1/auth/register",
            json={
                "email": email,
                "password": "password123",
                "full_name": "Del User",
                "account_type": "Estudiante",
                "knowledge_level": 1,
                "questionnaire_answers": [],
            },
        )
        return resp.json()["access_token"]

    def test_delete_me_returns_200(self, auth_client):
        token = self._register(auth_client)
        resp = auth_client.delete(
            "/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"}
        )
        assert resp.status_code == 200
        assert "eliminada" in resp.json()["message"].lower()

    def test_delete_me_invalidates_token(self, auth_client):
        token = self._register(auth_client, email="ghost@example.com")
        auth_client.delete(
            "/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"}
        )
        resp = auth_client.get(
            "/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"}
        )
        assert resp.status_code == 401

    def test_delete_me_requires_auth(self, auth_client):
        resp = auth_client.delete("/api/v1/auth/me")
        assert resp.status_code == 401

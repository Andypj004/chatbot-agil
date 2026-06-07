"""Tests for admin user management endpoints."""

import pytest
from fastapi.testclient import TestClient

from src.memory.session_manager import SessionManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ADMIN_EMAIL = "admin@testdomain.com"
ADMIN_PASSWORD = "adminpassword"

REGULAR_EMAIL = "regular@testdomain.com"
REGULAR_PASSWORD = "regularpassword"


@pytest.fixture
def admin_sm(tmp_path):
    return SessionManager(db_path=str(tmp_path / "admin_test.db"))


@pytest.fixture
def admin_client(tmp_path, monkeypatch):
    from src.api import dependencies
    from src.core import config as cfg_module
    from src.main import app

    db_path = tmp_path / "admin_endpoint_test.db"
    test_sm = SessionManager(db_path=str(db_path))
    monkeypatch.setattr(dependencies, "_session_manager", test_sm)
    monkeypatch.setattr(cfg_module.settings, "admin_emails", ADMIN_EMAIL)

    with TestClient(app) as client:
        yield client


def _register(client, email, password="password123", account_type="Estudiante"):
    resp = client.post(
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
    assert resp.status_code == 200, resp.text
    return resp.json()


# ---------------------------------------------------------------------------
# Admin list users
# ---------------------------------------------------------------------------


class TestAdminListUsers:
    def test_list_users_requires_auth(self, admin_client):
        resp = admin_client.get("/api/v1/admin/users")
        assert resp.status_code == 401

    def test_list_users_requires_admin(self, admin_client):
        data = _register(admin_client, REGULAR_EMAIL)
        token = data["access_token"]
        resp = admin_client.get(
            "/api/v1/admin/users", headers={"Authorization": f"Bearer {token}"}
        )
        assert resp.status_code == 403

    def test_admin_can_list_users(self, admin_client):
        admin_data = _register(admin_client, ADMIN_EMAIL, ADMIN_PASSWORD, "Profesor")
        _register(admin_client, REGULAR_EMAIL)
        admin_token = admin_data["access_token"]

        resp = admin_client.get(
            "/api/v1/admin/users", headers={"Authorization": f"Bearer {admin_token}"}
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "users" in body
        assert "total" in body
        emails = [u["email"] for u in body["users"]]
        assert ADMIN_EMAIL in emails
        assert REGULAR_EMAIL in emails

    def test_admin_flag_shown_in_list(self, admin_client):
        admin_data = _register(admin_client, ADMIN_EMAIL, ADMIN_PASSWORD, "Profesor")
        admin_token = admin_data["access_token"]

        resp = admin_client.get(
            "/api/v1/admin/users", headers={"Authorization": f"Bearer {admin_token}"}
        )
        admin_user = next(
            u for u in resp.json()["users"] if u["email"] == ADMIN_EMAIL
        )
        assert admin_user["is_admin"] is True


# ---------------------------------------------------------------------------
# Admin delete user
# ---------------------------------------------------------------------------


class TestAdminDeleteUser:
    def test_admin_can_delete_user(self, admin_client):
        admin_data = _register(admin_client, ADMIN_EMAIL, ADMIN_PASSWORD, "Profesor")
        regular_data = _register(admin_client, REGULAR_EMAIL)
        admin_token = admin_data["access_token"]
        regular_id = regular_data["user"]["user_id"]

        resp = admin_client.delete(
            f"/api/v1/admin/users/{regular_id}",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert resp.status_code == 200
        assert "eliminado" in resp.json()["message"].lower()

    def test_admin_delete_nonexistent_user_returns_404(self, admin_client):
        admin_data = _register(admin_client, ADMIN_EMAIL, ADMIN_PASSWORD, "Profesor")
        admin_token = admin_data["access_token"]

        resp = admin_client.delete(
            "/api/v1/admin/users/nonexistent-id",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert resp.status_code == 404

    def test_regular_user_cannot_delete_another_user(self, admin_client):
        _register(admin_client, ADMIN_EMAIL, ADMIN_PASSWORD, "Profesor")
        regular_data = _register(admin_client, REGULAR_EMAIL)
        other_data = _register(admin_client, "other@testdomain.com")
        regular_token = regular_data["access_token"]
        other_id = other_data["user"]["user_id"]

        resp = admin_client.delete(
            f"/api/v1/admin/users/{other_id}",
            headers={"Authorization": f"Bearer {regular_token}"},
        )
        assert resp.status_code == 403

    def test_admin_delete_requires_auth(self, admin_client):
        admin_data = _register(admin_client, ADMIN_EMAIL, ADMIN_PASSWORD, "Profesor")
        admin_id = admin_data["user"]["user_id"]
        resp = admin_client.delete(f"/api/v1/admin/users/{admin_id}")
        assert resp.status_code == 401

    def test_admin_cannot_delete_self_via_admin_route(self, admin_client):
        admin_data = _register(admin_client, ADMIN_EMAIL, ADMIN_PASSWORD, "Profesor")
        admin_token = admin_data["access_token"]
        admin_id = admin_data["user"]["user_id"]

        resp = admin_client.delete(
            f"/api/v1/admin/users/{admin_id}",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert resp.status_code == 400
        assert "propia cuenta" in resp.json()["detail"].lower()

    def test_deleted_user_token_is_invalidated(self, admin_client):
        admin_data = _register(admin_client, ADMIN_EMAIL, ADMIN_PASSWORD, "Profesor")
        regular_data = _register(admin_client, REGULAR_EMAIL)
        admin_token = admin_data["access_token"]
        regular_token = regular_data["access_token"]
        regular_id = regular_data["user"]["user_id"]

        admin_client.delete(
            f"/api/v1/admin/users/{regular_id}",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        resp = admin_client.get(
            "/api/v1/auth/me", headers={"Authorization": f"Bearer {regular_token}"}
        )
        assert resp.status_code == 401

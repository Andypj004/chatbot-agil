"""Smoke tests for the critical runtime API surface.

These checks are intentionally small and target the endpoints that should stay
reachable after startup and router registration.
"""

from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from src.main import app


client = TestClient(app)


def test_health_and_config_smoke():
    """Core operational endpoints should respond successfully."""
    health_response = client.get("/api/v1/health")
    assert health_response.status_code == 200

    config_response = client.get("/api/v1/config")
    assert config_response.status_code == 200


@patch("src.api.routes.forms.form_manager.start_form")
def test_forms_start_route_smoke(mock_start_form):
    """The forms start endpoint should be reachable at the mounted API path."""
    mock_start_form.return_value = "Primera pregunta"

    response = client.post(
        "/api/v1/forms/start",
        json={"form_id": "project_brief", "session_id": "audit-session"},
    )

    assert response.status_code == 200
    assert response.json()["question"] == "Primera pregunta"


@patch("src.api.routes.forms.form_manager.answer")
def test_forms_answer_route_smoke(mock_answer):
    """The forms answer endpoint should be reachable at the mounted API path."""
    mock_answer.return_value = {"ok": True, "next_question": None}

    response = client.post(
        "/api/v1/forms/answer",
        json={
            "form_id": "project_brief",
            "session_id": "audit-session",
            "name": "project_name",
            "value": "Agile Chatbot",
        },
    )

    assert response.status_code == 200
    assert response.json()["ok"] is True

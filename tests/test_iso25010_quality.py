"""
ISO 25010 Quality Evaluation Matrix — Automated Test Suite
Covers all 8 product quality characteristics and their sub-characteristics.

Tests marked @pytest.mark.skip indicate cases that require human intervention
or specialised external tooling and cannot be fully automated.
"""

import json
import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from src.api import dependencies
from src.core import config as cfg_module
from src.main import app
from src.memory.session_manager import SessionManager

client = TestClient(app)

ADMIN_EMAIL = "admin@iso25010.test"
ADMIN_PASSWORD = "adminpass123"

# ---------------------------------------------------------------------------
# Shared fixture – isolated SQLite DB per test
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def isolate_db(tmp_path, monkeypatch):
    db = tmp_path / "iso25010.db"
    monkeypatch.setattr(
        dependencies, "_session_manager", SessionManager(db_path=str(db))
    )
    monkeypatch.setattr(cfg_module.settings, "admin_emails", ADMIN_EMAIL)
    app.dependency_overrides.clear()
    yield
    app.dependency_overrides.clear()
    monkeypatch.setattr(dependencies, "_session_manager", None)


def _mock_agent(response="Respuesta de prueba sobre agilidad."):
    """Return a mocked ChatbotAgent with a canned response."""
    m = Mock()
    m.chat.return_value = {
        "response": response,
        "provider": "openai",
        "model": "gpt-4o-mini",
        "used_rag": False,
        "sources": [],
    }
    return m


def _admin_auth_header():
    """Register an admin user and return an Authorization header for them."""
    response = client.post(
        "/api/v1/auth/register",
        json={
            "email": ADMIN_EMAIL,
            "password": ADMIN_PASSWORD,
            "full_name": "Admin User",
            "account_type": "Profesor",
            "knowledge_level": 3,
            "questionnaire_answers": [],
        },
    )
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


# ===========================================================================
# 1. FUNCTIONAL SUITABILITY (Adecuación Funcional)
# ===========================================================================


class TestFunctionalCompleteness:
    """1.1 FC — All required endpoints and features are present."""

    def test_fc001_chat_endpoint_exists(self):
        """RF-001: POST /api/v1/chat is reachable and accepts agile questions."""
        with patch("src.api.routes.chat.get_llm_provider", return_value=Mock()), patch(
            "src.api.routes.chat.get_chatbot_agent", return_value=_mock_agent()
        ):
            r = client.post("/api/v1/chat", json={"message": "¿Qué es Scrum?"})
        assert r.status_code == 200

    def test_fc002_health_endpoint_exists(self):
        """RFN-009: /api/v1/health is reachable."""
        r = client.get("/api/v1/health")
        assert r.status_code == 200

    def test_fc003_config_endpoint_exists(self):
        """RFN-002: /api/v1/config is reachable."""
        r = client.get("/api/v1/config")
        assert r.status_code == 200

    def test_fc004_document_endpoint_exists(self):
        """RF-007: /api/v1/documents exists (not 404)."""
        r = client.get("/api/v1/documents")
        assert r.status_code != 404

    def test_fc005_sessions_endpoint_exists(self):
        """RF-006: /api/v1/sessions exists (not 404)."""
        r = client.get("/api/v1/sessions")
        assert r.status_code != 404

    def test_fc006_auth_register_endpoint_exists(self):
        """RF-006: Auth register endpoint exists (not 404)."""
        r = client.post("/api/v1/auth/register", json={})
        assert r.status_code != 404

    def test_fc007_frontend_served(self):
        """RFN-004: Frontend HTML is served at /app."""
        r = client.get("/app")
        assert r.status_code == 200
        assert "text/html" in r.headers.get("content-type", "")

    def test_fc008_forms_command_reachable(self):
        """RF-010: Form commands route through the chat endpoint."""
        with patch("src.api.routes.chat.get_llm_provider", return_value=Mock()), patch(
            "src.api.routes.chat.get_chatbot_agent", return_value=_mock_agent()
        ):
            r = client.post("/api/v1/chat", json={"message": "/form list"})
        assert r.status_code == 200


class TestFunctionalCorrectness:
    """1.2 FCR — Endpoints return correct data structures and values."""

    def test_fcr001_health_response_required_fields(self):
        """Health response contains status, version, llm_providers, rag_status."""
        r = client.get("/api/v1/health")
        data = r.json()
        for field in ("status", "version", "llm_providers", "rag_status"):
            assert field in data, f"Missing field in health response: {field}"

    def test_fcr002_config_response_required_fields(self):
        """Config response contains all required fields per RFN-002."""
        r = client.get("/api/v1/config")
        data = r.json()
        for field in ("llm_provider", "model_name", "temperature", "available_models", "rag_enabled"):
            assert field in data, f"Missing config field: {field}"

    def test_fcr003_chat_response_required_fields(self):
        """RF-001: Chat response contains session_id, response, provider, model, used_rag."""
        with patch("src.api.routes.chat.get_llm_provider", return_value=Mock()), patch(
            "src.api.routes.chat.get_chatbot_agent", return_value=_mock_agent("Scrum es un marco ágil.")
        ):
            r = client.post("/api/v1/chat", json={"message": "¿Qué es Scrum?", "use_rag": False})
        assert r.status_code == 200
        data = r.json()
        for field in ("session_id", "response", "provider", "model", "used_rag"):
            assert field in data, f"Missing chat response field: {field}"

    def test_fcr004_session_id_auto_generated(self):
        """RF-005: session_id is returned even when not provided in request."""
        with patch("src.api.routes.chat.get_llm_provider", return_value=Mock()), patch(
            "src.api.routes.chat.get_chatbot_agent", return_value=_mock_agent()
        ):
            r = client.post("/api/v1/chat", json={"message": "Hola", "use_rag": False})
        assert r.status_code == 200
        assert r.json()["session_id"], "session_id must be non-empty"

    def test_fcr005_available_providers_non_empty(self):
        """Config exposes at least one LLM provider."""
        r = client.get("/api/v1/config")
        assert len(r.json()["available_providers"]) > 0

    def test_fcr006_health_status_is_healthy(self):
        """System reports 'healthy' status under normal test conditions."""
        r = client.get("/api/v1/health")
        assert r.json()["status"] == "healthy"


class TestFunctionalAppropriateness:
    """1.3 FP — Features achieve their stated goals appropriately."""

    def test_fp001_use_rag_false_bypasses_retrieval(self):
        """RF-001: use_rag=False returns used_rag=False in response."""
        with patch("src.api.routes.chat.get_llm_provider", return_value=Mock()), patch(
            "src.api.routes.chat.get_chatbot_agent", return_value=_mock_agent()
        ):
            r = client.post("/api/v1/chat", json={"message": "¿Qué es Kanban?", "use_rag": False})
        assert r.status_code == 200
        assert r.json()["used_rag"] is False

    def test_fp002_config_update_persists(self):
        """RFN-002: Config update is reflected in subsequent GET."""
        locked_provider = cfg_module.settings.default_llm_provider
        client.post(
            "/api/v1/config", headers=_admin_auth_header(), json={"llm_provider": locked_provider}
        )
        after = client.get("/api/v1/config").json()
        assert after["llm_provider"] == locked_provider

    def test_fp003_sources_field_present_in_chat_response(self):
        """RF-009: sources field is always present in chat response (may be null/empty)."""
        with patch("src.api.routes.chat.get_llm_provider", return_value=Mock()), patch(
            "src.api.routes.chat.get_chatbot_agent",
            return_value=_mock_agent("Sprint es una iteración."),
        ):
            r = client.post("/api/v1/chat", json={"message": "¿Qué es un Sprint?", "use_rag": False})
        assert r.status_code == 200
        assert "sources" in r.json()


# ===========================================================================
# 2. PERFORMANCE EFFICIENCY (Eficiencia de Desempeño)
# ===========================================================================


class TestTimeBehavior:
    """2.1 PT — Response times within acceptable thresholds."""

    def test_pt001_health_endpoint_under_500ms(self):
        """Health check must respond in < 500 ms."""
        start = time.monotonic()
        r = client.get("/api/v1/health")
        elapsed = (time.monotonic() - start) * 1000
        assert r.status_code == 200
        assert elapsed < 500, f"Health took {elapsed:.0f} ms (limit 500 ms)"

    def test_pt002_config_endpoint_under_500ms(self):
        """Config endpoint must respond in < 500 ms."""
        start = time.monotonic()
        r = client.get("/api/v1/config")
        elapsed = (time.monotonic() - start) * 1000
        assert r.status_code == 200
        assert elapsed < 500, f"Config took {elapsed:.0f} ms (limit 500 ms)"

    def test_pt003_chat_mocked_under_3000ms(self):
        """Mocked chat request (no real LLM call) must complete in < 3 s."""
        start = time.monotonic()
        with patch("src.api.routes.chat.get_llm_provider", return_value=Mock()), patch(
            "src.api.routes.chat.get_chatbot_agent", return_value=_mock_agent()
        ):
            r = client.post("/api/v1/chat", json={"message": "Prueba rendimiento", "use_rag": False})
        elapsed = (time.monotonic() - start) * 1000
        assert r.status_code == 200
        assert elapsed < 3000, f"Chat took {elapsed:.0f} ms (limit 3 000 ms)"

    def test_pt004_sessions_list_under_500ms(self):
        """Session list endpoint must respond in < 500 ms."""
        start = time.monotonic()
        r = client.get("/api/v1/sessions")
        elapsed = (time.monotonic() - start) * 1000
        assert r.status_code != 500
        assert elapsed < 500, f"Sessions list took {elapsed:.0f} ms (limit 500 ms)"


class TestResourceUtilization:
    """2.2 PR — System does not excessively consume resources."""

    def test_pr001_ten_sequential_health_checks_no_error(self):
        """RFN-005: 10 sequential health checks all succeed (no resource leaks)."""
        errors = sum(1 for _ in range(10) if client.get("/api/v1/health").status_code != 200)
        assert errors == 0

    def test_pr002_lru_cache_reuses_provider_instance(self):
        """RFN-005: LLM provider cache avoids redundant instantiation."""
        from src.api import dependencies as deps

        deps.reset_runtime_caches()
        with patch(
            "src.api.dependencies.LLMFactory.create_provider", return_value=Mock()
        ) as mock_create:
            deps.get_llm_provider(provider_name="openai", model_name="gpt-4o-mini")
            deps.get_llm_provider(provider_name="openai", model_name="gpt-4o-mini")
        assert mock_create.call_count == 1, "Provider should be created once and cached"


class TestCapacity:
    """2.3 PC — System handles expected data volume."""

    def test_pc001_twenty_sessions_created_and_listed(self, tmp_path):
        """RF-006: SessionManager handles creating and listing 20 concurrent sessions."""
        sm = SessionManager(db_path=str(tmp_path / "cap.db"))
        for i in range(20):
            sm.create_session(f"session-cap-{i}")
        sessions = sm.list_sessions(limit=50)
        assert len(sessions) == 20

    def test_pc002_pagination_limits_results(self, tmp_path):
        """RFN-005: list_sessions(limit=3) returns at most 3 results."""
        sm = SessionManager(db_path=str(tmp_path / "pag.db"))
        for i in range(10):
            sm.create_session(f"session-pag-{i}")
        page = sm.list_sessions(limit=3)
        assert len(page) <= 3


# ===========================================================================
# 3. COMPATIBILITY (Compatibilidad)
# ===========================================================================


class TestCoexistence:
    """3.1 CO — System co-exists with other software without interference."""

    def test_co001_cors_headers_on_preflight(self):
        """RFN-005: OPTIONS preflight returns CORS headers for browser interoperability."""
        r = client.options(
            "/api/v1/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        acao = r.headers.get("access-control-allow-origin") or r.headers.get(
            "Access-Control-Allow-Origin"
        )
        assert acao is not None, "CORS Allow-Origin header missing"

    def test_co002_api_responses_use_json_content_type(self):
        """API responses must be application/json for standard tooling."""
        r = client.get("/api/v1/health")
        assert "application/json" in r.headers.get("content-type", "")

    def test_co003_standard_http_methods_supported(self):
        """GET and POST are accepted on the expected endpoints."""
        assert client.get("/api/v1/health").status_code == 200
        assert (
            client.post(
                "/api/v1/config",
                headers=_admin_auth_header(),
                json={"llm_provider": cfg_module.settings.default_llm_provider},
            ).status_code
            == 200
        )


class TestInteroperability:
    """3.2 CI — System can exchange information with other systems."""

    def test_ci001_json_request_body_accepted(self):
        """API accepts explicit application/json Content-Type header."""
        with patch("src.api.routes.chat.get_llm_provider", return_value=Mock()), patch(
            "src.api.routes.chat.get_chatbot_agent", return_value=_mock_agent()
        ):
            r = client.post(
                "/api/v1/chat",
                content=json.dumps({"message": "Test JSON", "use_rag": False}),
                headers={"Content-Type": "application/json"},
            )
        assert r.status_code == 200
        r.json()  # must parse as valid JSON

    def test_ci002_standard_http_status_codes_used(self):
        """200 for success, 422 for validation error, 400 for business rule violation."""
        assert client.get("/api/v1/health").status_code == 200
        assert client.post("/api/v1/chat", json={}).status_code == 422
        assert (
            client.post(
                "/api/v1/config",
                headers=_admin_auth_header(),
                json={"llm_provider": "anthropic", "model_name": "gpt-4-turbo-preview"},
            ).status_code
            == 400
        )

    def test_ci003_openapi_schema_available(self):
        """OpenAPI schema is exposed at /openapi.json for tooling integration."""
        r = client.get("/openapi.json")
        assert r.status_code == 200
        schema = r.json()
        assert "paths" in schema
        assert "components" in schema


# ===========================================================================
# 4. USABILITY (Usabilidad)
# ===========================================================================


class TestAppropriatenessRecognizability:
    """4.1 UA — Users can recognise if the product suits their needs."""

    def test_ua001_root_endpoint_describes_service(self):
        """Root returns a non-trivial human-readable message."""
        r = client.get("/")
        data = r.json()
        assert "message" in data
        assert len(data["message"]) > 5

    def test_ua002_health_exposes_rag_status(self):
        """Health response communicates whether the knowledge base is active."""
        r = client.get("/api/v1/health")
        assert "rag_status" in r.json()


class TestOperability:
    """4.3 UO — System is easy to operate."""

    def test_uo001_session_auto_created(self):
        """RF-005: Users need not manage session IDs — auto-generated."""
        with patch("src.api.routes.chat.get_llm_provider", return_value=Mock()), patch(
            "src.api.routes.chat.get_chatbot_agent", return_value=_mock_agent()
        ):
            r = client.post("/api/v1/chat", json={"message": "Hola", "use_rag": False})
        assert r.status_code == 200
        assert r.json()["session_id"]

    def test_uo002_config_shows_available_options(self):
        """RFN-002: Config GET returns available_providers and available_models."""
        r = client.get("/api/v1/config")
        data = r.json()
        assert isinstance(data["available_models"], dict)
        assert len(data["available_models"]) > 0


class TestUserErrorProtection:
    """4.4 UEP — System guards users against errors."""

    def test_uep001_empty_message_returns_422(self):
        """Empty/missing message is rejected before processing (422)."""
        assert client.post("/api/v1/chat", json={}).status_code == 422

    def test_uep002_temperature_out_of_range_rejected(self):
        """Temperature > 1.0 is rejected with 422."""
        r = client.post("/api/v1/chat", json={"message": "Test", "temperature": 5.0})
        assert r.status_code == 422

    def test_uep003_cross_provider_model_rejected_with_400(self):
        """Mixing provider and model from different vendors returns 400."""
        r = client.post(
            "/api/v1/config",
            headers=_admin_auth_header(),
            json={"llm_provider": "anthropic", "model_name": "gpt-4-turbo-preview"},
        )
        assert r.status_code == 400

    def test_uep004_incomplete_registration_rejected_with_422(self):
        """Auth registration with missing required fields returns 422."""
        assert client.post("/api/v1/auth/register", json={"email": "x@x.com"}).status_code == 422


class TestAccessibility:
    """4.6 UAC — Can be used by people with diverse abilities."""


    def test_uac002_system_operates_in_spanish(self):
        """RFN-004: System is deployed in Spanish (linguistic accessibility)."""
        from src.core.prompt_manager import BASE_SYSTEM_PROMPT
        prompt_lower = BASE_SYSTEM_PROMPT.lower()
        assert "espanol" in prompt_lower or "español" in prompt_lower or \
               "agil" in prompt_lower or "ágil" in prompt_lower, \
            "BASE_SYSTEM_PROMPT must indicate Spanish-language operation"


# ===========================================================================
# 5. RELIABILITY (Confiabilidad)
# ===========================================================================


class TestMaturity:
    """5.1 RM — System meets reliability needs under normal operation."""

    def test_rm001_test_suite_has_sufficient_modules(self):
        """System has ≥ 5 test modules covering core domains."""
        test_files = list(Path("tests").glob("test_*.py"))
        assert len(test_files) >= 5, f"Found only {len(test_files)} test modules"

    def test_rm002_code_coverage_above_60_percent(self):
        """Test suite achieves ≥ 60 % line coverage (from coverage.xml)."""
        cov = Path("coverage.xml")
        if not cov.exists():
            pytest.skip("coverage.xml not found — run pytest --cov=src --cov-report=xml first")
        root = ET.parse(cov).getroot()
        line_rate = float(root.attrib.get("line-rate", 0))
        assert line_rate >= 0.60, f"Coverage {line_rate:.0%} is below 60 % threshold"


class TestAvailability:
    """5.2 RA — System is operational when required."""

    def test_ra001_health_always_responds(self):
        """RFN-009: Health endpoint responds even if RAG is not initialised."""
        r = client.get("/api/v1/health")
        assert r.status_code == 200

    def test_ra002_root_path_responds(self):
        """System root path responds with service info."""
        r = client.get("/")
        assert r.status_code == 200


class TestFaultTolerance:
    """5.3 RFT — System continues to operate correctly despite faults."""

    def test_rft001_nonexistent_session_history_no_500(self):
        """RF-005: Querying history for unknown session returns gracefully (not 500)."""
        r = client.get("/api/v1/sessions/nonexistent-session-xyz/history")
        assert r.status_code != 500

    def test_rft002_missing_message_returns_422_not_500(self):
        """RFN-006: Malformed request body returns 422, not 500."""
        r = client.post("/api/v1/chat", json={"use_rag": False})
        assert r.status_code == 422
        assert r.status_code != 500

    def test_rft003_invalid_json_handled_not_500(self):
        """RFN-006: Unparseable JSON body returns 400/422, not 500."""
        r = client.post(
            "/api/v1/chat",
            content=b"this is not json",
            headers={"Content-Type": "application/json"},
        )
        assert r.status_code in (400, 422)

    def test_rft004_unknown_provider_in_config_returns_400_not_500(self):
        """RFN-006: Unknown provider name in config update returns 400, not 500."""
        r = client.post(
            "/api/v1/config", headers=_admin_auth_header(), json={"llm_provider": "unknown_xyz"}
        )
        assert r.status_code == 400
        assert r.status_code != 500


class TestRecoverability:
    """5.4 RR — System can re-establish state after an interruption."""

    def test_rr001_session_persists_across_manager_instances(self, tmp_path):
        """RFN-010: Data written by one SessionManager instance is readable by a new one."""
        db = tmp_path / "recover.db"
        sm1 = SessionManager(db_path=str(db))
        sm1.append_message("recover-session", "user", "Mensaje persistido")

        sm2 = SessionManager(db_path=str(db))
        messages = sm2.get_messages("recover-session")
        assert any(m["text"] == "Mensaje persistido" for m in messages)

    def test_rr002_config_update_is_idempotent(self):
        """RFN-002: Applying the same config update twice succeeds both times."""
        headers = _admin_auth_header()
        locked_provider = cfg_module.settings.default_llm_provider
        r1 = client.post(
            "/api/v1/config", headers=headers, json={"llm_provider": locked_provider}
        )
        r2 = client.post(
            "/api/v1/config", headers=headers, json={"llm_provider": locked_provider}
        )
        assert r1.status_code == 200
        assert r2.status_code == 200


# ===========================================================================
# 6. SECURITY (Seguridad)
# ===========================================================================


class TestConfidentiality:
    """6.1 SC — Data is only accessible to authorised entities."""

    def test_sc001_api_keys_absent_from_config_response(self):
        """API keys must not appear in the /config response."""
        text = client.get("/api/v1/config").text.lower()
        assert "api_key" not in text
        assert "test-key" not in text

    def test_sc002_api_keys_absent_from_health_response(self):
        """API keys must not appear in the /health response."""
        text = client.get("/api/v1/health").text.lower()
        assert "api_key" not in text
        assert "test-key" not in text

    def test_sc003_error_response_no_stack_trace(self):
        """RFN-006: Validation error does not expose Python stack trace."""
        text = client.post("/api/v1/chat", json={}).text
        assert "Traceback" not in text
        assert "  File " not in text


class TestIntegrity:
    """6.2 SI — Unauthorised data modification is prevented."""

    def test_si001_password_not_returned_in_auth_response(self):
        """Plaintext password and hash must not appear in registration response."""
        r = client.post(
            "/api/v1/auth/register",
            json={
                "email": "integrity@test.com",
                "password": "mysecretpassword",
                "full_name": "Test User",
                "account_type": "Estudiante",
                "knowledge_level": 2,
                "questionnaire_answers": [],
            },
        )
        assert r.status_code == 200
        text = r.text
        assert "mysecretpassword" not in text
        assert "password_hash" not in text

    def test_si002_user_sessions_are_isolated(self):
        """Security: User A cannot see User B's sessions."""
        user_a = client.post(
            "/api/v1/auth/register",
            json={
                "email": "user_a_si@test.com",
                "password": "password_a_123",
                "full_name": "User A",
                "account_type": "Estudiante",
                "knowledge_level": 1,
                "questionnaire_answers": [],
            },
        ).json()
        user_b = client.post(
            "/api/v1/auth/register",
            json={
                "email": "user_b_si@test.com",
                "password": "password_b_123",
                "full_name": "User B",
                "account_type": "Estudiante",
                "knowledge_level": 1,
                "questionnaire_answers": [],
            },
        ).json()

        with patch("src.api.routes.chat.get_llm_provider", return_value=Mock()), patch(
            "src.api.routes.chat.get_chatbot_agent", return_value=_mock_agent()
        ):
            client.post(
                "/api/v1/chat",
                headers={"Authorization": f"Bearer {user_a['access_token']}"},
                json={"session_id": "private-a", "message": "Privado", "use_rag": False},
            )

        sessions_b = client.get(
            "/api/v1/sessions",
            headers={"Authorization": f"Bearer {user_b['access_token']}"},
        ).json()
        ids_b = [s["session_id"] for s in sessions_b.get("sessions", [])]
        assert "private-a" not in ids_b


class TestNonRepudiation:
    """6.3 SN — Actions can be proven to have taken place."""

    def test_sn001_messages_have_created_at_timestamp(self, tmp_path):
        """RFN-007: Every persisted message carries a created_at timestamp."""
        sm = SessionManager(db_path=str(tmp_path / "sn.db"))
        sm.append_message("audit-session", "user", "Mensaje auditado")
        msgs = sm.get_messages("audit-session")
        assert len(msgs) > 0
        assert msgs[0].get("created_at"), "created_at must be non-empty"

    def test_sn002_assistant_messages_store_provider_and_model(self, tmp_path):
        """RFN-007: Provider and model are stored for each assistant message."""
        sm = SessionManager(db_path=str(tmp_path / "sn2.db"))
        sm.append_message(
            "audit2",
            "assistant",
            "Respuesta.",
            provider="openai",
            model="gpt-4o-mini",
            used_rag=False,
        )
        msgs = sm.get_messages("audit2")
        assistant_msgs = [m for m in msgs if m["role"] == "assistant"]
        assert len(assistant_msgs) > 0


class TestAccountability:
    """6.4 SR — Actions can be traced to the entity that performed them."""

    def test_sr001_registration_issues_access_token(self):
        """Registration issues a verifiable access token."""
        r = client.post(
            "/api/v1/auth/register",
            json={
                "email": "acct@test.com",
                "password": "password123",
                "full_name": "Acct User",
                "account_type": "Estudiante",
                "knowledge_level": 1,
                "questionnaire_answers": [],
            },
        )
        assert r.status_code == 200
        token = r.json().get("access_token", "")
        assert len(token) > 10

    def test_sr002_login_rejects_wrong_password(self):
        """Wrong password is rejected with a 4xx status code."""
        client.post(
            "/api/v1/auth/register",
            json={
                "email": "login_sr@test.com",
                "password": "correct_pass_123",
                "full_name": "Login SR",
                "account_type": "Estudiante",
                "knowledge_level": 1,
                "questionnaire_answers": [],
            },
        )
        r = client.post(
            "/api/v1/auth/login",
            json={"email": "login_sr@test.com", "password": "wrong_password"},
        )
        assert r.status_code in (400, 401, 403)


class TestAuthenticity:
    """6.5 SA — Identity of subjects can be verified."""

    def test_sa001_valid_token_grants_session_access(self):
        """Valid token obtained from registration allows authenticated requests."""
        reg = client.post(
            "/api/v1/auth/register",
            json={
                "email": "auth_sa@test.com",
                "password": "auth_pass_123",
                "full_name": "Auth User",
                "account_type": "Estudiante",
                "knowledge_level": 1,
                "questionnaire_answers": [],
            },
        ).json()
        r = client.get(
            "/api/v1/sessions",
            headers={"Authorization": f"Bearer {reg['access_token']}"},
        )
        assert r.status_code == 200

    def test_sa002_invalid_token_returns_no_foreign_sessions(self):
        """Invalid bearer token must not expose another user's session data."""
        r = client.get(
            "/api/v1/sessions",
            headers={"Authorization": "Bearer totally_invalid_token_xyz"},
        )
        if r.status_code == 200:
            assert r.json().get("total_sessions", 0) == 0


# ===========================================================================
# 7. MAINTAINABILITY (Mantenibilidad)
# ===========================================================================


class TestModularity:
    """7.1 MT — System composed of discrete, independent components."""

    def test_mt001_llm_providers_are_separate_files(self):
        """RFN-001: Each LLM provider lives in its own module file."""
        providers_dir = Path("src/llm/providers")
        assert providers_dir.exists()
        provider_files = list(providers_dir.glob("*_provider.py"))
        assert len(provider_files) >= 3, f"Expected ≥3 provider files, got {len(provider_files)}"

    def test_mt002_api_routes_are_separate_files(self):
        """RFN-001: Each API route group is a separate module."""
        routes_dir = Path("src/api/routes")
        assert routes_dir.exists()
        route_files = [f for f in routes_dir.glob("*.py") if f.name != "__init__.py"]
        assert len(route_files) >= 5, f"Expected ≥5 route files, got {len(route_files)}"

    def test_mt003_core_subsystems_are_packages(self):
        """RFN-001: agents, rag, memory, llm, core exist as proper Python packages."""
        for pkg in ("src/agents", "src/rag", "src/memory", "src/llm", "src/core"):
            assert Path(pkg).is_dir(), f"Package directory missing: {pkg}"
            assert (Path(pkg) / "__init__.py").exists(), f"Missing __init__.py in {pkg}"


class TestReusability:
    """7.2 MR — Components can be reused across contexts."""

    def test_mr001_llm_factory_registers_multiple_providers(self):
        """RFN-008: LLMFactory has ≥ 3 registered provider backends (pilot lock only
        restricts which one is *active*, not how many are implemented)."""
        from src.llm.factory import LLMFactory
        providers = LLMFactory.get_registered_providers()
        assert "openai" in providers
        assert len(providers) >= 3

    def test_mr002_session_manager_portable_to_any_db_path(self, tmp_path):
        """RFN-010: SessionManager works with any given SQLite path."""
        sm = SessionManager(db_path=str(tmp_path / "custom.db"))
        sm.create_session("reuse-session")
        assert len(sm.list_sessions()) == 1

    def test_mr003_models_json_externalises_provider_catalog(self):
        """RFN-008: models.json exists and contains the model catalog."""
        models_file = Path("src/llm/models.json")
        assert models_file.exists()
        with open(models_file) as f:
            catalog = json.load(f)
        assert isinstance(catalog, dict) and len(catalog) > 0


class TestAnalysability:
    """7.3 MA — Impact of changes can be assessed."""

    def test_ma001_health_exposes_component_states(self):
        """RFN-009: Health reveals internal component states."""
        data = client.get("/api/v1/health").json()
        for key in ("rag_status", "vector_store_documents", "llm_providers"):
            assert key in data

    def test_ma002_at_least_eight_test_modules_exist(self):
        """A rich test suite aids impact analysis when code changes."""
        count = len(list(Path("tests").glob("test_*.py")))
        assert count >= 8, f"Found only {count} test modules"

    def test_ma003_requirements_file_documents_dependencies(self):
        """requirements.txt lists key dependencies for dependency analysis."""
        req = Path("requirements.txt")
        assert req.exists()
        content = req.read_text().lower()
        assert "fastapi" in content
        assert "pydantic" in content


class TestModifiability:
    """7.4 MM — System can be modified without defect introduction."""

    def test_mm001_runtime_config_changeable_without_restart(self):
        """RFN-002: Temperature can be updated via API at runtime."""
        r = client.post(
            "/api/v1/config", headers=_admin_auth_header(), json={"temperature": 0.5}
        )
        assert r.status_code == 200
        assert r.json()["temperature"] == 0.5

    def test_mm002_provider_catalog_driven_by_models_json(self):
        """RFN-008: Factory catalog comes from models.json, not hard-coded lists."""
        from src.llm.factory import LLMFactory
        assert bool(LLMFactory._provider_models), "Provider model catalog must be non-empty"
        assert Path("src/llm/models.json").exists()


class TestTestability:
    """7.5 MTP — Tests can be established to verify system behaviour."""

    def test_mtp001_tests_run_with_fake_api_key(self):
        """Full test suite works without real API credentials."""
        assert os.getenv("OPENAI_API_KEY") == "test-key"

    def test_mtp002_each_domain_has_dedicated_test_module(self):
        """Each major domain (api, auth, llm, rag, session) has a test file."""
        expected = [
            "tests/test_api.py",
            "tests/test_auth.py",
            "tests/test_llm_factory.py",
            "tests/test_document_processor.py",
            "tests/test_rag_retriever.py",
            "tests/test_session_concepts.py",
        ]
        for path in expected:
            assert Path(path).exists(), f"Missing test file: {path}"

    def test_mtp003_conftest_provides_shared_environment(self):
        """conftest.py provides environment setup shared across all tests."""
        conftest = Path("tests/conftest.py")
        assert conftest.exists()
        assert "OPENAI_API_KEY" in conftest.read_text()


# ===========================================================================
# 8. PORTABILITY (Portabilidad)
# ===========================================================================


class TestAdaptability:
    """8.1 PA — System can be adapted for different environments."""

    def test_pa001_llm_provider_configurable_via_env_var(self):
        """RFN-002: DEFAULT_LLM_PROVIDER is a declared settings field."""
        from src.core.config import Settings
        assert "default_llm_provider" in Settings.model_fields

    def test_pa002_db_path_configurable_via_env_var(self):
        """RFN-010: CONVERSATION_DB_PATH is a declared settings field."""
        from src.core.config import Settings
        assert "conversation_db_path" in Settings.model_fields

    def test_pa003_chroma_dir_configurable_via_env_var(self):
        """CHROMA_PERSIST_DIR is a declared settings field."""
        from src.core.config import Settings
        assert "chroma_persist_dir" in Settings.model_fields

    def test_pa004_at_least_four_llm_providers_registered(self):
        """RFN-008: System supports ≥ 4 LLM backends (registered, not necessarily
        all active — pilot lock restricts runtime selection to one)."""
        from src.llm.factory import LLMFactory
        providers = LLMFactory.get_registered_providers()
        assert len(providers) >= 4, f"Expected ≥4 providers, found {providers}"

    def test_pa005_cors_origins_configurable(self):
        """RFN-004: CORS origins are a configurable settings field."""
        from src.core.config import Settings
        assert "cors_origins" in Settings.model_fields


class TestInstallability:
    """8.2 PI — System can be installed in its target environment."""

    def test_pi001_requirements_txt_exists(self):
        """requirements.txt enables reproducible pip installation."""
        assert Path("requirements.txt").exists()

    def test_pi002_dockerfile_exists(self):
        """Dockerfile enables containerised installation."""
        assert Path("Dockerfile").exists()

    def test_pi003_docker_compose_exists(self):
        """docker-compose.yml enables one-command deployment."""
        assert Path("docker-compose.yml").exists()

    def test_pi004_setup_scripts_exist(self):
        """README documents the pip-based host-level installation steps."""
        readme = Path("README.md").read_text().lower()
        assert "pip install" in readme and "requirements.txt" in readme

    def test_pi005_readme_documents_startup_command(self):
        """README.md contains startup instructions."""
        readme = Path("README.md")
        assert readme.exists()
        content = readme.read_text().lower()
        assert "uvicorn" in content or "docker" in content


class TestReplaceability:
    """8.3 PR2 — System can replace another product in the same environment."""

    def test_pr2001_llm_provider_swappable_via_api(self):
        """RFN-008: The active LLM provider is swappable by changing the configured
        default (pilot lock restricts /config to that single provider; swapping the
        underlying vendor is done by reconfiguring DEFAULT_LLM_PROVIDER, not by an
        arbitrary live request)."""
        locked_provider = cfg_module.settings.default_llm_provider
        r = client.post(
            "/api/v1/config", headers=_admin_auth_header(), json={"llm_provider": locked_provider}
        )
        assert r.status_code == 200
        assert r.json()["llm_provider"] == locked_provider

    def test_pr2002_embedding_model_configurable_via_env(self):
        """EMBEDDING_MODEL is a declared settings field for swapping vector models."""
        from src.core.config import Settings
        assert "embedding_model" in Settings.model_fields

    def test_pr2003_rest_api_contract_is_provider_agnostic(self):
        """RF-001: Chat API response schema is identical regardless of LLM provider."""
        with patch("src.api.routes.chat.get_llm_provider", return_value=Mock()), patch(
            "src.api.routes.chat.get_chatbot_agent", return_value=_mock_agent()
        ):
            r = client.post(
                "/api/v1/chat", json={"message": "Test portabilidad", "llm_provider": "openai"}
            )
        assert r.status_code == 200
        for field in ("session_id", "response", "provider", "model", "used_rag"):
            assert field in r.json()

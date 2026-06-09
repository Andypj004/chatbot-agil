"""
Performance tests for the full real pipeline — ISO 25010 § 2.4.

Strategy: mock only the external LLM HTTP boundary (ChatbotAgent._invoke_llm).
Everything else runs real: question classification, prompt building,
concept tracking, conversation history assembly, source filtering,
and SQLite session persistence.

New test IDs: PT-005 through PT-015.
"""

import time
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

try:
    from langchain.schema import Document
except ImportError:
    from langchain_core.documents import Document

from src.agents.chatbot_agent import ChatbotAgent
from src.api import dependencies
from src.core.question_classifier import classify_question
from src.memory.concept_tracker import build_history_note, extract_concepts
from src.memory.session_manager import SessionManager
from src.main import app

client = TestClient(app)

_LLM_RESPONSE = (
    "El Sprint en Scrum es una iteración de trabajo de duración fija, "
    "generalmente entre 1 y 4 semanas, al final del cual se produce un "
    "Incremento potencialmente entregable del producto."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_provider():
    provider = Mock()
    provider.get_provider_name.return_value = "openai"
    provider.model_name = "gpt-4o-mini"
    llm = Mock()
    llm.invoke.return_value = Mock(content=_LLM_RESPONSE)
    provider.get_llm.return_value = llm
    return provider


def _make_real_agent(latency_ms: float = 0, with_rag: bool = False) -> ChatbotAgent:
    """Real ChatbotAgent whose only mock boundary is _invoke_llm.

    All internal stages run real:
      - question classification (regex)
      - concept extraction and history note building
      - prompt construction via PromptManager
      - source filtering (_filter_relevant_sources)
      - form command parsing

    Args:
        latency_ms: artificial delay injected into the LLM call to simulate
                    real provider response time.
        with_rag: if True, attach a mock retriever that returns one document.
    """
    agent = ChatbotAgent(llm_provider=_make_mock_provider(), rag_retriever=None)

    if with_rag:
        mock_retriever = Mock()
        mock_retriever.has_documents.return_value = True
        mock_retriever.retrieve_documents.return_value = [
            Document(
                page_content="El Sprint es una iteración de trabajo fija de 1 a 4 semanas en Scrum.",
                metadata={
                    "filename": "scrum_guide.pdf",
                    "scope": "global_rag",
                    "file_hash": "abc123hash",
                    "source": "scrum_guide.pdf",
                },
            )
        ]
        mock_retriever._build_context.return_value = (
            "El Sprint es una iteración de trabajo fija de 1 a 4 semanas en Scrum."
        )
        agent.rag_retriever = mock_retriever

    def _invoke(prompt: str) -> str:
        if latency_ms > 0:
            time.sleep(latency_ms / 1000)
        return _LLM_RESPONSE

    agent._invoke_llm = _invoke
    return agent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def isolate_db(tmp_path, monkeypatch):
    """Isolated SQLite database per test for the HTTP client path."""
    db = tmp_path / "perf_http.db"
    monkeypatch.setattr(
        dependencies, "_session_manager", SessionManager(db_path=str(db))
    )
    app.dependency_overrides.clear()
    yield
    app.dependency_overrides.clear()
    monkeypatch.setattr(dependencies, "_session_manager", None)


# ===========================================================================
# Stage-level latency  (PT-005 – PT-008)
# ===========================================================================


class TestPipelineStageLatency:
    """Each pipeline stage tested in isolation to identify bottlenecks."""

    def test_pt005_question_classifier_100_queries_under_50ms(self):
        """PT-005: 100 classify_question() calls < 50 ms total (< 0.5 ms each).

        The classifier is on the critical path of every chat request. It uses
        compiled regex — 100 evaluations must complete well under 50 ms.
        """
        queries = [
            "¿Qué es el Sprint en Scrum?",
            "¿Cómo funciona Kanban?",
            "En mi proyecto queremos organizar el backlog, ¿cómo podemos hacerlo?",
            "¿Cuáles son los roles de Scrum?",
            "Como podemos mejorar nuestras dailies en el equipo",
        ] * 20
        start = time.monotonic()
        for q in queries:
            classify_question(q)
        elapsed = (time.monotonic() - start) * 1000
        assert elapsed < 50, (
            f"100 classify_question() calls took {elapsed:.1f} ms (limit 50 ms). "
            "Regex evaluation is too slow for production use."
        )

    def test_pt006_concept_extractor_100_queries_under_50ms(self):
        """PT-006: 100 extract_concepts() calls < 50 ms total.

        Concept extraction uses simple substring search. 100 calls on agile
        messages must stay under 50 ms.
        """
        queries = [
            "¿Cómo funciona el sprint planning?",
            "El daily scrum es clave en la metodología ágil",
            "El product owner revisa el backlog con el equipo",
            "La retrospectiva de sprint mejora el proceso kanban",
        ] * 25
        start = time.monotonic()
        for q in queries:
            extract_concepts(q)
        elapsed = (time.monotonic() - start) * 1000
        assert elapsed < 50, (
            f"100 extract_concepts() calls took {elapsed:.1f} ms (limit 50 ms)."
        )

    def test_pt007_history_note_builder_under_5ms(self):
        """PT-007: build_history_note() for a 5-concept list < 5 ms.

        This function runs on every non-trivial chat request. It is pure string
        formatting and must be negligible.
        """
        concepts = ["sprint", "scrum", "kanban", "backlog", "planning"]
        start = time.monotonic()
        note = build_history_note(concepts, repeated=True)
        elapsed = (time.monotonic() - start) * 1000
        assert elapsed < 5, (
            f"build_history_note() took {elapsed:.2f} ms (limit 5 ms)."
        )
        assert note  # sanity: non-empty result

    def test_pt008_source_filter_10_chunks_under_100ms(self):
        """PT-008: _filter_relevant_sources() with 10 document chunks < 100 ms.

        The source filter applies token-overlap and phrase-search scoring to
        every retrieved chunk. 10 chunks is a typical upper bound for a
        RAG-enabled response.
        """
        response = "El Sprint es una iteración de 2 semanas en Scrum con un objetivo claro."
        sources = [
            {
                "content": (
                    f"El Sprint en Scrum es una iteración de duración fija donde el equipo"
                    f" trabaja para alcanzar el Sprint Goal. Chunk número {i}."
                ),
                "metadata": {
                    "filename": f"doc_{i}.pdf",
                    "scope": "global_rag",
                    "file_hash": f"hash_{i:04d}",
                    "source": f"doc_{i}.pdf",
                    "page": i + 1,
                },
            }
            for i in range(10)
        ]
        start = time.monotonic()
        filtered = ChatbotAgent._filter_relevant_sources(response, sources)
        elapsed = (time.monotonic() - start) * 1000
        assert elapsed < 100, (
            f"_filter_relevant_sources() with 10 chunks took {elapsed:.1f} ms (limit 100 ms)."
        )
        assert isinstance(filtered, list)


# ===========================================================================
# Agent pipeline overhead  (PT-009 – PT-012)
# ===========================================================================


class TestAgentPipelineOverhead:
    """ChatbotAgent.chat() with instant LLM — measures non-LLM pipeline overhead.

    These tests call the agent directly (not through HTTP) to isolate the
    pipeline cost from network/ASGI overhead. The only mock is _invoke_llm.
    """

    def test_pt009_direct_response_overhead_under_300ms(self, tmp_path):
        """PT-009: Full direct-response pipeline (classify + prompt + session + filter) < 300 ms.

        Covers the happy path: a definition question (¿Qué es?) that the
        classifier routes to direct mode with no RAG.
        """
        sm = SessionManager(db_path=str(tmp_path / "agent_direct.db"))
        agent = _make_real_agent(latency_ms=0)

        start = time.monotonic()
        result = agent.chat(
            message="¿Qué es el Sprint en Scrum?",
            use_rag=False,
            session_id="perf-direct-001",
            session_manager=sm,
        )
        elapsed = (time.monotonic() - start) * 1000

        assert result["response"] == _LLM_RESPONSE
        assert elapsed < 300, (
            f"Direct-response pipeline overhead: {elapsed:.0f} ms (limit 300 ms). "
            "Non-LLM pipeline stages are too slow."
        )

    def test_pt010_socratic_response_overhead_under_300ms(self, tmp_path):
        """PT-010: Socratic-mode pipeline (project-context classifier + prompt build) < 300 ms.

        Uses a query that contains project-context signals so the classifier
        routes to socratic mode, exercising a different prompt path.
        """
        sm = SessionManager(db_path=str(tmp_path / "agent_socratic.db"))
        agent = _make_real_agent(latency_ms=0)

        start = time.monotonic()
        result = agent.chat(
            message="¿Cómo podemos organizar el backlog de nuestro proyecto para la entrega?",
            use_rag=False,
            session_id="perf-socratic-001",
            session_manager=sm,
        )
        elapsed = (time.monotonic() - start) * 1000

        assert result["response"] == _LLM_RESPONSE
        assert elapsed < 300, (
            f"Socratic pipeline overhead: {elapsed:.0f} ms (limit 300 ms)."
        )

    def test_pt011_pipeline_with_12_message_history_under_400ms(self, tmp_path):
        """PT-011: Pipeline with full 12-message history window < 400 ms.

        Pre-loads 12 messages (the max context window) into the session and
        passes them as conversation_messages to verify that history assembly
        and prompt building scale acceptably.
        """
        sm = SessionManager(db_path=str(tmp_path / "agent_history.db"))
        session_id = "perf-history-001"
        sm.create_session(session_id)
        for i in range(12):
            role = "user" if i % 2 == 0 else "assistant"
            sm.append_message(
                session_id,
                role,
                f"Mensaje de prueba {i} sobre Scrum, Sprint y Kanban para el backlog del proyecto.",
            )

        history = sm.get_messages(session_id, limit=12)
        assert len(history) == 12

        agent = _make_real_agent(latency_ms=0)

        start = time.monotonic()
        agent.chat(
            message="¿Qué es el Sprint Goal?",
            use_rag=False,
            conversation_messages=history,
            session_id=session_id,
            session_manager=sm,
        )
        elapsed = (time.monotonic() - start) * 1000

        assert elapsed < 400, (
            f"Pipeline with 12-message history: {elapsed:.0f} ms (limit 400 ms). "
            "Conversation context assembly is a bottleneck."
        )

    def test_pt012_rag_pipeline_overhead_under_400ms(self, tmp_path):
        """PT-012: Full pipeline with RAG retrieval (mocked vector store, instant LLM) < 400 ms.

        Exercises the RAG code path: retrieve_documents → _build_context →
        prompt injection → _filter_relevant_sources → source extraction.
        The vector store call itself is mocked; all orchestration code runs real.
        """
        sm = SessionManager(db_path=str(tmp_path / "agent_rag.db"))
        agent = _make_real_agent(latency_ms=0, with_rag=True)

        start = time.monotonic()
        result = agent.chat(
            message="¿Qué es el Sprint?",
            use_rag=True,
            session_id="perf-rag-001",
            session_manager=sm,
        )
        elapsed = (time.monotonic() - start) * 1000

        assert result["used_rag"] is True
        assert elapsed < 400, (
            f"RAG pipeline overhead (no vector IO): {elapsed:.0f} ms (limit 400 ms)."
        )


# ===========================================================================
# End-to-end with simulated LLM latency  (PT-013 – PT-015)
# ===========================================================================


class TestEndToEndSimulatedLLMLatency:
    """Full HTTP pipeline with controlled LLM response time.

    These tests inject the real agent (only _invoke_llm mocked) into the
    FastAPI route and measure the wall-clock time of the full HTTP cycle:
    ASGI dispatch → auth → session setup → agent.chat() → session save → JSON serialize.

    They answer: given a known LLM latency, is the pipeline overhead acceptable?
    """

    def test_pt013_full_http_flow_500ms_llm_under_1000ms(self):
        """PT-013: HTTP flow with 500 ms LLM simulation completes in < 1 000 ms.

        The pipeline overhead budget is therefore < 500 ms. This validates
        that session lookups, prompt building, and serialization stay cheap
        even when added to a half-second LLM call.
        """
        agent = _make_real_agent(latency_ms=500)
        start = time.monotonic()
        with patch("src.api.routes.chat.get_chatbot_agent", return_value=agent):
            r = client.post(
                "/api/v1/chat",
                json={"message": "¿Qué es el Sprint en Scrum?", "use_rag": False},
            )
        elapsed = (time.monotonic() - start) * 1000

        assert r.status_code == 200
        assert elapsed < 1000, (
            f"Full HTTP flow (500 ms LLM): {elapsed:.0f} ms total (limit 1 000 ms). "
            "Pipeline overhead exceeds 500 ms budget."
        )

    def test_pt014_full_http_flow_1000ms_llm_under_1500ms(self):
        """PT-014: HTTP flow with 1 000 ms LLM simulation completes in < 1 500 ms.

        Pipeline overhead budget remains < 500 ms. Verifies the budget holds
        at higher LLM latencies (e.g. large models or cloud provider slowness).
        """
        agent = _make_real_agent(latency_ms=1000)
        start = time.monotonic()
        with patch("src.api.routes.chat.get_chatbot_agent", return_value=agent):
            r = client.post(
                "/api/v1/chat",
                json={"message": "¿Cómo funciona Kanban?", "use_rag": False},
            )
        elapsed = (time.monotonic() - start) * 1000

        assert r.status_code == 200
        assert elapsed < 1500, (
            f"Full HTTP flow (1 000 ms LLM): {elapsed:.0f} ms total (limit 1 500 ms)."
        )

    def test_pt015_five_consecutive_requests_each_under_300ms(self):
        """PT-015: 5 consecutive requests to same session each complete in < 300 ms (instant LLM).

        Validates that SQLite history growth across a conversation (up to 5
        accumulated message pairs) does not cause noticeable latency
        degradation. Each individual request must stay under 300 ms.
        """
        agent = _make_real_agent(latency_ms=0)
        session_id = "perf-consecutive-001"

        for i in range(5):
            start = time.monotonic()
            with patch("src.api.routes.chat.get_chatbot_agent", return_value=agent):
                r = client.post(
                    "/api/v1/chat",
                    json={
                        "message": f"Consulta {i + 1}: ¿Qué es el Sprint Goal en Scrum?",
                        "session_id": session_id,
                        "use_rag": False,
                    },
                )
            elapsed = (time.monotonic() - start) * 1000

            assert r.status_code == 200, f"Request {i + 1}/5 failed: {r.status_code}"
            assert elapsed < 300, (
                f"Request {i + 1}/5 took {elapsed:.0f} ms (limit 300 ms). "
                "Session history accumulation is degrading performance."
            )

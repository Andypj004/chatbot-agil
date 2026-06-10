"""Tests for lightweight session concept tracking."""

from src.memory.session_manager import SessionManager


def test_record_and_retrieve_session_concepts(tmp_path):
    db_path = tmp_path / "concepts.db"
    manager = SessionManager(db_path=str(db_path))

    session_id = "session-concepts"
    concepts = manager.record_concepts(
        session_id,
        "¿Qué es Scrum y cómo funciona un sprint?",
        "Scrum ayuda a organizar el trabajo del sprint.",
    )

    assert "scrum" in concepts
    assert "sprint" in concepts
    assert manager.has_seen_concept(session_id, "scrum") is True

    records = manager.get_session_concepts(session_id)
    assert records
    assert records[0]["mention_count"] >= 1


def test_record_and_retrieve_session_citations(tmp_path):
    db_path = tmp_path / "citations.db"
    manager = SessionManager(db_path=str(db_path))

    session_id = "session-citations"
    citations = [
        {
            "document_id": "doc-1",
            "filename": "scrum-guide.pdf",
            "source": "data/uploads/global/scrum-guide.pdf",
            "page": 12,
            "section": "Roles de Scrum",
            "scope": "global_rag",
            "excerpt": "Scrum define roles del equipo.",
        },
        {
            "document_id": "doc-1",
            "filename": "scrum-guide.pdf",
            "source": "data/uploads/global/scrum-guide.pdf",
            "page": 12,
            "section": "Roles de Scrum",
            "scope": "global_rag",
            "excerpt": "Scrum define roles del equipo.",
        },
        {
            "document_id": "doc-2",
            "filename": "kanban-guide.pdf",
            "source": "data/uploads/global/kanban-guide.pdf",
            "page": 37,
            "section": "Métricas",
            "scope": "global_rag",
            "excerpt": "La previsión con exactitud es un problema difícil.",
        },
    ]

    recorded = manager.record_source_citations(session_id, citations)

    assert len(recorded) == 2
    recent_keys = manager.get_recent_source_citation_keys(session_id)
    assert set(recent_keys) == set(recorded)

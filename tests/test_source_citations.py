"""Tests for source citation normalization."""

from unittest.mock import Mock

from src.agents.chatbot_agent import ChatbotAgent


def test_extract_sources_includes_page_and_section():
    sources = ChatbotAgent._extract_sources(
        [
            {
                "content": "Roles y responsabilidades de Scrum.",
                "metadata": {
                    "file_hash": "abc123",
                    "filename": "scrum-guide.pdf",
                    "source": "/docs/scrum-guide.pdf",
                    "page": 12,
                    "heading": "Roles de Scrum",
                    "score": 0.91,
                },
            }
        ]
    )

    assert sources[0]["page"] == 12
    assert sources[0]["section"] == "Roles de Scrum"
    assert sources[0]["relevance"] == 0.91


def test_filter_relevant_sources_keeps_matching_sources_only():
    response = "Los criterios de aceptacion deben ser claros, medibles y verificables."
    raw_sources = [
        {
            "content": "Los criterios de aceptacion deben ser claros, medibles y verificables.",
            "metadata": {"filename": "scrum-guide.pdf", "section": "Acceptance Criteria"},
        },
        {
            "content": "Historia general sobre marcos de trabajo sin relacion directa.",
            "metadata": {"filename": "other.pdf", "section": "Contexto general"},
        },
    ]

    filtered = ChatbotAgent._filter_relevant_sources(response, raw_sources)

    assert len(filtered) == 1
    assert filtered[0]["metadata"]["filename"] == "scrum-guide.pdf"
    assert filtered[0]["metadata"]["relevance"] >= 0.25


def test_filter_relevant_sources_prefers_unseen_citations():
    response = "Scrum define roles y el Product Owner prioriza el backlog."
    seen_source = {
        "content": "Scrum define roles del equipo.",
        "metadata": {
            "filename": "scrum-guide.pdf",
            "source": "data/uploads/global/scrum-guide.pdf",
            "page": 12,
            "section": "Roles de Scrum",
        },
    }
    fresh_source = {
        "content": "El Product Owner prioriza el backlog.",
        "metadata": {
            "filename": "scrum-guide.pdf",
            "source": "data/uploads/global/scrum-guide.pdf",
            "page": 18,
            "section": "Product Owner",
        },
    }

    filtered = ChatbotAgent._filter_relevant_sources(
        response,
        [seen_source, fresh_source],
        seen_citation_keys={ChatbotAgent._source_citation_key(seen_source)},
    )

    assert len(filtered) == 2
    assert ChatbotAgent._source_citation_key(filtered[0]) == ChatbotAgent._source_citation_key(fresh_source)
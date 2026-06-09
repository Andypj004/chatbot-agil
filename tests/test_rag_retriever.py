"""Unit tests for RAG retriever dual-scope behavior."""

from unittest.mock import Mock
from langchain.schema import Document

from src.rag.retriever import RAGRetriever


class _DummyLLM:
    def invoke(self, _prompt):
        return "ok"


class _DummyProvider:
    model_name = "dummy-model"

    def get_llm(self):
        return _DummyLLM()

    def get_provider_name(self):
        return "dummy"


def _pair_with_scores(docs):
    return [(doc, 0.9 - idx * 0.01) for idx, doc in enumerate(docs)]


def test_retrieve_documents_combines_global_and_session():
    """Retriever should always include global scope and add session scope when session_id exists."""
    vector_store = Mock()

    global_docs = [
        Document(page_content="global a", metadata={"scope": "global_rag", "file_hash": "g1"}),
        Document(page_content="global b", metadata={"scope": "global_rag", "file_hash": "g2"}),
        Document(page_content="global c", metadata={"scope": "global_rag", "file_hash": "g3"}),
    ]
    session_docs = [
        Document(
            page_content="session a",
            metadata={"scope": "session_chat", "session_id": "s1", "file_hash": "s1a"},
        ),
        Document(
            page_content="session b",
            metadata={"scope": "session_chat", "session_id": "s1", "file_hash": "s1b"},
        ),
    ]

    def _similarity_search_with_score(query, k=5, filter=None):
        _ = query
        _ = k
        if filter == {"scope": "global_rag"}:
            return _pair_with_scores(global_docs)
        if filter == {"session_id": "s1"}:
            return _pair_with_scores(session_docs)
        return []

    vector_store.similarity_search_with_score.side_effect = _similarity_search_with_score

    retriever = RAGRetriever(vector_store=vector_store, llm_provider=_DummyProvider(), top_k=5)
    docs = retriever.retrieve_documents(query="pregunta", session_id="s1")

    assert len(docs) == 5
    assert any(item.metadata.get("scope") == "global_rag" for item in docs)
    assert any(item.metadata.get("scope") == "session_chat" for item in docs)


def test_retrieve_documents_without_session_returns_only_global():
    """Retriever should return only global docs when no session_id is provided."""
    vector_store = Mock()
    global_docs = [
        Document(page_content="global only", metadata={"scope": "global_rag", "file_hash": "g-only"}),
    ]

    def _similarity_search_with_score(query, k=5, filter=None):
        _ = query
        _ = k
        if filter == {"scope": "global_rag"}:
            return _pair_with_scores(global_docs)
        return []

    vector_store.similarity_search_with_score.side_effect = _similarity_search_with_score

    retriever = RAGRetriever(vector_store=vector_store, llm_provider=_DummyProvider(), top_k=3)
    docs = retriever.retrieve_documents(query="pregunta")

    assert len(docs) == 1
    assert all(item.metadata.get("scope") == "global_rag" for item in docs)


def test_query_overfetches_sources_when_return_sources_enabled():
    vector_store = Mock()
    global_docs = [
        Document(page_content=f"global {idx}", metadata={"scope": "global_rag", "file_hash": f"g{idx}"})
        for idx in range(1, 10)
    ]

    def _similarity_search_with_score(query, k=5, filter=None):
        _ = query
        _ = k
        if filter == {"scope": "global_rag"}:
            return _pair_with_scores(global_docs)
        return []

    vector_store.similarity_search_with_score.side_effect = _similarity_search_with_score

    retriever = RAGRetriever(vector_store=vector_store, llm_provider=_DummyProvider(), top_k=3)
    result = retriever.query("pregunta", return_sources=True)

    assert result["num_sources"] == 3
    assert len(result["sources"]) >= 3
    assert vector_store.similarity_search_with_score.call_count >= 2

"""RAG retriever for combining vector search with LLM"""

from typing import Any, Dict, List, Optional, TYPE_CHECKING
import re
import unicodedata

from src.rag.vector_store import VectorStore
from src.llm.base import BaseLLMProvider
from src.core.config import settings
from src.core.logger import get_logger

if TYPE_CHECKING:
    from langchain_core.prompts import PromptTemplate
    from langchain_core.documents import Document
else:
    PromptTemplate = Any
    Document = Any

logger = get_logger()


class RAGRetriever:
    """Retrieval-Augmented Generation system"""

    DEFAULT_PROMPT_TEMPLATE = """Eres un experto en metodologías ágiles. Responde siempre en español, de forma clara, directa y educativa.
Usa el siguiente conocimiento para fundamentar tu respuesta. Habla con autoridad propia sin mencionar ni insinuar que tienes un "contexto" o "documentos".

Conocimiento:
{context}

Pregunta: {question}

Respuesta:"""

    def __init__(
        self,
        vector_store: VectorStore,
        llm_provider: BaseLLMProvider,
        prompt_template: Optional[str] = None,
        top_k: Optional[int] = None,
    ):
        """Initialize RAG retriever

        Args:
            vector_store: VectorStore instance
            llm_provider: LLM provider instance
            prompt_template: Custom prompt template
            top_k: Number of documents to retrieve
        """
        self.vector_store = vector_store
        self.llm_provider = llm_provider
        self.top_k = top_k or settings.top_k_results

        from langchain_core.prompts import PromptTemplate as LangChainPromptTemplate

        # Set up prompt
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT_TEMPLATE
        self.prompt = LangChainPromptTemplate(
            template=self.prompt_template, input_variables=["context", "question"]
        )

        logger.info(f"RAG retriever initialized with top_k={self.top_k}")

    def has_documents(self) -> bool:
        """Return whether the backing vector store has indexed documents."""
        return self.vector_store.get_collection_count() > 0

    def _build_context(self, documents: List[Document]) -> str:
        """Build a bounded context string from retrieved documents."""
        max_chars = settings.rag_context_max_chars
        context_parts: List[str] = []
        current_chars = 0

        for document in documents:
            content = document.page_content.strip()
            if not content:
                continue

            remaining = max_chars - current_chars
            if remaining <= 0:
                break

            snippet = content[:remaining]
            context_parts.append(snippet)
            current_chars += len(snippet) + 2

        return "\n\n".join(context_parts)

    def retrieve_documents(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> List[Document]:
        """Retrieve relevant documents for a query

        Args:
            query: Query string
            k: Number of documents to retrieve (overrides default)
            filter: Optional metadata filter

        Returns:
            List of relevant documents
        """
        k = k or self.top_k
        logger.info(f"Retrieving documents for query: '{query}'")

        if filter is not None:
            return self.vector_store.similarity_search(
                query=query,
                k=k,
                filter=filter,
            )

        return self._retrieve_combined_documents(
            query=query, k=k, session_id=session_id
        )

    @staticmethod
    def _normalize_scope(metadata: Dict[str, Any]) -> str:
        scope = metadata.get("scope")
        if scope:
            return str(scope)
        return "global_rag"

    @staticmethod
    def _doc_key(doc: Document) -> str:
        metadata = doc.metadata or {}
        return "|".join(
            [
                str(metadata.get("file_hash") or ""),
                str(metadata.get("chunk_id") or ""),
                str(metadata.get("source") or ""),
                str(metadata.get("filename") or ""),
            ]
        )

    @staticmethod
    def _normalize_for_ranking(text: str) -> str:
        normalized = unicodedata.normalize("NFKD", text.lower().strip())
        normalized = re.sub(r"[\u0300-\u036f]", "", normalized)
        normalized = re.sub(r"[^a-z0-9\s]+", " ", normalized)
        return " ".join(normalized.split())

    @classmethod
    def _ranking_keywords(cls, query: str) -> List[str]:
        normalized = cls._normalize_for_ranking(query)
        keywords = {
            token for token in re.findall(r"\b\w+\b", normalized) if len(token) > 2
        }
        return sorted(keywords, key=len, reverse=True)

    @classmethod
    def _query_alignment_score(cls, query: str, doc: Document) -> float:
        metadata = doc.metadata or {}
        content = cls._normalize_for_ranking(doc.page_content or "")
        filename = cls._normalize_for_ranking(str(metadata.get("filename") or ""))
        source = cls._normalize_for_ranking(str(metadata.get("source") or ""))
        haystack = f"{filename} {source} {content}"
        query_norm = cls._normalize_for_ranking(query)
        if not query_norm or not haystack.strip():
            return 0.0

        query_tokens = [
            token for token in re.findall(r"\b\w+\b", query_norm) if len(token) > 2
        ]
        if not query_tokens:
            return 0.0

        unique_tokens = list(dict.fromkeys(query_tokens))
        overlap = sum(1 for token in unique_tokens if token in haystack)
        score = overlap / max(len(unique_tokens), 1)

        if query_norm in haystack:
            score += 1.5

        filename_hits = sum(1 for token in unique_tokens if token in filename)
        source_hits = sum(1 for token in unique_tokens if token in source)
        score += min(filename_hits * 0.2, 0.8)
        score += min(source_hits * 0.1, 0.4)

        if "title" in metadata and metadata.get("title"):
            title_text = cls._normalize_for_ranking(str(metadata.get("title") or ""))
            title_hits = sum(1 for token in unique_tokens if token in title_text)
            score += min(title_hits * 0.25, 1.0)

        if len(unique_tokens) >= 4:
            score += min(len(unique_tokens) / 20.0, 0.5)

        return score

    def _retrieve_combined_documents(
        self, query: str, k: int, session_id: Optional[str]
    ) -> List[Document]:
        """Retrieve global docs always, plus session docs when session_id is available."""
        global_quota = max(1, int(round(k * 0.6)))
        session_quota = max(0, k - global_quota)

        logger.debug(
            f"_retrieve_combined_documents: k={k}, global_quota={global_quota}, session_quota={session_quota}"
        )

        overfetch_global = max(k * 4, global_quota * 4, 12)
        global_results = self.vector_store.similarity_search_with_score(
            query=query,
            k=overfetch_global,
            filter={"scope": "global_rag"},
        )

        session_docs: List[Document] = []
        if session_id and session_quota > 0:
            overfetch_session = max(k * 2, session_quota * 4, 8)
            candidates = self.vector_store.similarity_search_with_score(
                query=query,
                k=overfetch_session,
                filter={"session_id": session_id},
            )
            for doc, _score in candidates:
                metadata = doc.metadata or {}
                if self._normalize_scope(metadata) != "session_chat":
                    continue
                session_docs.append(doc)

        max_per_file = getattr(settings, "rag_max_chunks_per_file", 3)
        candidate_entries: dict[str, tuple[Document, float, str]] = {}

        def _ingest_results(
            results: List[tuple[Document, float]], source_name: str
        ) -> None:
            for rank, (doc, _score) in enumerate(results):
                metadata = doc.metadata or {}
                if (
                    source_name == "session"
                    and self._normalize_scope(metadata) != "session_chat"
                ):
                    continue

                key = self._doc_key(doc)
                if not key:
                    continue

                file_key = str(
                    metadata.get("file_hash") or metadata.get("filename") or "unknown"
                )
                base_score = 1.0 / (rank + 1)
                alignment_score = self._query_alignment_score(query, doc)
                total_score = base_score + alignment_score

                existing = candidate_entries.get(key)
                if existing is None or total_score > existing[1]:
                    candidate_entries[key] = (doc, total_score, file_key)

        _ingest_results(global_results, "global")
        if session_docs:
            for doc in session_docs:
                key = self._doc_key(doc)
                metadata = doc.metadata or {}
                if not key:
                    continue
                file_key = str(
                    metadata.get("file_hash") or metadata.get("filename") or "unknown"
                )
                alignment_score = self._query_alignment_score(query, doc)
                existing = candidate_entries.get(key)
                total_score = alignment_score + 0.5
                if existing is None or total_score > existing[1]:
                    candidate_entries[key] = (doc, total_score, file_key)

        if not candidate_entries:
            return []

        grouped_by_file: dict[str, List[tuple[float, str, Document]]] = {}
        for key, (doc, score, file_key) in candidate_entries.items():
            grouped_by_file.setdefault(file_key, []).append((score, key, doc))

        for entries in grouped_by_file.values():
            entries.sort(key=lambda item: item[0], reverse=True)

        file_order = sorted(
            grouped_by_file.items(),
            key=lambda item: item[1][0][0] if item[1] else 0.0,
            reverse=True,
        )

        merged: List[Document] = []
        seen: set[str] = set()
        per_file_counts: dict[str, int] = {}

        while len(merged) < k:
            progressed = False
            for file_key, entries in file_order:
                if not entries:
                    continue

                count = per_file_counts.get(file_key, 0)
                if count >= max_per_file:
                    continue

                score, key, doc = entries.pop(0)
                if key in seen:
                    continue

                seen.add(key)
                merged.append(doc)
                per_file_counts[file_key] = count + 1
                progressed = True

                if len(merged) >= k:
                    break

            if not progressed:
                break

        logger.debug(
            f"_retrieve_combined_documents: returned {len(merged)} docs from {len(per_file_counts)} files"
        )
        return merged

    def retrieve_with_scores(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[tuple[Document, float]]:
        """Retrieve relevant documents with relevance scores

        Args:
            query: Query string
            k: Number of documents to retrieve
            filter: Optional metadata filter

        Returns:
            List of tuples (document, score)
        """
        k = k or self.top_k
        logger.info(f"Retrieving documents with scores for query: '{query}'")

        results = self.vector_store.similarity_search_with_score(
            query=query, k=k, filter=filter
        )

        return results

    def query(
        self,
        question: str,
        k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        return_sources: bool = False,
    ) -> Dict[str, Any]:
        """Query the RAG system

        Args:
            question: Question to answer
            k: Number of documents to retrieve
            filter: Optional metadata filter
            return_sources: Whether to return source documents

        Returns:
            Dictionary with answer and optional sources
        """
        logger.info(f"Processing RAG query: '{question}'")
        effective_k = k or self.top_k

        # Retrieve relevant documents
        documents = self.retrieve_documents(
            query=question,
            k=effective_k,
            filter=filter,
            session_id=session_id,
        )

        if not documents:
            logger.warning("No relevant documents found")
            return {
                "answer": "No encontré información relevante en los documentos para responder esa pregunta.",
                "sources": [] if return_sources else None,
            }

        # Format context from documents
        context = self._build_context(documents)
        if not context:
            logger.warning("Retrieved documents had no usable context")
            return {
                "answer": "No encontré información relevante en los documentos para responder esa pregunta.",
                "sources": [] if return_sources else None,
            }

        # Generate answer using LLM
        llm = self.llm_provider.get_llm()
        formatted_prompt = self.prompt.format(context=context, question=question)

        logger.info(f"Generating answer using {self.llm_provider.get_provider_name()}")
        result = llm.invoke(formatted_prompt)
        # BaseChatModel.invoke() returns AIMessage; BaseLLM.invoke() returns str.
        response = result.content if hasattr(result, "content") else result

        result = {"answer": response.strip(), "num_sources": len(documents)}

        if return_sources:
            source_k = max(effective_k, effective_k * 4)
            source_documents = self.retrieve_documents(
                query=question,
                k=source_k,
                filter=filter,
                session_id=session_id,
            )
            result["sources"] = [
                {"content": doc.page_content, "metadata": doc.metadata}
                for doc in source_documents
            ]

        logger.info("RAG query completed successfully")
        return result

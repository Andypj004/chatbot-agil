"""Main chatbot agent with a direct fast path for chat and RAG."""

from typing import Optional, Dict, Any, List, Iterator
import base64
import json
import re
from pathlib import Path
import unicodedata

import httpx
from langchain_core.messages import HumanMessage

from src.core import PromptManager, classify_question
from src.llm.base import BaseLLMProvider
from src.rag.retriever import RAGRetriever
from src.core.config import settings
from src.core.logger import get_logger
from src.memory.concept_tracker import build_history_note, extract_concepts
from src.llm.providers.ollama_provider import _candidate_base_urls, _is_ollama_reachable

logger = get_logger()

_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
_IMAGE_MIME = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
}


class ChatbotAgent:
    """Chatbot agent optimized for direct LLM and RAG flows."""

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        rag_retriever: Optional[RAGRetriever] = None,
        enable_memory: bool = False,
    ):
        """Initialize chatbot agent.

        Args:
            llm_provider: LLM provider instance
            rag_retriever: Optional RAG retriever for knowledge base
            enable_memory: Reserved for future session-aware memory support
        """
        self.llm_provider = llm_provider
        self.rag_retriever = rag_retriever
        self.enable_memory = enable_memory
        self.prompt_manager = PromptManager()

        logger.info(
            "Chatbot agent initialized with fast path, "
            f"rag={'enabled' if rag_retriever else 'disabled'}, "
            f"memory={'disabled' if not enable_memory else 'not_persisted'}"
        )

    def _build_conversation_block(
        self, conversation_messages: Optional[List[Dict[str, Any]]]
    ) -> str:
        """Format previous messages as a compact context block."""
        if not conversation_messages:
            return ""

        lines: List[str] = []
        for item in conversation_messages:
            role = item.get("role", "user")
            text = item.get("text") or item.get("content") or ""
            if not text:
                continue
            label = "Usuario" if role == "user" else "Asistente"
            lines.append(f"{label}: {text}")

        if not lines:
            return ""

        return "\n".join(lines[-12:])

    @staticmethod
    def _expand_rag_query(message: str) -> str:
        """Return the original message unchanged.

        Query expansion was removed because it injected terms the user did not ask for
        and made the response less faithful to the corpus evidence.
        """
        return message

    def _invoke_llm(self, prompt: str) -> str:
        llm = self.llm_provider.get_llm()
        result = llm.invoke(prompt)
        return str(result.text) if hasattr(result, "text") else result

    def _build_direct_prompt(
        self,
        message: str,
        conversation_messages: Optional[List[Dict[str, Any]]] = None,
        rag_hint: Optional[str] = None,
        history_note: Optional[str] = None,
        user_profile_note: Optional[str] = None,
    ) -> str:
        conversation_block = self._build_conversation_block(conversation_messages)
        return self.prompt_manager.build_direct_prompt(
            message=message,
            conversation_block=conversation_block or None,
            rag_hint=rag_hint,
            history_note=history_note,
            user_profile_note=user_profile_note,
        )

    def _build_socratic_prompt(
        self,
        message: str,
        conversation_messages: Optional[List[Dict[str, Any]]] = None,
        rag_hint: Optional[str] = None,
        history_note: Optional[str] = None,
        user_profile_note: Optional[str] = None,
    ) -> str:
        conversation_block = self._build_conversation_block(conversation_messages)
        return self.prompt_manager.build_socratic_prompt(
            message=message,
            conversation_block=conversation_block or None,
            rag_hint=rag_hint,
            history_note=history_note,
            user_profile_note=user_profile_note,
        )

    def _generate_direct_response(
        self,
        message: str,
        conversation_messages: Optional[List[Dict[str, Any]]] = None,
        rag_hint: Optional[str] = None,
        history_note: Optional[str] = None,
        user_profile_note: Optional[str] = None,
    ) -> str:
        """Generate a direct response without orchestration overhead."""
        prompt = self._build_direct_prompt(
            message=message,
            conversation_messages=conversation_messages,
            rag_hint=rag_hint,
            history_note=history_note,
            user_profile_note=user_profile_note,
        )
        return self._invoke_llm(prompt)

    def _generate_socratic_response(
        self,
        message: str,
        conversation_messages: Optional[List[Dict[str, Any]]] = None,
        rag_hint: Optional[str] = None,
        history_note: Optional[str] = None,
        user_profile_note: Optional[str] = None,
    ) -> str:
        prompt = self._build_socratic_prompt(
            message=message,
            conversation_messages=conversation_messages,
            rag_hint=rag_hint,
            history_note=history_note,
            user_profile_note=user_profile_note,
        )
        return self._invoke_llm(prompt)

    def _generate_direct_response_stream(
        self,
        message: str,
        conversation_messages: Optional[List[Dict[str, Any]]] = None,
        rag_hint: Optional[str] = None,
        history_note: Optional[str] = None,
        user_profile_note: Optional[str] = None,
    ) -> Iterator[str]:
        """Yield direct response tokens/chunks as they arrive."""
        llm = self.llm_provider.get_llm()
        prompt = self._build_direct_prompt(
            message=message,
            conversation_messages=conversation_messages,
            rag_hint=rag_hint,
            history_note=history_note,
            user_profile_note=user_profile_note,
        )

        if hasattr(llm, "stream"):
            try:
                for chunk in llm.stream(prompt):
                    text = str(chunk.text) if hasattr(chunk, "text") else str(chunk)
                    if text:
                        yield text
                return
            except Exception as exc:
                logger.warning(
                    f"LLM native streaming failed, falling back to chunked text: {exc}"
                )

        # Fallback for models/providers without native streaming.
        try:
            full = self._generate_direct_response(
                message,
                conversation_messages,
                rag_hint=rag_hint,
                history_note=history_note,
            )
        except Exception as exc:
            logger.error(f"LLM invocation failed during stream fallback: {exc}")
            yield f"I encountered an error: {self._format_llm_error(exc)}"
            return

        for token in full.split(" "):
            if token:
                yield f"{token} "

    def _format_llm_error(self, exc: Exception) -> str:
        """Translate a raw provider exception into a user-facing error message."""
        error_message = str(exc)
        if (
            self.llm_provider.get_provider_name() == "google"
            and "quota exceeded" in error_message.lower()
        ):
            return (
                "Google Gemini quota exceeded for the selected model. "
                "Choose another Google model in Config or switch provider "
                "(e.g., deepseek/openai), then retry."
            )
        return error_message

    @staticmethod
    def _is_image_document(item: Dict[str, Any]) -> bool:
        file_type = str(item.get("file_type") or "").lower()
        return file_type in _IMAGE_EXTENSIONS

    @staticmethod
    def _encode_image_base64(image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def _build_multimodal_prompt(
        self,
        message: str,
        conversation_messages: Optional[List[Dict[str, Any]]],
        rag_hint: Optional[str],
        user_profile_note: Optional[str] = None,
    ) -> str:
        conversation_block = self._build_conversation_block(conversation_messages)
        prompt_parts = [
            self.prompt_manager.build_direct_prompt(
                message=message,
                conversation_block=conversation_block or None,
                rag_hint=rag_hint,
                user_profile_note=user_profile_note,
            )
        ]
        prompt_parts.append(
            "Analiza las imagenes adjuntas directamente (sin OCR) y responde con base en su contenido visual."
        )
        return "\n\n".join(prompt_parts)

    def _resolve_ollama_base_url(self) -> Optional[str]:
        for candidate in _candidate_base_urls(settings.ollama_base_url):
            if _is_ollama_reachable(candidate):
                return candidate
        return None

    def _generate_ollama_multimodal_response(
        self, prompt: str, image_paths: List[str]
    ) -> str:
        base_url = self._resolve_ollama_base_url()
        if not base_url:
            raise ValueError(
                "No se pudo conectar a Ollama para analisis de imagen. Verifica OLLAMA_BASE_URL."
            )

        payload = {
            "model": self.llm_provider.model_name,
            "prompt": prompt,
            "stream": False,
            "images": [self._encode_image_base64(path) for path in image_paths],
            "options": {
                "temperature": self.llm_provider.temperature,
            },
        }
        response = httpx.post(f"{base_url}/api/generate", json=payload, timeout=90.0)
        response.raise_for_status()
        data = response.json()
        return str(data.get("response") or "").strip()

    def _generate_multimodal_response(
        self,
        prompt: str,
        image_paths: List[str],
    ) -> str:
        provider_name = self.llm_provider.get_provider_name()

        if provider_name == "ollama":
            return self._generate_ollama_multimodal_response(prompt, image_paths)

        llm = self.llm_provider.get_llm()
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for image_path in image_paths:
            extension = Path(image_path).suffix.lower().lstrip(".")
            mime_type = _IMAGE_MIME.get(extension, "image/jpeg")
            image_b64 = self._encode_image_base64(image_path)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{image_b64}"},
                }
            )

        result = llm.invoke([HumanMessage(content=content)])
        return str(result.text) if hasattr(result, "text") else str(result)

    @staticmethod
    def _extract_sources(raw_sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize retriever sources into concise citations."""
        normalized: List[Dict[str, Any]] = []
        for item in raw_sources:
            metadata = item.get("metadata") or {}
            content = (item.get("content") or "").strip()
            page = metadata.get("page")
            if page is None:
                page = metadata.get("page_label")
            try:
                page = int(page) if page is not None and str(page).isdigit() else None
            except Exception:
                page = None

            section = (
                metadata.get("section")
                or metadata.get("heading")
                or metadata.get("title")
            )
            relevance = metadata.get("relevance") or metadata.get("score")
            try:
                relevance = float(relevance) if relevance is not None else None
            except Exception:
                relevance = None

            normalized.append(
                {
                    "document_id": metadata.get("file_hash")
                    or metadata.get("chunk_id"),
                    "filename": metadata.get("filename"),
                    "source": metadata.get("source"),
                    "page": page,
                    "section": section,
                    "scope": metadata.get("scope", "global_rag"),
                    "session_id": metadata.get("session_id"),
                    "relevance": relevance,
                    "excerpt": content[:280],
                }
            )
        return normalized

    @staticmethod
    def _normalize_citation_text(value: Any) -> str:
        text = unicodedata.normalize("NFKD", str(value or "").lower())
        text = "".join(char for char in text if not unicodedata.combining(char))
        return " ".join(text.split())

    @classmethod
    def _source_citation_key(cls, item: Dict[str, Any]) -> str:
        metadata = item.get("metadata") or {}
        content = (item.get("content") or "").strip()[:280]
        payload = [
            cls._normalize_citation_text(
                metadata.get("file_hash")
                or metadata.get("chunk_id")
                or metadata.get("document_id")
            ),
            cls._normalize_citation_text(metadata.get("filename")),
            cls._normalize_citation_text(metadata.get("source")),
            cls._normalize_citation_text(
                metadata.get("page") or metadata.get("page_label")
            ),
            cls._normalize_citation_text(
                metadata.get("section")
                or metadata.get("heading")
                or metadata.get("title")
            ),
            cls._normalize_citation_text(metadata.get("scope") or "global_rag"),
            cls._normalize_citation_text(content),
        ]
        return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))

    @staticmethod
    def _normalize_text(value: str) -> str:
        text = unicodedata.normalize("NFKD", str(value or "").lower())
        text = "".join(char for char in text if not unicodedata.combining(char))
        return " ".join(text.split())

    @classmethod
    def _source_relevance_score(
        cls, response_text: str, source_text: str, section: Optional[str] = None
    ) -> float:
        response = cls._normalize_text(response_text)
        source = cls._normalize_text(source_text)

        if not response or not source:
            return 0.0

        if source in response:
            return 1.0

        response_tokens = {
            token for token in re.findall(r"\w+", response) if len(token) > 3
        }
        source_tokens = [
            token for token in re.findall(r"\w+", source) if len(token) > 3
        ]
        if not source_tokens:
            return 0.0

        overlap = sum(1 for token in source_tokens if token in response_tokens) / len(
            source_tokens
        )

        phrase_bonus = 0.0
        for window in range(min(12, len(source_tokens)), 4, -1):
            for start in range(0, len(source_tokens) - window + 1):
                phrase = " ".join(source_tokens[start : start + window])
                if phrase and phrase in response:
                    phrase_bonus = max(phrase_bonus, min(0.95, 0.2 + window * 0.06))
                    break
            if phrase_bonus:
                break

        section_bonus = 0.0
        if section:
            normalized_section = cls._normalize_text(section)
            if normalized_section and normalized_section in response:
                section_bonus = 0.08

        return min(1.0, max(overlap * 0.85, phrase_bonus) + section_bonus)

    @classmethod
    def _filter_relevant_sources(
        cls,
        response_text: str,
        raw_sources: List[Dict[str, Any]],
        seen_citation_keys: Optional[set[str]] = None,
    ) -> List[Dict[str, Any]]:
        scored_sources: Dict[str, Dict[str, Any]] = {}

        for item in raw_sources:
            metadata = item.get("metadata") or {}
            content = (item.get("content") or "").strip()
            section = (
                metadata.get("section")
                or metadata.get("heading")
                or metadata.get("title")
            )
            score = cls._source_relevance_score(response_text, content, section=section)
            if score < 0.25:
                continue

            citation_key = cls._source_citation_key(item)
            existing = scored_sources.get(citation_key)
            candidate = {
                **item,
                "metadata": {
                    **metadata,
                    "relevance": score,
                },
            }

            if existing is None or score > float(
                (existing.get("metadata") or {}).get("relevance") or 0.0
            ):
                scored_sources[citation_key] = candidate

        ordered_sources = list(scored_sources.values())
        seen_keys = seen_citation_keys or set()

        def _sort_key(item: Dict[str, Any]) -> tuple[bool, float, int]:
            metadata = item.get("metadata") or {}
            citation_key = cls._source_citation_key(item)
            return (
                citation_key not in seen_keys,
                float(metadata.get("relevance") or 0.0),
                len(str(item.get("content") or "")),
            )

        ordered_sources.sort(key=_sort_key, reverse=True)
        return ordered_sources

    def chat(
        self,
        message: str,
        use_rag: bool = True,
        conversation_messages: Optional[List[Dict[str, Any]]] = None,
        session_id: Optional[str] = None,
        session_documents: Optional[List[Dict[str, Any]]] = None,
        session_manager: Optional[Any] = None,
        user_profile_note: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Send a message to the chatbot.

        Args:
            message: User message
            use_rag: Whether to use RAG for this query

        Returns:
            Dictionary with response and metadata
        """
        logger.info(f"Processing chat message: '{message}'")
        logger.info(f"Options: use_rag={use_rag}")

        try:
            classification = classify_question(message)
            used_rag = False

            sources: List[Dict[str, Any]] = []
            concepts = extract_concepts(message)
            repeated_concepts: List[str] = []
            if session_manager is not None and session_id and concepts:
                for concept in concepts:
                    if session_manager.has_seen_concept(session_id, concept):
                        repeated_concepts.append(concept)

            history_note = (
                build_history_note(concepts, repeated=bool(repeated_concepts))
                if concepts
                else None
            )
            recent_citation_keys: set[str] = set()
            if (
                session_manager is not None
                and session_id
                and hasattr(session_manager, "get_recent_source_citation_keys")
            ):
                try:
                    recent_citation_keys = set(
                        session_manager.get_recent_source_citation_keys(session_id)
                    )
                except Exception:
                    recent_citation_keys = set()

            rag_hint: Optional[str] = None
            rag_documents: List[Any] = []

            if use_rag and self.rag_retriever and self.rag_retriever.has_documents():
                contextual_message = message
                conversation_block = self._build_conversation_block(
                    conversation_messages
                )
                if conversation_block:
                    contextual_message = (
                        "Contexto conversacional reciente:\n"
                        f"{conversation_block}\n\n"
                        f"Pregunta actual: {message}"
                    )

                rag_documents = self.rag_retriever.retrieve_documents(
                    contextual_message,
                    session_id=session_id,
                )
                if rag_documents:
                    rag_hint = self.rag_retriever._build_context(rag_documents)
                    used_rag = True

            image_documents = [
                item
                for item in (session_documents or [])
                if self._is_image_document(item)
            ]
            image_paths = [
                str(item.get("source") or "")
                for item in image_documents
                if item.get("source")
            ]

            if classification.is_project_context:
                response = self._generate_socratic_response(
                    message,
                    conversation_messages=conversation_messages,
                    rag_hint=rag_hint,
                    history_note=history_note,
                    user_profile_note=user_profile_note,
                )
            elif image_paths:
                multimodal_prompt = self._build_multimodal_prompt(
                    message=message,
                    conversation_messages=conversation_messages,
                    rag_hint=rag_hint,
                    user_profile_note=user_profile_note,
                )
                response = self._generate_multimodal_response(
                    multimodal_prompt, image_paths
                )
            else:
                response = self._generate_direct_response(
                    message,
                    conversation_messages=conversation_messages,
                    rag_hint=rag_hint,
                    history_note=history_note,
                    user_profile_note=user_profile_note,
                )

            if used_rag:
                raw_sources = [
                    {"content": doc.page_content, "metadata": doc.metadata}
                    for doc in rag_documents
                ]
                filtered_sources = self._filter_relevant_sources(
                    response,
                    raw_sources or [],
                    seen_citation_keys=recent_citation_keys,
                )
                if not filtered_sources and raw_sources:
                    filtered_sources = raw_sources
                sources = self._extract_sources(filtered_sources)[
                    : settings.top_k_results
                ]

            logger.info("Chat response generated successfully")

            return {
                "response": response,
                "provider": self.llm_provider.get_provider_name(),
                "model": self.llm_provider.model_name,
                "used_rag": used_rag,
                "sources": sources,
            }

        except Exception as e:
            logger.error(f"Error processing chat message: {e}")
            error_message = self._format_llm_error(e)
            return {
                "response": f"I encountered an error: {error_message}",
                "error": error_message,
                "provider": self.llm_provider.get_provider_name(),
                "model": self.llm_provider.model_name,
                "used_rag": False,
                "sources": [],
            }

    def chat_stream(
        self,
        message: str,
        use_rag: bool = True,
        conversation_messages: Optional[List[Dict[str, Any]]] = None,
        session_id: Optional[str] = None,
        session_documents: Optional[List[Dict[str, Any]]] = None,
        session_manager: Optional[Any] = None,
        user_profile_note: Optional[str] = None,
    ) -> Iterator[Dict[str, Any]]:
        """Yield partial chunks and a final payload for streaming responses."""
        used_rag = False
        sources: List[Dict[str, Any]] = []
        response_parts: List[str] = []

        classification = classify_question(message)
        rag_hint: Optional[str] = None
        concepts = extract_concepts(message)
        repeated_concepts: List[str] = []
        if session_manager is not None and session_id and concepts:
            for concept in concepts:
                if session_manager.has_seen_concept(session_id, concept):
                    repeated_concepts.append(concept)

        history_note = (
            build_history_note(concepts, repeated=bool(repeated_concepts))
            if concepts
            else None
        )
        recent_citation_keys: set[str] = set()
        if (
            session_manager is not None
            and session_id
            and hasattr(session_manager, "get_recent_source_citation_keys")
        ):
            try:
                recent_citation_keys = set(
                    session_manager.get_recent_source_citation_keys(session_id)
                )
            except Exception:
                recent_citation_keys = set()

        rag_documents: List[Any] = []

        if use_rag and self.rag_retriever and self.rag_retriever.has_documents():
            contextual_message = message
            conversation_block = self._build_conversation_block(conversation_messages)
            if conversation_block:
                contextual_message = (
                    "Contexto conversacional reciente:\n"
                    f"{conversation_block}\n\n"
                    f"Pregunta actual: {message}"
                )

            rag_documents = self.rag_retriever.retrieve_documents(
                contextual_message,
                session_id=session_id,
            )
            if rag_documents:
                rag_hint = self.rag_retriever._build_context(rag_documents)
                used_rag = True

        image_documents = [
            item for item in (session_documents or []) if self._is_image_document(item)
        ]
        image_paths = [
            str(item.get("source") or "")
            for item in image_documents
            if item.get("source")
        ]

        if classification.is_project_context:
            response = self._generate_socratic_response(
                message,
                conversation_messages=conversation_messages,
                rag_hint=rag_hint,
                history_note=history_note,
                user_profile_note=user_profile_note,
            )
            for token in response.split(" "):
                if token:
                    yield {"type": "delta", "content": f"{token} "}
                    response_parts.append(f"{token} ")
        elif image_paths:
            response = self._generate_multimodal_response(
                prompt=self._build_multimodal_prompt(
                    message=message,
                    conversation_messages=conversation_messages,
                    rag_hint=rag_hint,
                    user_profile_note=user_profile_note,
                ),
                image_paths=image_paths,
            )
            for token in response.split(" "):
                if token:
                    yield {"type": "delta", "content": f"{token} "}
                    response_parts.append(f"{token} ")
        else:
            for chunk in self._generate_direct_response_stream(
                message,
                conversation_messages=conversation_messages,
                rag_hint=rag_hint,
                history_note=history_note,
                user_profile_note=user_profile_note,
            ):
                yield {"type": "delta", "content": chunk}
                response_parts.append(chunk)

        full_response = "".join(response_parts).strip()
        raw_sources = [
            {"content": doc.page_content, "metadata": doc.metadata}
            for doc in rag_documents
        ]
        filtered_sources = self._filter_relevant_sources(
            full_response,
            raw_sources or [],
            seen_citation_keys=recent_citation_keys,
        )
        if not filtered_sources and raw_sources:
            filtered_sources = raw_sources
        filtered_sources = self._extract_sources(filtered_sources)[
            : settings.top_k_results
        ]

        yield {
            "type": "final",
            "provider": self.llm_provider.get_provider_name(),
            "model": self.llm_provider.model_name,
            "used_rag": used_rag,
            "sources": filtered_sources,
            "response_type": (
                "socratic" if classification.is_project_context else "direct"
            ),
        }

    def clear_memory(self):
        """Clear conversation memory."""
        logger.info("Conversation memory is not enabled in fast path mode")

    def get_memory_messages(self) -> List[Dict[str, str]]:
        """Get conversation history from memory."""
        return []

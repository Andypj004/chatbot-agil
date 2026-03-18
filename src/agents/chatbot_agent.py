"""Main chatbot agent with a direct fast path for chat and RAG."""

from typing import Optional, Dict, Any, List, Iterator

from src.llm.base import BaseLLMProvider
from src.rag.retriever import RAGRetriever
from src.core.logger import get_logger

logger = get_logger()

_SYSTEM_PROMPT = (
    "Eres un asistente experto en metodologías ágiles (Scrum, Kanban, XP, SAFe, Lean, etc.). "
    "Responde siempre en español, de forma clara, precisa y detallada. "
    "Si la pregunta no está relacionada con metodologías ágiles, respóndela igualmente en español."
)


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

        logger.info(
            "Chatbot agent initialized with fast path, "
            f"rag={'enabled' if rag_retriever else 'disabled'}, "
            f"memory={'disabled' if not enable_memory else 'not_persisted'}"
        )

    def _build_conversation_block(self, conversation_messages: Optional[List[Dict[str, Any]]]) -> str:
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

    def _build_direct_prompt(
        self,
        message: str,
        conversation_messages: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        conversation_block = self._build_conversation_block(conversation_messages)
        if conversation_block:
            return (
                f"{_SYSTEM_PROMPT}\n\n"
                "Contexto conversacional reciente:\n"
                f"{conversation_block}\n\n"
                f"Pregunta actual: {message}\n\n"
                "Respuesta:"
            )
        return f"{_SYSTEM_PROMPT}\n\nPregunta: {message}\n\nRespuesta:"

    def _generate_direct_response(
        self,
        message: str,
        conversation_messages: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Generate a direct response without orchestration overhead."""
        llm = self.llm_provider.get_llm()
        prompt = self._build_direct_prompt(message, conversation_messages)
        result = llm.invoke(prompt)
        # BaseChatModel.invoke() returns an AIMessage; BaseLLM.invoke() returns str.
        return result.content if hasattr(result, "content") else result

    def _generate_direct_response_stream(
        self,
        message: str,
        conversation_messages: Optional[List[Dict[str, Any]]] = None,
    ) -> Iterator[str]:
        """Yield direct response tokens/chunks as they arrive."""
        llm = self.llm_provider.get_llm()
        prompt = self._build_direct_prompt(message, conversation_messages)

        if hasattr(llm, "stream"):
            try:
                for chunk in llm.stream(prompt):
                    text = chunk.content if hasattr(chunk, "content") else str(chunk)
                    if text:
                        yield text
                return
            except Exception as exc:
                logger.warning(f"LLM native streaming failed, falling back to chunked text: {exc}")

        # Fallback for models/providers without native streaming.
        full = self._generate_direct_response(message, conversation_messages)
        for token in full.split(" "):
            if token:
                yield f"{token} "

    @staticmethod
    def _extract_sources(raw_sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize retriever sources into concise citations."""
        normalized: List[Dict[str, Any]] = []
        for item in raw_sources:
            metadata = item.get("metadata") or {}
            content = (item.get("content") or "").strip()
            normalized.append(
                {
                    "document_id": metadata.get("file_hash") or metadata.get("chunk_id"),
                    "filename": metadata.get("filename"),
                    "source": metadata.get("source"),
                    "excerpt": content[:280],
                }
            )
        return normalized

    def chat(
        self,
        message: str,
        use_rag: bool = True,
        conversation_messages: Optional[List[Dict[str, Any]]] = None,
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
            used_rag = False

            sources: List[Dict[str, Any]] = []

            if use_rag and self.rag_retriever and self.rag_retriever.has_documents():
                contextual_message = message
                conversation_block = self._build_conversation_block(conversation_messages)
                if conversation_block:
                    contextual_message = (
                        "Contexto conversacional reciente:\n"
                        f"{conversation_block}\n\n"
                        f"Pregunta actual: {message}"
                    )

                rag_result = self.rag_retriever.query(contextual_message, return_sources=True)
                response = rag_result["answer"]
                sources = self._extract_sources(rag_result.get("sources") or [])
                used_rag = True
            else:
                response = self._generate_direct_response(
                    message,
                    conversation_messages=conversation_messages,
                )

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
            error_message = str(e)
            if (
                self.llm_provider.get_provider_name() == "google"
                and "quota exceeded" in error_message.lower()
            ):
                error_message = (
                    "Google Gemini quota exceeded for the selected model. "
                    "Choose another Google model in Config or switch provider "
                    "(e.g., deepseek/openai), then retry."
                )
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
    ) -> Iterator[Dict[str, Any]]:
        """Yield partial chunks and a final payload for streaming responses."""
        used_rag = False
        sources: List[Dict[str, Any]] = []

        if use_rag and self.rag_retriever and self.rag_retriever.has_documents():
            contextual_message = message
            conversation_block = self._build_conversation_block(conversation_messages)
            if conversation_block:
                contextual_message = (
                    "Contexto conversacional reciente:\n"
                    f"{conversation_block}\n\n"
                    f"Pregunta actual: {message}"
                )

            rag_result = self.rag_retriever.query(contextual_message, return_sources=True)
            response = rag_result["answer"]
            sources = self._extract_sources(rag_result.get("sources") or [])
            used_rag = True
            for token in response.split(" "):
                if token:
                    yield {"type": "delta", "content": f"{token} "}
        else:
            for chunk in self._generate_direct_response_stream(
                message,
                conversation_messages=conversation_messages,
            ):
                yield {"type": "delta", "content": chunk}

        yield {
            "type": "final",
            "provider": self.llm_provider.get_provider_name(),
            "model": self.llm_provider.model_name,
            "used_rag": used_rag,
            "sources": sources,
        }

    def clear_memory(self):
        """Clear conversation memory."""
        logger.info("Conversation memory is not enabled in fast path mode")

    def get_memory_messages(self) -> List[Dict[str, str]]:
        """Get conversation history from memory."""
        return []

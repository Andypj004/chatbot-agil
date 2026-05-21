"""Chat endpoint for interacting with the chatbot"""

import json
from uuid import uuid4

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from src.api.models import ChatRequest, ChatResponse
from src.api.dependencies import get_llm_provider, get_chatbot_agent, get_session_manager
from src.core.config import settings
from src.core.logger import get_logger

logger = get_logger()

router = APIRouter(prefix="/chat", tags=["chat"])


def _build_human_session_title(message: str) -> str:
    """Create a concise, readable title from the first user prompt."""
    compact = " ".join(message.strip().split())
    if not compact:
        return "Nueva conversacion"

    max_len = 56
    if len(compact) <= max_len:
        return compact

    return f"{compact[:max_len].rstrip()}..."


@router.post("", response_model=ChatResponse, summary="Send a message to the chatbot")
async def chat(request: ChatRequest):
    """Send a message to the chatbot and get a response
    
    The chatbot can use:
    - RAG (Retrieval-Augmented Generation) for knowledge base queries
    - Multiple LLM providers (OpenAI, Claude, Gemini, Deepseek)
    
    Args:
        request: Chat request with message and options
        
    Returns:
        Chat response with answer and metadata
    """
    logger.info(f"Received chat request: {request.message[:100]}...")
    
    try:
        session_id = request.session_id or str(uuid4())
        session_manager = get_session_manager()
        session_manager.create_session(session_id)
        session_documents = session_manager.get_session_documents_by_ids(
            session_id=session_id,
            document_ids=request.session_document_ids,
        )

        existing_count = session_manager.get_message_count(session_id)

        history = session_manager.get_messages(
            session_id,
            limit=settings.conversation_context_messages,
            offset=max(
                existing_count - settings.conversation_context_messages,
                0,
            ),
        )

        # Persist user message before generation so session continuity is guaranteed.
        session_manager.append_message(
            session_id=session_id,
            role="user",
            text=request.message,
        )

        # Create a friendly title on the first user turn.
        if existing_count == 0:
            session_manager.update_session_title(
                session_id=session_id,
                title=_build_human_session_title(request.message),
            )

        if request.session_attachments:
            session_manager.append_message(
                session_id=session_id,
                role="system",
                text="",
                attachments=[attachment.model_dump() for attachment in request.session_attachments],
            )

        # Get LLM provider
        llm_provider = get_llm_provider(
            provider_name=request.llm_provider,
            model_name=request.model_name,
            temperature=request.temperature
        )
        
        # Get chatbot agent
        agent = get_chatbot_agent(
            llm_provider=llm_provider,
            provider_name=request.llm_provider,
            model_name=request.model_name,
            temperature=request.temperature,
            use_rag=request.use_rag,
        )

        if request.stream:
            def event_generator():
                full_response = ""
                final_payload = {
                    "provider": llm_provider.get_provider_name(),
                    "model": llm_provider.model_name,
                    "used_rag": False,
                    "sources": [],
                }

                for event in agent.chat_stream(
                    message=request.message,
                    use_rag=request.use_rag,
                    conversation_messages=history,
                    session_id=session_id,
                    session_documents=session_documents,
                    session_manager=session_manager,
                ):
                    if event.get("type") == "delta":
                        content = event.get("content", "")
                        full_response += content
                        yield f"data: {json.dumps({'type': 'delta', 'content': content})}\n\n"
                    elif event.get("type") == "final":
                        final_payload = event

                session_manager.append_message(
                    session_id=session_id,
                    role="assistant",
                    text=full_response.strip(),
                    provider=final_payload.get("provider"),
                    model=final_payload.get("model"),
                    used_rag=final_payload.get("used_rag"),
                    sources=final_payload.get("sources") or [],
                )

                yield (
                    "data: "
                    f"{json.dumps({'type': 'final', 'session_id': session_id, **final_payload})}"
                    "\n\n"
                )

            return StreamingResponse(event_generator(), media_type="text/event-stream")

        # Process message (non-stream)
        result = agent.chat(
            message=request.message,
            use_rag=request.use_rag,
            conversation_messages=history,
            session_id=session_id,
            session_documents=session_documents,
            session_manager=session_manager,
        )

        session_manager.append_message(
            session_id=session_id,
            role="assistant",
            text=result["response"],
            provider=result.get("provider"),
            model=result.get("model"),
            used_rag=result.get("used_rag"),
            sources=result.get("sources") or [],
        )

        session_manager.record_concepts(session_id, request.message, result["response"])

        result["session_id"] = session_id
        return ChatResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing your message: {str(e)}"
        )

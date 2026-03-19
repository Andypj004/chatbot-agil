"""Conversation session and history endpoints."""

from fastapi import APIRouter, Query

from src.api.dependencies import get_session_manager
from src.api.models import SessionHistoryResponse, SessionListResponse, SessionSummary, ConversationMessage
from src.core.config import settings
from src.core.logger import get_logger

logger = get_logger()

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.get("", response_model=SessionListResponse, summary="List conversation sessions")
async def list_sessions(
    q: str | None = Query(default=None, description="Optional text query"),
    limit: int = Query(default=settings.conversation_list_limit, ge=1, le=200),
):
    """List conversation sessions ordered by most recent activity."""
    manager = get_session_manager()
    sessions = manager.list_sessions(limit=limit, query=q)
    return SessionListResponse(
        total_sessions=len(sessions),
        sessions=[SessionSummary(**item) for item in sessions],
    )


@router.get("/{session_id}/history", response_model=SessionHistoryResponse, summary="Get session history")
async def get_session_history(
    session_id: str,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=200, ge=1, le=1000),
):
    """Get messages from a conversation session."""
    manager = get_session_manager()
    messages = manager.get_messages(session_id=session_id, limit=limit, offset=offset)
    total = manager.get_message_count(session_id)

    return SessionHistoryResponse(
        session_id=session_id,
        total_messages=total,
        offset=offset,
        limit=limit,
        messages=[ConversationMessage(**item) for item in messages],
    )


@router.delete("/{session_id}", summary="Delete a conversation session")
async def delete_session(session_id: str):
    """Delete one conversation session and all of its messages."""
    manager = get_session_manager()
    manager.clear_session(session_id)
    return {"message": f"Session {session_id} deleted successfully"}

"""Conversation session and history endpoints."""

from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, status, Depends

from src.api.dependencies import (
    get_session_manager,
    get_vector_store,
    get_current_user_optional,
)
from src.api.models import (
    SessionHistoryResponse,
    SessionListResponse,
    SessionSummary,
    ConversationMessage,
    SessionTitleUpdateRequest,
)
from src.core.config import settings
from src.core.logger import get_logger

logger = get_logger()

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.get(
    "", response_model=SessionListResponse, summary="List conversation sessions"
)
async def list_sessions(
    q: str | None = Query(default=None, description="Optional text query"),
    limit: int = Query(default=settings.conversation_list_limit, ge=1, le=200),
    current_user=Depends(get_current_user_optional),
):
    """List conversation sessions ordered by most recent activity."""
    manager = get_session_manager()
    sessions = manager.list_sessions(
        limit=limit,
        query=q,
        user_id=current_user.get("user_id") if current_user else None,
    )
    return SessionListResponse(
        total_sessions=len(sessions),
        sessions=[SessionSummary(**item) for item in sessions],
    )


@router.get(
    "/{session_id}/history",
    response_model=SessionHistoryResponse,
    summary="Get session history",
)
async def get_session_history(
    session_id: str,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=200, ge=1, le=1000),
    current_user=Depends(get_current_user_optional),
):
    """Get messages from a conversation session."""
    manager = get_session_manager()
    owner_id = current_user.get("user_id") if current_user else None
    session = manager.get_session_record(session_id)
    if session is None or session.get("user_id") not in (None, owner_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Session not found"
        )

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
async def delete_session(
    session_id: str,
    current_user=Depends(get_current_user_optional),
):
    """Delete one conversation session and all of its messages."""
    manager = get_session_manager()
    vector_store = get_vector_store()

    owner_id = current_user.get("user_id") if current_user else None
    session = manager.get_session_record(session_id)
    if session is None or session.get("user_id") not in (None, owner_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Session not found"
        )

    vector_store.delete_by_metadata({"scope": "session_chat", "session_id": session_id})

    session_docs = manager.list_session_documents(session_id)
    for item in session_docs:
        source = item.get("source")
        if source:
            Path(source).unlink(missing_ok=True)

    uploads_dir = Path(f"data/uploads/sessions/{session_id}")
    if uploads_dir.exists():
        for file_path in uploads_dir.glob("*"):
            if file_path.is_file() and not file_path.name.startswith("."):
                file_path.unlink(missing_ok=True)

    manager.clear_session(session_id)
    return {"message": f"Session {session_id} deleted successfully"}


@router.put("/{session_id}/title", summary="Update conversation title")
async def update_session_title(
    session_id: str,
    payload: SessionTitleUpdateRequest,
    current_user=Depends(get_current_user_optional),
):
    """Update a human-readable title for one session."""
    manager = get_session_manager()

    owner_id = current_user.get("user_id") if current_user else None
    session = manager.get_session_record(session_id)
    if session is None or session.get("user_id") not in (None, owner_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found",
        )

    if not payload.title.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Title cannot be empty",
        )

    updated = manager.update_session_title(session_id=session_id, title=payload.title)
    if not updated:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found",
        )

    session = manager.get_session(session_id, user_id=owner_id)
    return {
        "message": "Session title updated successfully",
        "session": session,
    }

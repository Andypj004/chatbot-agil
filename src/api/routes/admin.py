"""Admin-only endpoints for user management."""

from fastapi import APIRouter, Depends, HTTPException, Query, status

from src.api.dependencies import (
    cleanup_user_sessions,
    get_current_admin,
    get_session_manager,
)
from src.api.models import UserListResponse, UserProfileResponse
from src.memory.session_manager import SessionManager

router = APIRouter(prefix="/admin", tags=["admin"])


def _profile(profile: dict) -> UserProfileResponse:
    return UserProfileResponse(
        user_id=profile["user_id"],
        email=profile["email"],
        full_name=profile["full_name"],
        account_type=profile["account_type"],
        knowledge_level=profile["knowledge_level"],
        agile_adoption_level=profile["agile_adoption_level"],
        agile_adoption_label=profile["agile_adoption_label"],
        is_admin=profile.get("is_admin", False),
        created_at=profile["created_at"],
        updated_at=profile["updated_at"],
        last_login_at=profile.get("last_login_at"),
    )


@router.get("/users", response_model=UserListResponse, summary="List all users (admin)")
def list_all_users(
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    _admin=Depends(get_current_admin),
    session_manager: SessionManager = Depends(get_session_manager),
):
    """Return a paginated list of all registered users."""
    users = session_manager.list_users(limit=limit, offset=offset)
    total = session_manager.count_users()
    return UserListResponse(total=total, users=[_profile(u) for u in users])


@router.delete(
    "/users/{user_id}", summary="Delete any user account (admin)"
)
def admin_delete_user(
    user_id: str,
    _admin=Depends(get_current_admin),
    session_manager: SessionManager = Depends(get_session_manager),
):
    """Permanently delete a user account and all their data."""
    if user_id == _admin["user_id"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Un administrador no puede eliminar su propia cuenta desde este endpoint. Use DELETE /auth/me.",
        )
    session_ids = session_manager.delete_user(user_id)
    if session_ids is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Usuario no encontrado",
        )
    cleanup_user_sessions(session_ids)
    return {
        "message": "Usuario eliminado exitosamente",
        "sessions_deleted": len(session_ids),
    }

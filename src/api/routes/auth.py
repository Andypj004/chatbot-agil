"""Authentication routes for user registration and login."""

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.dependencies import get_current_user, get_session_manager
from src.api.models import (
    AuthResponse,
    UserLoginRequest,
    UserProfileResponse,
    UserRegistrationRequest,
)
from src.memory.session_manager import SessionManager

router = APIRouter(prefix="/auth", tags=["auth"])


def _profile_response(profile: dict) -> UserProfileResponse:
    return UserProfileResponse(
        user_id=profile["user_id"],
        email=profile["email"],
        full_name=profile["full_name"],
        account_type=profile["account_type"],
        knowledge_level=profile["knowledge_level"],
        agile_adoption_level=profile["agile_adoption_level"],
        agile_adoption_label=profile["agile_adoption_label"],
        created_at=profile["created_at"],
        updated_at=profile["updated_at"],
        last_login_at=profile.get("last_login_at"),
    )


@router.post("/register", response_model=AuthResponse, summary="Register a new user")
def register_user(
    request: UserRegistrationRequest,
    session_manager: SessionManager = Depends(get_session_manager),
):
    """Create a user account and return an auth token."""
    try:
        questionnaire_answers = [
            item.model_dump() for item in request.questionnaire_answers
        ]
        profile = session_manager.create_user(
            email=request.email,
            password=request.password,
            full_name=request.full_name,
            account_type=request.account_type,
            knowledge_level=request.knowledge_level,
            questionnaire_answers=questionnaire_answers,
        )
        token = session_manager.issue_user_token(profile["user_id"])
        return AuthResponse(access_token=token, user=_profile_response(profile))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


@router.post(
    "/login", response_model=AuthResponse, summary="Log in with email and password"
)
def login_user(
    request: UserLoginRequest,
    session_manager: SessionManager = Depends(get_session_manager),
):
    """Authenticate a user and return a fresh auth token."""
    profile = session_manager.authenticate_user(request.email, request.password)
    if profile is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    token = profile.pop("access_token")
    return AuthResponse(access_token=token, user=_profile_response(profile))


@router.get(
    "/me",
    response_model=UserProfileResponse,
    summary="Get the authenticated user profile",
)
def get_me(current_user=Depends(get_current_user)):
    """Return the current authenticated user."""
    return _profile_response(current_user)

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Any

from src.core.forms.form_manager import form_manager
from src.api.dependencies import get_session_manager, get_current_user_optional
from src.memory.session_manager import SessionManager

router = APIRouter(prefix="/forms", tags=["forms"])


class StartFormRequest(BaseModel):
    form_id: str
    session_id: str


class AnswerRequest(BaseModel):
    form_id: str
    session_id: str
    name: str
    value: Any


@router.post("/start")
def start_form(
    req: StartFormRequest,
    session_manager: SessionManager = Depends(get_session_manager),
):
    try:
        q = form_manager.start_form(
            req.form_id, req.session_id, session_manager=session_manager
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="form not found")
    return {"ok": True, "question": q}


@router.post("/answer")
def answer_form(
    req: AnswerRequest,
    session_manager: SessionManager = Depends(get_session_manager),
    current_user=Depends(get_current_user_optional),
):
    try:
        res = form_manager.answer(
            req.form_id,
            req.session_id,
            req.name,
            req.value,
            session_manager=session_manager,
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    if (
        req.form_id == "agile_adoption_assessment"
        and res.get("completed")
        and current_user is not None
    ):
        answers = res.get("answers") or {}
        session_manager.update_user_agile_profile(
            user_id=current_user["user_id"],
            questionnaire_answers=[
                {
                    "question_number": index + 1,
                    "answer": answers.get(f"question_{index + 1}"),
                }
                for index in range(5)
            ],
            knowledge_level=current_user.get("knowledge_level"),
        )

    return res

from typing import Any, Dict, Optional

from src.memory.session_manager import SessionManager


def init_session_form(
    session_manager: SessionManager,
    session_id: str,
    form_id: str,
    spec,
) -> None:
    state = {"current_index": 0, "answers": {}}
    session_manager.upsert_form_state(session_id, form_id, state)


def get_state(
    session_manager: SessionManager,
    session_id: str,
    form_id: str,
) -> Optional[Dict[str, Any]]:
    return session_manager.get_form_state(session_id, form_id)


def save_answer(
    session_manager: SessionManager,
    session_id: str,
    form_id: str,
    name: str,
    value: Any,
    spec_len: int,
) -> bool:
    state = session_manager.get_form_state(session_id, form_id) or {"current_index": 0, "answers": {}}
    state["answers"][name] = value
    state["current_index"] = state.get("current_index", 0) + 1
    session_manager.upsert_form_state(session_id, form_id, state)
    return state["current_index"] >= spec_len


def get_answers(
    session_manager: SessionManager,
    session_id: str,
    form_id: str,
) -> Dict[str, Any]:
    state = get_state(session_manager, session_id, form_id)
    if not state:
        return {}
    return dict(state.get("answers", {}))

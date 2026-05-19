from threading import Lock
from typing import Any, Dict

_store: Dict[str, Dict[str, Any]] = {}
_lock = Lock()


def _key(session_id: str, form_id: str) -> str:
    return f"{session_id}:::{form_id}"


def init_session_form(session_id: str, form_id: str, spec) -> None:
    k = _key(session_id, form_id)
    with _lock:
        _store[k] = {"spec": spec, "answers": {}, "current_index": 0}


def get_state(session_id: str, form_id: str) -> Dict[str, Any]:
    return _store.get(_key(session_id, form_id))


def save_answer(session_id: str, form_id: str, name: str, value: Any) -> bool:
    k = _key(session_id, form_id)
    with _lock:
        state = _store.get(k)
        if not state:
            raise KeyError("form not started")

        state["answers"][name] = value
        state["current_index"] = state.get("current_index", 0) + 1
        spec = state.get("spec")
        if state["current_index"] >= len(spec.fields):
            return True
        return False


def get_answers(session_id: str, form_id: str) -> Dict[str, Any]:
    state = get_state(session_id, form_id)
    if not state:
        return {}
    return dict(state.get("answers", {}))

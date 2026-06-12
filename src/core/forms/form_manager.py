from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.core.forms import validators as validators_module
from src.memory import form_state
from src.memory import form_state_session
from src.memory.session_manager import SessionManager


@dataclass
class FormSpec:
    form_id: str
    title: str
    fields: List[Dict[str, Any]]
    cross_validators: Optional[List[Dict[str, Any]]] = None


class FormManager:
    """Registry and driver for conversational forms."""

    def __init__(self) -> None:
        self._registry: Dict[str, FormSpec] = {}

    def register_form(self, spec: FormSpec) -> None:
        self._registry[spec.form_id] = spec

    def get_form(self, form_id: str) -> Optional[FormSpec]:
        return self._registry.get(form_id)

    def start_form(
        self,
        form_id: str,
        session_id: str,
        session_manager: Optional[SessionManager] = None,
    ) -> Dict[str, Any]:
        spec = self.get_form(form_id)
        if not spec:
            raise KeyError(f"unknown form: {form_id}")

        if session_manager is not None:
            form_state_session.init_session_form(
                session_manager, session_id, form_id, spec
            )
        else:
            form_state.init_session_form(session_id, form_id, spec)

        return self._current_question(
            session_id, form_id, session_manager=session_manager
        )

    def answer(
        self,
        form_id: str,
        session_id: str,
        name: str,
        value: Any,
        session_manager: Optional[SessionManager] = None,
    ) -> Dict[str, Any]:
        spec = self.get_form(form_id)
        if not spec:
            raise KeyError(f"unknown form: {form_id}")

        field = next((f for f in spec.fields if f.get("name") == name), None)
        if not field:
            raise KeyError(f"unknown field: {name}")

        validator = field.get("validator")
        if validator:
            valid, msg = validators_module.run_validator(validator, value, field)
        else:
            valid, msg = validators_module.validate_field(field, value)

        if not valid:
            return {"ok": False, "error": msg}

        if session_manager is not None:
            completed = form_state_session.save_answer(
                session_manager,
                session_id,
                form_id,
                name,
                value,
                len(spec.fields),
            )
            answers = form_state_session.get_answers(
                session_manager, session_id, form_id
            )
        else:
            completed = form_state.save_answer(session_id, form_id, name, value)
            answers = form_state.get_answers(session_id, form_id)

        cross_validators = spec.cross_validators or []
        if cross_validators:
            ok, msg = validators_module.validate_cross_fields(cross_validators, answers)
            if not ok:
                return {"ok": False, "error": msg}

        if completed:
            return {"ok": True, "completed": True, "answers": answers}

        return {
            "ok": True,
            "completed": False,
            "next": self._current_question(
                session_id, form_id, session_manager=session_manager
            ),
        }

    def _current_question(
        self,
        session_id: str,
        form_id: str,
        session_manager: Optional[SessionManager] = None,
    ) -> Dict[str, Any]:
        if session_manager is not None:
            state = form_state_session.get_state(session_manager, session_id, form_id)
        else:
            state = form_state.get_state(session_id, form_id)

        if not state:
            raise KeyError("form not started")

        idx = state.get("current_index", 0)
        spec = self.get_form(form_id)
        idx = self._next_applicable_index(spec, idx, state.get("answers", {}))
        if idx >= len(spec.fields):
            return {"completed": True}

        # Persist updated index when we skip hidden fields.
        if idx != state.get("current_index", 0):
            if session_manager is not None:
                state["current_index"] = idx
                session_manager.upsert_form_state(session_id, form_id, state)
            else:
                state["current_index"] = idx

        field = spec.fields[idx]
        return {
            "name": field.get("name"),
            "label": field.get("label"),
            "type": field.get("type"),
            "required": field.get("required", False),
        }

    @staticmethod
    def _next_applicable_index(
        spec: FormSpec, start: int, answers: Dict[str, Any]
    ) -> int:
        idx = start
        while idx < len(spec.fields):
            field = spec.fields[idx]
            depends_on = field.get("depends_on")
            if not depends_on:
                return idx

            ref = depends_on.get("field")
            expected = depends_on.get("value")
            expected_any = depends_on.get("values")
            actual = answers.get(ref)

            if expected_any is not None:
                if actual in expected_any:
                    return idx
            elif expected is not None:
                if actual == expected:
                    return idx
            else:
                if actual is not None:
                    return idx

            idx += 1

        return idx


form_manager = FormManager()

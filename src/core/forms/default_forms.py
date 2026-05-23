from src.core.forms.form_manager import FormManager, FormSpec

_REGISTERED = False


def register_default_forms(manager: FormManager) -> None:
    global _REGISTERED
    if _REGISTERED:
        return

    manager.register_form(
        FormSpec(
            form_id="project_brief",
            title="Project Brief",
            fields=[
                {
                    "name": "project_name",
                    "label": "Nombre del proyecto",
                    "type": "text",
                    "required": True,
                    "validator": "non_empty_str",
                },
                {
                    "name": "team_size",
                    "label": "Tamano del equipo",
                    "type": "int",
                    "required": True,
                },
                {
                    "name": "sprint_length_weeks",
                    "label": "Duracion del sprint (semanas)",
                    "type": "int",
                    "required": True,
                },
                {
                    "name": "current_role",
                    "label": "Rol actual",
                    "type": "text",
                    "required": True,
                    "choices": ["Product Owner", "Scrum Master", "Developer", "Otro"],
                },
            ],
        )
    )

    _REGISTERED = True

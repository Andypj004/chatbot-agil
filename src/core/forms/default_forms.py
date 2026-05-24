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

    manager.register_form(
        FormSpec(
            form_id="agile_adoption_assessment",
            title="Cuestionario de Evaluacion Agil Sustentado",
            fields=[
                {
                    "name": "question_1",
                    "label": "Ante una modificación imprevista en los requisitos a mitad del ciclo, ¿cuál es la postura metodológica correcta?",
                    "type": "text",
                    "required": True,
                    "choices": ["a", "b", "c", "d"],
                },
                {
                    "name": "question_2",
                    "label": "Con respecto a la frecuencia de las entregas y la planificación del producto, ¿cuál descripción le parece correcta?",
                    "type": "text",
                    "required": True,
                    "choices": ["a", "b", "c", "d"],
                },
                {
                    "name": "question_3",
                    "label": "¿Cómo se concibe la dinámica de trabajo, la asignación de tareas y las interacciones dentro del equipo?",
                    "type": "text",
                    "required": True,
                    "choices": ["a", "b", "c", "d"],
                },
                {
                    "name": "question_4",
                    "label": "¿Cuándo se define que una funcionalidad está realmente concluida?",
                    "type": "text",
                    "required": True,
                    "choices": ["a", "b", "c", "d"],
                },
                {
                    "name": "question_5",
                    "label": "¿Cómo debe ser la relación e interacción con el cliente y los stakeholders durante el desarrollo?",
                    "type": "text",
                    "required": True,
                    "choices": ["a", "b", "c", "d"],
                },
            ],
        )
    )

    _REGISTERED = True

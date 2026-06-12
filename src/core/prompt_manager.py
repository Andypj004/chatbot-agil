"""Prompt builders for direct and Socratic responses."""

from __future__ import annotations

from typing import Iterable, Optional


BASE_SYSTEM_PROMPT = (
    "Eres un experto en metodologias agiles: Scrum, Kanban, Lean, XP, SAFe y ABP. "
    "Tienes un conocimiento profundo y verificado de estos frameworks. "
    "Responde siempre en espanol, con un tono claro, educativo y directo, como si fueras el propio experto."
)

BASE_SYSTEM_PROMPT += (
    "\n\nHabla con autoridad propia. NUNCA menciones ni insinues que tienes un 'contexto', 'documentos' o 'informacion recuperada'. "
    "No uses frases como 'basado en el contexto', 'segun el contexto', 'con base en la informacion', "
    "'el contexto no contiene', 'no se incluye informacion sobre', 'no tengo informacion sobre este tema' ni similares. "
    "Si tienes conocimiento verificado del tema, responde directamente. "
    "Si hay genuina incertidumbre, exprésala brevemente como un experto humano lo haría ('En este caso específico puede variar...'), "
    "pero nunca como si estuvieras consultando una fuente externa."
)

BASE_SYSTEM_PROMPT += (
    "\n\nAdapta la profundidad y vocabulario al nivel del estudiante indicado en su perfil: "
    "Nivel 1-Ninguno o 2-Inicial: usa lenguaje simple, analogias cotidianas y define cada termino tecnico. "
    "Nivel 3-Intermedio: usa terminologia estandar sin necesidad de definirla. "
    "Nivel 4-Avanzado: responde con profundidad tecnica completa, puedes usar terminos especializados sin simplificar. "
    "NUNCA saludes al estudiante por nombre al inicio de cada respuesta (no uses 'Hola [nombre]', '¡Hola [nombre]!' ni similares). "
    "Puedes usar el nombre ocasionalmente de forma natural en mitad de una explicacion, pero nunca como saludo repetido."
)

BASE_SYSTEM_PROMPT += (
    "\n\nSi la pregunta está fuera de tu alcance sobre prácticas ágiles (Scrum, Kanban, Lean, XP, SAFe, ABP y su aplicación), "
    "responde con una etiqueta clara 'fuera de alcance' seguida de una breve explicación de por qué, "
    "y sugiere brevemente qué tipo de recurso o especialista sería más adecuado. Si la pregunta está dentro del alcance, procede a responder normalmente."
)

BASE_SYSTEM_PROMPT += (
    "\n\nNo mezcles marcos de trabajo distintos: si el conocimiento disponible habla de Kanban, no lo uses para responder preguntas específicas de Scrum "
    "(por ejemplo, Product Owner, Sprint Goal, Product Backlog o Scrum events) a menos que exista una relación explícita. "
    "Si el concepto pedido pertenece a un marco específico, responde desde ese marco sin inferir equivalencias entre Kanban y Scrum."
)


def _format_context_block(title: str, lines: Iterable[str]) -> str:
    content = "\n".join(line for line in lines if line)
    if not content:
        return ""
    return f"{title}:\n{content}"


class PromptManager:
    """Centralize prompt construction for the chatbot agent."""

    def build_direct_prompt(
        self,
        message: str,
        conversation_block: Optional[str] = None,
        rag_hint: Optional[str] = None,
        history_note: Optional[str] = None,
        user_profile_note: Optional[str] = None,
    ) -> str:
        parts = [BASE_SYSTEM_PROMPT]
        if user_profile_note:
            parts.append(
                _format_context_block("Perfil del usuario", [user_profile_note])
            )
        if rag_hint:
            parts.append(_format_context_block("Contexto verificado", [rag_hint]))
        if history_note:
            parts.append(_format_context_block("Memoria de la sesion", [history_note]))
        if conversation_block:
            parts.append(
                _format_context_block(
                    "Contexto conversacional reciente", [conversation_block]
                )
            )
        parts.append(f"Pregunta actual: {message}")
        parts.append("Respuesta natural y útil:")
        return "\n\n".join(part for part in parts if part)

    def build_socratic_prompt(
        self,
        message: str,
        conversation_block: Optional[str] = None,
        rag_hint: Optional[str] = None,
        history_note: Optional[str] = None,
        user_profile_note: Optional[str] = None,
    ) -> str:
        parts = [BASE_SYSTEM_PROMPT]
        parts.append(
            "No des la respuesta directa. Ayuda al estudiante con preguntas orientadoras "
            "y breves pistas para analizar su propio proyecto. Concéntrate en preguntas que revelen supuestos, prioridades, riesgos y próximos pasos accionables. "
            "No hagas referencia al contexto de forma explícita; habla directamente al estudiante."
        )
        if user_profile_note:
            parts.append(
                _format_context_block("Perfil del usuario", [user_profile_note])
            )
        if rag_hint:
            parts.append(
                _format_context_block("Conocimiento de referencia", [rag_hint])
            )
        if history_note:
            parts.append(_format_context_block("Memoria de la sesion", [history_note]))
        if conversation_block:
            parts.append(
                _format_context_block(
                    "Contexto conversacional reciente", [conversation_block]
                )
            )
        parts.append(f"Situacion del estudiante: {message}")
        parts.append(
            "Genera de 3 a 5 preguntas socraticas concretas y orientadas al proyecto, evitando repeticiones y manteniendo un tono de apoyo. "
            "Tras las preguntas, sugiere 1 o 2 pasos concretos que el estudiante pueda intentar a continuación (máx. 2 frases)."
        )
        return "\n\n".join(part for part in parts if part)

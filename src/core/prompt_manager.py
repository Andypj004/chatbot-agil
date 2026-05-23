"""Prompt builders for direct and Socratic responses."""

from __future__ import annotations

from typing import Iterable, Optional


BASE_SYSTEM_PROMPT = (
    "Eres un asistente experto en metodologias agiles. Responde siempre en espanol, "
    "con un tono claro, consistente y educativo. Evita inventar informacion."
)

BASE_SYSTEM_PROMPT += (
    "\n\nResponde de forma natural y directa. No empieces con frases como 'basado en el contexto', "
    "'segun el contexto' o 'con base en la informacion recuperada'. Usa la evidencia solo para sostener tu respuesta, "
    "sin mencionarla de forma explicita salvo que el usuario lo pida."
)

# Instruccion adicional: pide al LLM que determine si la pregunta cae fuera del alcance
# de practicas agiles y lo marque claramente cuando corresponda.
BASE_SYSTEM_PROMPT += (
    "\n\nSi la pregunta está fuera de tu alcance sobre prácticas ágiles (Scrum, Kanban, Lean, XP, SAFe, ABP y su aplicación), "
    "responde con una etiqueta clara 'fuera de alcance' seguida de una breve explicación de por qué, "
    "y sugiere brevemente qué tipo de recurso o especialista sería más adecuado. Si la pregunta está dentro del alcance, procede a responder normalmente."
)

BASE_SYSTEM_PROMPT += (
    "\n\nNo mezcles marcos de trabajo distintos: si el contexto recuperado habla de Kanban, no lo uses para responder preguntas específicas de Scrum "
    "(por ejemplo, Product Owner, Sprint Goal, Product Backlog o Scrum events) a menos que el contexto lo mencione explícitamente. "
    "Si la evidencia recuperada no contiene el concepto exacto pedido, indícalo y no infieras equivalencias entre Kanban y Scrum."
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
    ) -> str:
        parts = [BASE_SYSTEM_PROMPT]
        if rag_hint:
            parts.append(_format_context_block("Contexto verificado", [rag_hint]))
        if history_note:
            parts.append(_format_context_block("Memoria de la sesion", [history_note]))
        if conversation_block:
            parts.append(_format_context_block("Contexto conversacional reciente", [conversation_block]))
        parts.append(f"Pregunta actual: {message}")
        parts.append("Respuesta natural y útil:")
        return "\n\n".join(part for part in parts if part)

    def build_socratic_prompt(
        self,
        message: str,
        conversation_block: Optional[str] = None,
        rag_hint: Optional[str] = None,
        history_note: Optional[str] = None,
    ) -> str:
        parts = [BASE_SYSTEM_PROMPT]
        parts.append(
            "No des la respuesta directa. Ayuda al estudiante con preguntas orientadoras "
            "y breves pistas para analizar su propio proyecto. Concéntrate en preguntas que revelen supuestos, prioridades, riesgos y próximos pasos accionables. "
            "No hagas referencia al contexto de forma explícita; habla directamente al estudiante."
        )
        if rag_hint:
            parts.append(_format_context_block("Conocimiento de referencia", [rag_hint]))
        if history_note:
            parts.append(_format_context_block("Memoria de la sesion", [history_note]))
        if conversation_block:
            parts.append(_format_context_block("Contexto conversacional reciente", [conversation_block]))
        parts.append(f"Situacion del estudiante: {message}")
        parts.append(
            "Genera de 3 a 5 preguntas socraticas concretas y orientadas al proyecto, evitando repeticiones y manteniendo un tono de apoyo. "
            "Tras las preguntas, sugiere 1 o 2 pasos concretos que el estudiante pueda intentar a continuación (máx. 2 frases)."
        )
        return "\n\n".join(part for part in parts if part)
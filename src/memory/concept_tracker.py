"""Helpers for tracking repeated agile concepts within a session."""

from __future__ import annotations

from typing import Iterable


AGILE_CONCEPTS = (
    "scrum master",
    "product owner",
    "development team",
    "daily",
    "dailies",
    "retrospectiva",
    "sprint",
    "backlog",
    "planning",
    "review",
    "scrum",
    "kanban",
    "lean",
    "xp",
)


def _normalize(text: str) -> str:
    return " ".join(text.lower().strip().split())


def extract_concepts(text: str) -> list[str]:
    """Return a deduplicated list of known agile concepts found in text."""

    normalized = _normalize(text)
    matches = []
    for concept in AGILE_CONCEPTS:
        if concept in normalized:
            matches.append(concept)
    return list(dict.fromkeys(matches))


def build_history_note(concepts: Iterable[str], repeated: bool) -> str:
    concept_list = ", ".join(dict.fromkeys(concepts))
    if not concept_list:
        return ""

    if repeated:
        return (
            f"El estudiante ya ha consultado antes sobre {concept_list}. "
            "Evita repetir la definicion base y aumenta el nivel de detalle con un ejemplo o una comparacion."
        )

    return (
        f"Se identificaron los conceptos {concept_list}. "
        "Da una explicacion clara y luego ampliala con una aplicacion practica breve."
    )
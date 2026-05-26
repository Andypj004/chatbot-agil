"""Question classification helpers for pedagogical routing."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable


_PROJECT_KEYWORDS = (
    "mi proyecto",
    "nuestro proyecto",
    "en mi proyecto",
    "en nuestro proyecto",
    "nuestro sprint",
    "mi sprint",
    "nuestro equipo",
    "mi equipo",
    "nuestro caso",
    "en mi caso",
    "en nuestro caso",
    "para la entrega",
    "para el proyecto",
    "para nuestra retrospectiva",
    "para nuestras dailies",
    "mi producto",
    "nuestro producto",
    "en mi producto",
    "en nuestro producto",
)


_PROJECT_PATTERNS = (
    r"\bcomo\s+podemos\b",
    r"\bcomo\s+deberiamos\b",
    r"\bque\s+podemos\s+hacer\b",
    r"\bcomo\s+organizamos\b",
    r"\bcomo\s+aplicamos\b",
    r"\bque\s+deberiamos\b",
    r"\bcomo\s+deberi[áa]mos\b",
)

# Patrones que indican intención de obtener conocimiento/definición.
# Cuando la pregunta busca una definición o explicación, no debe enrutarse
# como contexto de proyecto aunque mencione "mi equipo" u otras señales.
_KNOWLEDGE_INTENT_PATTERNS = (
    r"\bqu[eé]\s+es\b",
    r"\bqu[eé]\s+son\b",
    r"\bqu[eé]\s+significa\b",
    r"\bcómo\s+funciona\b",
    r"\bcomo\s+funciona\b",
    r"\bcu[aá]l\s+es\s+la\s+diferencia\b",
    r"\bpor\s+qu[eé]\s+es\s+importante\b",
    r"\bpara\s+qu[eé]\s+sirve\b",
    r"\bcu[aá]ndo\s+se\s+usa\b",
    r"\bexpl[ií]came\b",
    r"\bexplica\s+qu[eé]\b",
    r"\bdefine\b",
    r"\bdefin[ií]\b",
    r"\bqu[eé]\s+significa\b",
    r"\bqu[eé]\s+implica\b",
    r"\bcómo\s+se\s+define\b",
    r"\bcomo\s+se\s+define\b",
    r"\bqu[eé]\s+es\s+un\b",
    r"\bqu[eé]\s+es\s+una\b",
    r"\bqu[eé]\s+es\s+el\b",
    r"\bqu[eé]\s+es\s+la\b",
    r"\bqu[eé]\s+es\s+eso\b",
    r"\bpu[eé]des?\s+explicar\b",
    r"\bpodrias?\s+explicar\b",
)


@dataclass(frozen=True)
class QuestionClassification:
    """Lightweight routing decision for a user question."""

    is_project_context: bool
    confidence: float
    matched_signals: tuple[str, ...] = ()


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().strip().split())


def _match_patterns(text: str, patterns: Iterable[str]) -> tuple[str, ...]:
    matches = []
    for pattern in patterns:
        if re.search(pattern, text):
            matches.append(pattern)
    return tuple(matches)


def classify_question(message: str) -> QuestionClassification:
    """Classify whether the question is about the learner's project context.

    Knowledge-intent patterns (¿Qué es?, ¿Cómo funciona?, etc.) take priority
    over project-context signals: a student asking for a definition should receive
    a direct answer backed by RAG, not a Socratic prompt.
    """

    normalized = _normalize_text(message)

    knowledge_signals = _match_patterns(normalized, _KNOWLEDGE_INTENT_PATTERNS)
    if knowledge_signals:
        return QuestionClassification(
            is_project_context=False,
            confidence=0.85,
            matched_signals=knowledge_signals,
        )

    matched_keywords = tuple(keyword for keyword in _PROJECT_KEYWORDS if keyword in normalized)
    matched_patterns = _match_patterns(normalized, _PROJECT_PATTERNS)

    signals = matched_keywords + matched_patterns
    if signals:
        confidence = min(0.95, 0.55 + 0.1 * len(signals))
        return QuestionClassification(
            is_project_context=True,
            confidence=confidence,
            matched_signals=signals,
        )

    return QuestionClassification(
        is_project_context=False,
        confidence=0.6,
        matched_signals=(),
    )
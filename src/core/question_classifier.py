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
    """Classify whether the question is about the learner's project context."""

    normalized = _normalize_text(message)
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
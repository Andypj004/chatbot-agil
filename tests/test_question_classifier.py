"""Tests for question classification and prompt routing."""

from src.core.prompt_manager import PromptManager
from src.core.question_classifier import classify_question


def test_classify_technical_question_as_non_project_context():
    result = classify_question("¿Qué es Scrum y cuáles son sus roles?")

    assert result.is_project_context is False
    assert result.confidence >= 0.6
    assert result.matched_signals  # knowledge-intent pattern detected


def test_classify_pure_factual_question_as_non_project_context():
    result = classify_question("El tablero tiene columnas de tareas")

    assert result.is_project_context is False
    assert result.confidence == 0.6
    assert result.matched_signals == ()


def test_classify_project_question_as_project_context():
    result = classify_question("¿Cómo podemos organizar nuestro sprint para el proyecto final?")

    assert result.is_project_context is True
    assert result.confidence > 0.55
    assert result.matched_signals


def test_prompt_manager_builds_socratic_prompt():
    prompt = PromptManager().build_socratic_prompt(
        message="¿Cómo deberíamos abordar la retrospectiva de nuestro equipo?",
        conversation_block="Usuario: ya tuvimos conflictos con las tareas",
        rag_hint="La retrospectiva sirve para inspeccionar y adaptar el proceso.",
    )

    assert "No des la respuesta directa" in prompt
    assert "preguntas socraticas" in prompt.lower()
    assert "Contexto conversacional reciente" in prompt
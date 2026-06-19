"""LLM-as-judge evaluation for RAG generation quality.

Computes Faithfulness, Answer Relevancy and Hallucination Rate via an LLM
judge. Context Precision/Recall are intentionally NOT requested from the
judge — those remain deterministic metrics from `retrieval_metrics.py`, to
avoid having two different numbers sharing the same name.
"""

import json
import re
from dataclasses import dataclass
from typing import Optional

from src.llm.factory import LLMFactory


JUDGE_PROMPT_TEMPLATE = """Eres un evaluador experto e imparcial de sistemas de respuesta a preguntas basados en recuperación de información (RAG) sobre metodologías ágiles.

Se te proporciona:
1. Una PREGUNTA realizada por un estudiante.
2. El CONTEXTO recuperado por el sistema (fragmentos de documentos) que se usó para generar la respuesta.
3. La RESPUESTA GENERADA por el sistema.
4. Opcionalmente, una RESPUESTA DE REFERENCIA (elaborada por un experto humano) para comparación.

Evalúa la RESPUESTA GENERADA según los siguientes tres criterios, cada uno con un puntaje entre 0.0 y 1.0 (puedes usar decimales), y una breve justificación en español para cada uno:

- "faithfulness" (Fidelidad): ¿La respuesta generada está fundamentada y es consistente con la información presente en el CONTEXTO? 1.0 = todas las afirmaciones están respaldadas por el contexto; 0.0 = la respuesta contradice o no tiene relación alguna con el contexto.
- "answer_relevancy" (Relevancia de la respuesta): ¿La respuesta generada responde directamente y de forma completa a la PREGUNTA realizada? 1.0 = responde completa y directamente; 0.0 = no responde a la pregunta o es irrelevante.
- "hallucination_rate" (Tasa de alucinación): ¿Qué proporción de la respuesta contiene afirmaciones que NO se pueden verificar con el CONTEXTO ni son hechos generales y ampliamente aceptados sobre metodologías ágiles? 1.0 = la respuesta está plagada de afirmaciones inventadas o no verificables; 0.0 = no hay ninguna afirmación inventada.

PREGUNTA:
{question}

CONTEXTO RECUPERADO:
{context}

RESPUESTA GENERADA:
{generated_answer}

RESPUESTA DE REFERENCIA (puede estar vacía):
{reference_answer}

Responde EXCLUSIVAMENTE con un objeto JSON válido, sin texto adicional antes ni después, sin bloques de código markdown, con exactamente esta estructura:

{{
  "faithfulness": {{"score": 0.0, "justification": "..."}},
  "answer_relevancy": {{"score": 0.0, "justification": "..."}},
  "hallucination_rate": {{"score": 0.0, "justification": "..."}}
}}
"""

_REQUIRED_KEYS = ("faithfulness", "answer_relevancy", "hallucination_rate")

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)
_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True)
class JudgeScore:
    """A single judge-assigned score with its justification."""
    score: float
    justification: str


@dataclass(frozen=True)
class JudgeResult:
    """Full result of an LLM-as-judge evaluation for one generated answer."""
    faithfulness: JudgeScore
    answer_relevancy: JudgeScore
    hallucination_rate: JudgeScore
    raw_response: str
    parse_error: Optional[str] = None


def build_judge_prompt(
    question: str,
    context: str,
    generated_answer: str,
    reference_answer: str = "",
) -> str:
    """Build the judge prompt by substituting the four placeholders.

    Args:
        question: The question asked by the student.
        context: Retrieved context used to generate the answer.
        generated_answer: The system-generated answer to evaluate.
        reference_answer: Optional human-authored reference answer.

    Returns:
        The fully substituted prompt string, ready to send to an LLM.
    """
    return JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        context=context,
        generated_answer=generated_answer,
        reference_answer=reference_answer,
    )


def invoke_judge(prompt: str, llm_provider=None) -> str:
    """Invoke the judge LLM with the given prompt and return the raw text.

    Args:
        prompt: The judge prompt to send.
        llm_provider: Optional provider override, used only for test
            injection. If None, uses `LLMFactory.create_provider()` with
            no overrides, i.e. whatever `default_llm_provider`/`default_model`
            are configured in `.env`.

    Returns:
        The raw text response from the LLM.
    """
    if llm_provider is None:
        llm_provider = LLMFactory.create_provider()

    llm = llm_provider.get_llm()
    result = llm.invoke(prompt)
    return str(result.text) if hasattr(result, "text") else result


def _strip_code_fences(text: str) -> str:
    """Strip surrounding markdown code fences (```json ... ``` or ``` ... ```)."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = _CODE_FENCE_RE.sub("", stripped).strip()
    return stripped


def _clamp_score(value) -> float:
    """Coerce a value to float and clamp it into [0.0, 1.0]."""
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score


def _build_score(data: dict, key: str) -> JudgeScore:
    """Build a JudgeScore for `key` from parsed JSON `data`.

    Missing field, missing sub-fields, or wrong types are handled
    defensively: score defaults to 0.0 and a synthetic justification
    is produced.
    """
    entry = data.get(key) if isinstance(data, dict) else None

    if not isinstance(entry, dict):
        return JudgeScore(
            score=0.0,
            justification=f"Campo '{key}' ausente o inválido en la respuesta del juez.",
        )

    has_score = "score" in entry
    score = _clamp_score(entry.get("score")) if has_score else 0.0

    justification = entry.get("justification")
    if not isinstance(justification, str) or not justification.strip():
        missing_parts = []
        if not has_score:
            missing_parts.append("score")
        missing_parts.append("justification")
        justification = (
            f"Justificación sintética: falta el campo "
            f"{'/'.join(dict.fromkeys(missing_parts))} para '{key}' en la respuesta del juez."
        )
    elif not has_score:
        justification = (
            f"{justification} [Justificación sintética: falta el campo score para '{key}'.]"
        )

    return JudgeScore(score=score, justification=justification)


def _error_result(raw_text: str, error_message: str) -> JudgeResult:
    """Build an all-zero JudgeResult with parse_error set, for total parse failure."""
    zero = JudgeScore(score=0.0, justification=f"No se pudo parsear la respuesta del juez: {error_message}")
    return JudgeResult(
        faithfulness=zero,
        answer_relevancy=zero,
        hallucination_rate=zero,
        raw_response=raw_text,
        parse_error=error_message,
    )


def parse_judge_response(raw_text: str) -> JudgeResult:
    """Defensively parse the judge's raw text response into a JudgeResult.

    Strips markdown code fences, attempts `json.loads`; on failure,
    regex-extracts the first `{...}` block and retries. On total failure,
    returns a JudgeResult with all scores 0.0 and `parse_error` set. Never
    raises an exception. Out-of-range scores are clamped to [0, 1]. Missing
    required fields get score=0.0 and a synthetic justification.

    Args:
        raw_text: Raw text response from the judge LLM.

    Returns:
        A JudgeResult, always — never raises.
    """
    if raw_text is None:
        raw_text = ""
    raw_text = str(raw_text)

    cleaned = _strip_code_fences(raw_text)

    data = None
    last_error = ""
    try:
        data = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError) as exc:
        last_error = str(exc)
        match = _JSON_BLOCK_RE.search(cleaned)
        if match:
            try:
                data = json.loads(match.group(0))
            except (json.JSONDecodeError, ValueError) as exc2:
                last_error = str(exc2)
                data = None
        else:
            data = None

    if not isinstance(data, dict):
        return _error_result(raw_text, last_error or "respuesta no es un objeto JSON")

    try:
        return JudgeResult(
            faithfulness=_build_score(data, "faithfulness"),
            answer_relevancy=_build_score(data, "answer_relevancy"),
            hallucination_rate=_build_score(data, "hallucination_rate"),
            raw_response=raw_text,
            parse_error=None,
        )
    except Exception as exc:  # defensive: never raise from this function
        return _error_result(raw_text, str(exc))

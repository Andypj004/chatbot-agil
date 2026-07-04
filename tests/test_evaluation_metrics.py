"""Unit tests for RAG evaluation metrics (retrieval metrics and LLM judge)."""

from langchain_core.documents import Document

from src.evaluation.dataset import RelevantChunkDescriptor
from src.evaluation.retrieval_metrics import (
    aggregate_mrr,
    context_precision,
    context_recall,
    is_relevant_document,
    mean_reciprocal_rank,
)
from src.evaluation.judge import (
    build_judge_prompt,
    invoke_judge,
    parse_judge_response,
)


class _DummyLLM:
    """Dummy LLM whose invoke() returns an object with a `.text` attribute."""

    def __init__(self, response_text):
        self._response_text = response_text
        self.last_prompt = None

    def invoke(self, prompt):
        self.last_prompt = prompt
        return _DummyLLMResponse(self._response_text)


class _DummyLLMResponse:
    def __init__(self, text):
        self.text = text


class _DummyProvider:
    """Dummy provider exposing get_llm(), mirroring _DummyProvider in test_rag_retriever.py."""

    model_name = "dummy-model"

    def __init__(self, llm):
        self._llm = llm

    def get_llm(self):
        return self._llm

    def get_provider_name(self):
        return "dummy"


class _PlainStringLLM:
    """Dummy LLM whose invoke() returns a plain string with no `.text` attribute."""

    def __init__(self, response_text):
        self._response_text = response_text
        self.last_prompt = None

    def invoke(self, prompt):
        self.last_prompt = prompt
        return self._response_text


def _descriptor(source_contains, anchor_contains):
    return RelevantChunkDescriptor(
        source_contains=source_contains, anchor_contains=anchor_contains
    )


def _doc(content, filename="scrum_guide.pdf"):
    return Document(page_content=content, metadata={"filename": filename})


# ---------------------------------------------------------------------------
# context_precision
# ---------------------------------------------------------------------------


def test_context_precision_empty_retrieved_list():
    descriptors = [_descriptor("scrum_guide.pdf", ["sprint"])]
    assert context_precision([], descriptors) == 0.0


def test_context_precision_no_relevant_docs():
    descriptors = [_descriptor("scrum_guide.pdf", ["sprint planning"])]
    docs = [
        _doc("Esto no menciona nada relevante.", filename="other.pdf"),
        _doc("Tampoco esto.", filename="other2.pdf"),
    ]
    assert context_precision(docs, descriptors) == 0.0


def test_context_precision_all_relevant():
    descriptors = [_descriptor("scrum_guide.pdf", ["sprint"])]
    docs = [
        _doc("El Sprint es un bloque de tiempo."),
        _doc("Durante el sprint se desarrolla el incremento."),
    ]
    assert context_precision(docs, descriptors) == 1.0


def test_context_precision_partial_overlap():
    descriptors = [_descriptor("scrum_guide.pdf", ["sprint"])]
    docs = [
        _doc("El Sprint es un bloque de tiempo."),
        _doc("Esto es irrelevante.", filename="other.pdf"),
    ]
    assert context_precision(docs, descriptors) == 0.5


# ---------------------------------------------------------------------------
# context_recall
# ---------------------------------------------------------------------------


def test_context_recall_empty_ground_truth_is_one():
    docs = [_doc("contenido cualquiera")]
    assert context_recall(docs, []) == 1.0


def test_context_recall_all_descriptors_found():
    descriptors = [
        _descriptor("scrum_guide.pdf", ["sprint"]),
        _descriptor("kanban.pdf", ["tablero"]),
    ]
    docs = [
        _doc("El Sprint es clave.", filename="scrum_guide.pdf"),
        _doc("El tablero Kanban visualiza el flujo.", filename="kanban.pdf"),
    ]
    assert context_recall(docs, descriptors) == 1.0


def test_context_recall_none_found():
    descriptors = [
        _descriptor("scrum_guide.pdf", ["sprint"]),
        _descriptor("kanban.pdf", ["tablero"]),
    ]
    docs = [_doc("Contenido sin relacion.", filename="otro.pdf")]
    assert context_recall(docs, descriptors) == 0.0


def test_context_recall_partial():
    descriptors = [
        _descriptor("scrum_guide.pdf", ["sprint"]),
        _descriptor("kanban.pdf", ["tablero"]),
    ]
    docs = [_doc("El Sprint es clave.", filename="scrum_guide.pdf")]
    assert context_recall(docs, descriptors) == 0.5


# ---------------------------------------------------------------------------
# mean_reciprocal_rank
# ---------------------------------------------------------------------------


def test_mrr_first_doc_relevant():
    descriptors = [_descriptor("scrum_guide.pdf", ["sprint"])]
    docs = [
        _doc("El Sprint es clave.", filename="scrum_guide.pdf"),
        _doc("Irrelevante.", filename="other.pdf"),
    ]
    assert mean_reciprocal_rank(docs, descriptors) == 1.0


def test_mrr_later_rank_relevant():
    descriptors = [_descriptor("scrum_guide.pdf", ["sprint"])]
    docs = [
        _doc("Irrelevante 1.", filename="other.pdf"),
        _doc("Irrelevante 2.", filename="other2.pdf"),
        _doc("El Sprint es clave.", filename="scrum_guide.pdf"),
    ]
    assert mean_reciprocal_rank(docs, descriptors) == 1.0 / 3.0


def test_mrr_none_relevant():
    descriptors = [_descriptor("scrum_guide.pdf", ["sprint"])]
    docs = [
        _doc("Irrelevante 1.", filename="other.pdf"),
        _doc("Irrelevante 2.", filename="other2.pdf"),
    ]
    assert mean_reciprocal_rank(docs, descriptors) == 0.0


def test_mrr_empty_list():
    descriptors = [_descriptor("scrum_guide.pdf", ["sprint"])]
    assert mean_reciprocal_rank([], descriptors) == 0.0


# ---------------------------------------------------------------------------
# aggregate_mrr
# ---------------------------------------------------------------------------


def test_aggregate_mrr_averages_multiple_values():
    assert aggregate_mrr([1.0, 0.5, 0.0]) == 0.5


def test_aggregate_mrr_empty_list_is_zero():
    assert aggregate_mrr([]) == 0.0


# ---------------------------------------------------------------------------
# is_relevant_document
# ---------------------------------------------------------------------------


def test_is_relevant_document_case_insensitive_anchor_match():
    descriptor = _descriptor("scrum_guide.pdf", ["sprint"])
    doc_upper = _doc("El SPRINT es un bloque de tiempo.")
    doc_title = _doc("El Sprint es un bloque de tiempo.")
    assert is_relevant_document(doc_upper, [descriptor]) is True
    assert is_relevant_document(doc_title, [descriptor]) is True


def test_is_relevant_document_accent_insensitive_anchor_match():
    descriptor = _descriptor("scrum_guide.pdf", ["planificacion"])
    doc_accented = _doc("La planificación del sprint ocurre al inicio.")
    assert is_relevant_document(doc_accented, [descriptor]) is True

    descriptor_accented = _descriptor("scrum_guide.pdf", ["planificación"])
    doc_unaccented = _doc("La planificacion del sprint ocurre al inicio.")
    assert is_relevant_document(doc_unaccented, [descriptor_accented]) is True


def test_is_relevant_document_requires_all_anchors():
    descriptor = _descriptor("scrum_guide.pdf", ["sprint", "retrospectiva"])
    doc_only_one_anchor = _doc("El Sprint es un bloque de tiempo.")
    doc_both_anchors = _doc(
        "El Sprint termina con una retrospectiva para mejorar el proceso."
    )
    assert is_relevant_document(doc_only_one_anchor, [descriptor]) is False
    assert is_relevant_document(doc_both_anchors, [descriptor]) is True


def test_is_relevant_document_or_semantics_across_descriptors():
    descriptors = [
        _descriptor("scrum_guide.pdf", ["sprint"]),
        _descriptor("kanban.pdf", ["tablero"]),
    ]
    # Matches only the second descriptor.
    doc = _doc("El tablero Kanban visualiza el flujo.", filename="kanban.pdf")
    assert is_relevant_document(doc, descriptors) is True


def test_is_relevant_document_no_descriptors_is_false():
    doc = _doc("contenido cualquiera")
    assert is_relevant_document(doc, []) is False


def test_is_relevant_document_source_mismatch_is_false():
    descriptor = _descriptor("scrum_guide.pdf", ["sprint"])
    doc = _doc("El Sprint es clave.", filename="otro_documento.pdf")
    assert is_relevant_document(doc, [descriptor]) is False


# ---------------------------------------------------------------------------
# parse_judge_response
# ---------------------------------------------------------------------------


def _valid_judge_json():
    return (
        '{"faithfulness": {"score": 0.9, "justification": "bien fundamentado"}, '
        '"answer_relevancy": {"score": 0.8, "justification": "responde la pregunta"}, '
        '"hallucination_rate": {"score": 0.1, "justification": "pocas invenciones"}}'
    )


def test_parse_judge_response_valid_json():
    result = parse_judge_response(_valid_judge_json())
    assert result.parse_error is None
    assert result.faithfulness.score == 0.9
    assert result.faithfulness.justification == "bien fundamentado"
    assert result.answer_relevancy.score == 0.8
    assert result.hallucination_rate.score == 0.1
    assert result.raw_response == _valid_judge_json()


def test_parse_judge_response_wrapped_in_code_fences():
    wrapped = f"```json\n{_valid_judge_json()}\n```"
    result = parse_judge_response(wrapped)
    assert result.parse_error is None
    assert result.faithfulness.score == 0.9
    assert result.answer_relevancy.score == 0.8
    assert result.hallucination_rate.score == 0.1


def test_parse_judge_response_malformed_json_sets_parse_error():
    malformed = '{"faithfulness": {"score": 0.9, "justification": "trunc'
    result = parse_judge_response(malformed)
    assert result.parse_error is not None
    assert result.faithfulness.score == 0.0
    assert result.answer_relevancy.score == 0.0
    assert result.hallucination_rate.score == 0.0
    assert result.raw_response == malformed


def test_parse_judge_response_missing_field_gets_synthetic_default():
    missing_field_json = (
        '{"faithfulness": {"score": 0.9, "justification": "bien fundamentado"}, '
        '"answer_relevancy": {"score": 0.8, "justification": "responde la pregunta"}}'
    )
    result = parse_judge_response(missing_field_json)
    assert result.parse_error is None
    # Present fields parse normally.
    assert result.faithfulness.score == 0.9
    assert result.answer_relevancy.score == 0.8
    # Missing field gets a synthetic default.
    assert result.hallucination_rate.score == 0.0
    assert "ausente" in result.hallucination_rate.justification.lower() or \
        "hallucination_rate" in result.hallucination_rate.justification


def test_parse_judge_response_score_out_of_range_clamped_high():
    out_of_range_json = (
        '{"faithfulness": {"score": 1.7, "justification": "excelente"}, '
        '"answer_relevancy": {"score": 0.5, "justification": "ok"}, '
        '"hallucination_rate": {"score": 0.1, "justification": "ok"}}'
    )
    result = parse_judge_response(out_of_range_json)
    assert result.parse_error is None
    assert result.faithfulness.score == 1.0


def test_parse_judge_response_score_out_of_range_clamped_low():
    out_of_range_json = (
        '{"faithfulness": {"score": -0.3, "justification": "malo"}, '
        '"answer_relevancy": {"score": 0.5, "justification": "ok"}, '
        '"hallucination_rate": {"score": 0.1, "justification": "ok"}}'
    )
    result = parse_judge_response(out_of_range_json)
    assert result.parse_error is None
    assert result.faithfulness.score == 0.0


def test_parse_judge_response_non_numeric_score_fails_only_that_field():
    non_numeric_json = (
        '{"faithfulness": {"score": "alta", "justification": "bien fundamentado"}, '
        '"answer_relevancy": {"score": 0.8, "justification": "responde la pregunta"}, '
        '"hallucination_rate": {"score": 0.1, "justification": "ok"}}'
    )
    result = parse_judge_response(non_numeric_json)
    assert result.parse_error is None
    # Non-numeric score for faithfulness fails gracefully to 0.0...
    assert result.faithfulness.score == 0.0
    # ...but other fields parse normally.
    assert result.answer_relevancy.score == 0.8
    assert result.hallucination_rate.score == 0.1


def test_parse_judge_response_prose_around_json_recovered_by_regex():
    prose_wrapped = (
        "Aqui esta mi evaluacion detallada del sistema RAG.\n\n"
        f"{_valid_judge_json()}\n\n"
        "Espero que esta evaluacion sea de utilidad."
    )
    result = parse_judge_response(prose_wrapped)
    assert result.parse_error is None
    assert result.faithfulness.score == 0.9
    assert result.answer_relevancy.score == 0.8
    assert result.hallucination_rate.score == 0.1


def test_parse_judge_response_empty_string_sets_parse_error():
    result = parse_judge_response("")
    assert result.parse_error is not None
    assert result.faithfulness.score == 0.0
    assert result.answer_relevancy.score == 0.0
    assert result.hallucination_rate.score == 0.0


# ---------------------------------------------------------------------------
# build_judge_prompt
# ---------------------------------------------------------------------------


def test_build_judge_prompt_includes_all_four_inputs():
    prompt = build_judge_prompt(
        question="Que es un sprint?",
        context="Un sprint es un bloque de tiempo fijo en Scrum.",
        generated_answer="Un sprint es un periodo de tiempo limitado.",
        reference_answer="Un sprint es un periodo de tiempo fijo usado en Scrum.",
    )
    assert "Que es un sprint?" in prompt
    assert "Un sprint es un bloque de tiempo fijo en Scrum." in prompt
    assert "Un sprint es un periodo de tiempo limitado." in prompt
    assert "Un sprint es un periodo de tiempo fijo usado en Scrum." in prompt


# ---------------------------------------------------------------------------
# invoke_judge
# ---------------------------------------------------------------------------


def test_invoke_judge_unwraps_text_attribute():
    dummy_llm = _DummyLLM("respuesta del juez")
    provider = _DummyProvider(dummy_llm)

    result = invoke_judge("mi prompt", llm_provider=provider)

    assert dummy_llm.last_prompt == "mi prompt"
    assert result == "respuesta del juez"


def test_invoke_judge_returns_plain_string_as_is():
    plain_llm = _PlainStringLLM("respuesta en texto plano")
    provider = _DummyProvider(plain_llm)

    result = invoke_judge("mi prompt", llm_provider=provider)

    assert plain_llm.last_prompt == "mi prompt"
    assert result == "respuesta en texto plano"

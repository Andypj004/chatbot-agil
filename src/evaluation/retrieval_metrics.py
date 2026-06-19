"""Metrics for evaluating RAG retrieval quality."""

import re
import unicodedata
from typing import Any, List, TYPE_CHECKING

from src.evaluation.dataset import RelevantChunkDescriptor

if TYPE_CHECKING:
    from langchain_core.documents import Document
else:
    Document = Any


def normalize_text(text: str) -> str:
    """Normalize text for comparison.

    Applies NFKD normalization, lowercasing, accent removal via
    combining character stripping, and whitespace collapsing.

    Args:
        text: Text to normalize.

    Returns:
        Normalized text.
    """
    normalized = unicodedata.normalize("NFKD", text.lower().strip())
    normalized = re.sub(r"[̀-ͯ]", "", normalized)
    normalized = re.sub(r"[^a-z0-9\s]+", " ", normalized)
    return " ".join(normalized.split())


def is_relevant_document(
    doc: Document,
    descriptors: List[RelevantChunkDescriptor]
) -> bool:
    """Check if a document is relevant to any descriptor.

    A document is relevant if ANY descriptor matches:
    - The normalized source_contains is a substring of the
      normalized filename (or source metadata if filename absent), AND
    - All normalized anchor_contains strings are substrings of the
      normalized page_content.

    Args:
        doc: LangChain Document to check.
        descriptors: List of relevance descriptors.

    Returns:
        True if the document matches at least one descriptor.
    """
    if not descriptors:
        return False

    metadata = doc.metadata or {}
    source_text = metadata.get("filename") or metadata.get("source") or ""
    source_normalized = normalize_text(source_text)
    content_normalized = normalize_text(doc.page_content or "")

    for descriptor in descriptors:
        source_contains_norm = normalize_text(descriptor.source_contains)

        # Check if source_contains is a substring of normalized source
        if source_contains_norm not in source_normalized:
            continue

        # Check if all anchor_contains are substrings of normalized content
        all_anchors_match = all(
            normalize_text(anchor) in content_normalized
            for anchor in descriptor.anchor_contains
        )

        if all_anchors_match:
            return True

    return False


def context_precision(
    retrieved_docs: List[Document],
    descriptors: List[RelevantChunkDescriptor]
) -> float:
    """Compute context precision: fraction of retrieved docs that are relevant.

    Args:
        retrieved_docs: List of retrieved documents.
        descriptors: List of relevance descriptors.

    Returns:
        Fraction of relevant documents, or 0.0 if retrieved_docs is empty.
    """
    if not retrieved_docs:
        return 0.0

    relevant_count = sum(
        1 for doc in retrieved_docs
        if is_relevant_document(doc, descriptors)
    )
    return relevant_count / len(retrieved_docs)


def context_recall(
    retrieved_docs: List[Document],
    descriptors: List[RelevantChunkDescriptor]
) -> float:
    """Compute context recall: fraction of descriptors matched by at least one doc.

    Args:
        retrieved_docs: List of retrieved documents.
        descriptors: List of relevance descriptors.

    Returns:
        Fraction of descriptors covered, or 1.0 if descriptors is empty
        (vacuously true).
    """
    if not descriptors:
        return 1.0

    # Track which descriptors are covered
    covered_descriptors = set()
    for doc in retrieved_docs:
        for i, descriptor in enumerate(descriptors):
            metadata = doc.metadata or {}
            source_text = metadata.get("filename") or metadata.get("source") or ""
            source_normalized = normalize_text(source_text)
            content_normalized = normalize_text(doc.page_content or "")

            source_contains_norm = normalize_text(descriptor.source_contains)

            # Check if source_contains is a substring
            if source_contains_norm not in source_normalized:
                continue

            # Check if all anchor_contains are substrings
            if all(
                normalize_text(anchor) in content_normalized
                for anchor in descriptor.anchor_contains
            ):
                covered_descriptors.add(i)

    return len(covered_descriptors) / len(descriptors)


def mean_reciprocal_rank(
    retrieved_docs: List[Document],
    descriptors: List[RelevantChunkDescriptor]
) -> float:
    """Compute mean reciprocal rank of the first relevant document.

    Args:
        retrieved_docs: List of retrieved documents.
        descriptors: List of relevance descriptors.

    Returns:
        1/(1-based rank of first relevant doc), or 0.0 if no relevant
        document found or list is empty.
    """
    if not retrieved_docs:
        return 0.0

    for i, doc in enumerate(retrieved_docs):
        if is_relevant_document(doc, descriptors):
            return 1.0 / (i + 1)

    return 0.0


def aggregate_mrr(per_question_mrr: List[float]) -> float:
    """Aggregate mean reciprocal rank scores by simple arithmetic mean.

    Args:
        per_question_mrr: List of MRR scores, one per question.

    Returns:
        Arithmetic mean of the scores.
    """
    if not per_question_mrr:
        return 0.0
    return sum(per_question_mrr) / len(per_question_mrr)

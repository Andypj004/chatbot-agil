"""Dataset loading and dataclass definitions for RAG evaluation."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union


@dataclass(frozen=True)
class RelevantChunkDescriptor:
    """Descriptor of a relevant chunk for evaluation."""
    source_contains: str
    anchor_contains: List[str]


@dataclass(frozen=True)
class EvalQuestion:
    """Evaluation question with ground truth metadata."""
    id: str
    subtopic: str
    question: str
    reference_answer: str
    relevant_chunks: List[RelevantChunkDescriptor]


def load_eval_dataset(path: Union[str, Path]) -> List[EvalQuestion]:
    """Load evaluation dataset from JSON file.

    Args:
        path: Path to the evaluation dataset JSON file.

    Returns:
        List of EvalQuestion instances parsed from the JSON.
    """
    path = Path(path) if isinstance(path, str) else path

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    questions = []
    for q_data in data.get('questions', []):
        relevant_chunks = [
            RelevantChunkDescriptor(
                source_contains=chunk['source_contains'],
                anchor_contains=chunk['anchor_contains']
            )
            for chunk in q_data.get('relevant_chunks', [])
        ]

        question = EvalQuestion(
            id=q_data['id'],
            subtopic=q_data['subtopic'],
            question=q_data['question'],
            reference_answer=q_data['reference_answer'],
            relevant_chunks=relevant_chunks
        )
        questions.append(question)

    return questions

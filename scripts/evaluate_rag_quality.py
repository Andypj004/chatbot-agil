#!/usr/bin/env python3
"""Offline CLI to evaluate RAG pipeline quality against a ground-truth dataset.

Runs the REAL retrieval pipeline (ChromaDB) and, optionally, the REAL
generation + LLM-as-judge pipeline (real API calls). This is a manual,
cost-incurring reporting tool — it is NOT a pytest test and is not run as
part of the automated test suite.

Usage:
    python scripts/evaluate_rag_quality.py --mode both --k 8 \\
        --dataset tests/data/agile_rag_eval_dataset.json \\
        --output reports/rag_evaluation/
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean

# Allow running as `python scripts/evaluate_rag_quality.py` from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.config import settings
from src.evaluation.dataset import EvalQuestion, load_eval_dataset
from src.evaluation.judge import build_judge_prompt, invoke_judge, parse_judge_response
from src.evaluation.retrieval_metrics import (
    aggregate_mrr,
    context_precision,
    context_recall,
    mean_reciprocal_rank,
)
from src.llm.factory import LLMFactory
from src.rag.retriever import RAGRetriever
from src.rag.vector_store import VectorStore

RETRIEVAL_METRIC_KEYS = ("context_precision", "context_recall", "mrr")
JUDGE_METRIC_KEYS = ("faithfulness", "answer_relevancy", "hallucination_rate")


def parse_args(argv=None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate RAG pipeline quality (retrieval + generation) "
        "against a ground-truth dataset. Makes real ChromaDB queries and, "
        "in 'generation'/'both' modes, real LLM API calls."
    )
    parser.add_argument(
        "--mode",
        choices=["retrieval", "generation", "both"],
        default="both",
        help="Which evaluation phases to run (default: both).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Number of documents to retrieve per question "
        "(default: settings.top_k_results).",
    )
    parser.add_argument(
        "--dataset",
        default="tests/data/agile_rag_eval_dataset.json",
        help="Path to the evaluation dataset JSON file.",
    )
    parser.add_argument(
        "--output",
        default="reports/rag_evaluation/",
        help="Output directory for the generated report(s).",
    )
    parser.add_argument(
        "--no-markdown",
        action="store_true",
        help="Skip writing the human-readable Markdown report.",
    )
    parser.add_argument(
        "--question-id",
        default=None,
        help="Filter to a single question by its id (cheap iteration).",
    )
    return parser.parse_args(argv)


def select_questions(
    questions: list, question_id: str = None
) -> list:
    """Filter the loaded questions to a single id if requested."""
    if question_id is None:
        return questions
    return [q for q in questions if q.id == question_id]


def run_retrieval_phase(
    retriever: RAGRetriever, question: EvalQuestion, k: int
) -> dict:
    """Run the retrieval phase for one question: retrieve docs + compute metrics."""
    retrieved_docs = retriever.retrieve_documents(query=question.question, k=k)
    descriptors = question.relevant_chunks

    precision = context_precision(retrieved_docs, descriptors)
    recall = context_recall(retrieved_docs, descriptors)
    mrr = mean_reciprocal_rank(retrieved_docs, descriptors)

    return {
        "context_precision": precision,
        "context_recall": recall,
        "mrr": mrr,
        "num_retrieved": len(retrieved_docs),
        "retrieved_sources": [
            (doc.metadata or {}).get("filename")
            or (doc.metadata or {}).get("source")
            or "unknown"
            for doc in retrieved_docs
        ],
    }


def run_generation_phase(
    retriever: RAGRetriever, question: EvalQuestion, k: int
) -> dict:
    """Run the generation phase for one question: answer + judge evaluation."""
    query_result = retriever.query(question.question, k=k, return_sources=True)
    answer = query_result.get("answer", "")
    sources = query_result.get("sources") or []
    context = "\n\n".join(source.get("content", "") for source in sources)

    judge_prompt = build_judge_prompt(
        question=question.question,
        context=context,
        generated_answer=answer,
        reference_answer=question.reference_answer,
    )
    raw_judge_response = invoke_judge(judge_prompt)
    judge_result = parse_judge_response(raw_judge_response)

    return {
        "answer": answer,
        "num_sources": query_result.get("num_sources"),
        "faithfulness": judge_result.faithfulness.score,
        "faithfulness_justification": judge_result.faithfulness.justification,
        "answer_relevancy": judge_result.answer_relevancy.score,
        "answer_relevancy_justification": judge_result.answer_relevancy.justification,
        "hallucination_rate": judge_result.hallucination_rate.score,
        "hallucination_rate_justification": judge_result.hallucination_rate.justification,
        "parse_error": judge_result.parse_error,
    }


def aggregate_metrics(per_question_results: list, metric_keys: tuple) -> dict:
    """Compute overall and per-subtopic mean for each metric key.

    Returns a dict: {"overall": {metric: mean}, "by_subtopic": {subtopic: {metric: mean}}}
    """
    overall = {}
    for key in metric_keys:
        values = [
            r["metrics"][key]
            for r in per_question_results
            if key in r["metrics"] and r["metrics"][key] is not None
        ]
        overall[key] = mean(values) if values else 0.0

    by_subtopic_values = defaultdict(lambda: defaultdict(list))
    for r in per_question_results:
        subtopic = r["subtopic"]
        for key in metric_keys:
            value = r["metrics"].get(key)
            if value is not None:
                by_subtopic_values[subtopic][key].append(value)

    by_subtopic = {}
    for subtopic, metrics in by_subtopic_values.items():
        by_subtopic[subtopic] = {
            key: (mean(values) if values else 0.0)
            for key, values in metrics.items()
        }

    return {"overall": overall, "by_subtopic": by_subtopic}


def print_console_summary(
    mode: str,
    num_questions: int,
    retrieval_agg: dict,
    generation_agg: dict,
    parse_error_count: int,
) -> None:
    """Print a human-readable summary table to the console."""
    print()
    print("=" * 70)
    print("RAG QUALITY EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Mode: {mode}    Questions evaluated: {num_questions}")
    print()

    if retrieval_agg is not None:
        print("-- Retrieval metrics (overall) --")
        for key, value in retrieval_agg["overall"].items():
            print(f"  {key:20s}: {value:.4f}")
        print()
        print("-- Retrieval metrics (by subtopic) --")
        for subtopic, metrics in sorted(retrieval_agg["by_subtopic"].items()):
            metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            print(f"  {subtopic:20s}: {metrics_str}")
        print()

    if generation_agg is not None:
        print("-- Generation / judge metrics (overall) --")
        for key, value in generation_agg["overall"].items():
            print(f"  {key:20s}: {value:.4f}")
        print()
        print("-- Generation / judge metrics (by subtopic) --")
        for subtopic, metrics in sorted(generation_agg["by_subtopic"].items()):
            metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            print(f"  {subtopic:20s}: {metrics_str}")
        print()
        print(f"Judge parse errors: {parse_error_count}")
        print()

    print("=" * 70)


def build_markdown_report(report: dict) -> str:
    """Build a human-readable Markdown report from the structured report dict."""
    lines = []
    lines.append("# RAG Quality Evaluation Report")
    lines.append("")
    lines.append(f"- **Timestamp**: {report['timestamp']}")
    lines.append(f"- **Mode**: {report['mode']}")
    lines.append(f"- **k**: {report['k']}")
    lines.append(f"- **Dataset**: {report['dataset_path']}")
    lines.append(f"- **Questions evaluated**: {report['num_questions']}")
    if report.get("retrieval") is not None:
        lines.append(f"- **Judge parse errors**: {report['judge_parse_error_count']}")
    lines.append("")

    if report.get("retrieval") is not None:
        retrieval_agg = report["retrieval"]["aggregate"]
        lines.append("## Retrieval Metrics")
        lines.append("")
        lines.append("### Overall")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        for key, value in retrieval_agg["overall"].items():
            lines.append(f"| {key} | {value:.4f} |")
        lines.append("")
        lines.append("### By Subtopic")
        lines.append("")
        subtopic_keys = sorted(retrieval_agg["by_subtopic"].keys())
        if subtopic_keys:
            header_metrics = list(RETRIEVAL_METRIC_KEYS)
            lines.append("| Subtopic | " + " | ".join(header_metrics) + " |")
            lines.append("|---|" + "---|" * len(header_metrics))
            for subtopic in subtopic_keys:
                metrics = retrieval_agg["by_subtopic"][subtopic]
                row = [f"{metrics.get(k, 0.0):.4f}" for k in header_metrics]
                lines.append(f"| {subtopic} | " + " | ".join(row) + " |")
        lines.append("")

    if report.get("generation") is not None:
        generation_agg = report["generation"]["aggregate"]
        lines.append("## Generation / Judge Metrics")
        lines.append("")
        lines.append("### Overall")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        for key, value in generation_agg["overall"].items():
            lines.append(f"| {key} | {value:.4f} |")
        lines.append("")
        lines.append("### By Subtopic")
        lines.append("")
        subtopic_keys = sorted(generation_agg["by_subtopic"].keys())
        if subtopic_keys:
            header_metrics = list(JUDGE_METRIC_KEYS)
            lines.append("| Subtopic | " + " | ".join(header_metrics) + " |")
            lines.append("|---|" + "---|" * len(header_metrics))
            for subtopic in subtopic_keys:
                metrics = generation_agg["by_subtopic"][subtopic]
                row = [f"{metrics.get(k, 0.0):.4f}" for k in header_metrics]
                lines.append(f"| {subtopic} | " + " | ".join(row) + " |")
        lines.append("")

    lines.append("## Per-Question Detail")
    lines.append("")
    for q in report["questions"]:
        lines.append(f"### {q['id']} ({q['subtopic']})")
        lines.append("")
        lines.append(f"**Question**: {q['question']}")
        lines.append("")
        if "retrieval" in q:
            r = q["retrieval"]
            lines.append(
                f"- Retrieval: context_precision={r['context_precision']:.4f}, "
                f"context_recall={r['context_recall']:.4f}, mrr={r['mrr']:.4f}, "
                f"num_retrieved={r['num_retrieved']}"
            )
        if "generation" in q:
            g = q["generation"]
            lines.append(
                f"- Generation: faithfulness={g['faithfulness']:.4f}, "
                f"answer_relevancy={g['answer_relevancy']:.4f}, "
                f"hallucination_rate={g['hallucination_rate']:.4f}"
                + (f", parse_error={g['parse_error']}" if g.get("parse_error") else "")
            )
            lines.append("")
            lines.append(f"**Generated answer**: {g['answer']}")
        lines.append("")

    return "\n".join(lines)


def main(argv=None) -> int:
    args = parse_args(argv)

    try:
        questions = load_eval_dataset(args.dataset)
    except Exception as exc:
        print(f"ERROR: failed to load dataset '{args.dataset}': {exc}", file=sys.stderr)
        return 1

    questions = select_questions(questions, args.question_id)

    if not questions:
        print(
            f"ERROR: no questions to evaluate (dataset='{args.dataset}', "
            f"question_id='{args.question_id}')",
            file=sys.stderr,
        )
        return 1

    k = args.k or settings.top_k_results

    print(f"Loaded {len(questions)} question(s) from {args.dataset}")
    print(f"Mode: {args.mode}    k: {k}")
    print("Initializing vector store...")
    vector_store = VectorStore()
    print("Initializing LLM provider...")
    llm_provider = LLMFactory.create_provider()
    retriever = RAGRetriever(vector_store, llm_provider, top_k=k)

    do_retrieval = args.mode in ("retrieval", "both")
    do_generation = args.mode in ("generation", "both")

    per_question_reports = []
    parse_error_count = 0

    for question in questions:
        print(f"Evaluating question '{question.id}' ({question.subtopic})...")
        q_report = {
            "id": question.id,
            "subtopic": question.subtopic,
            "question": question.question,
            "reference_answer": question.reference_answer,
        }

        if do_retrieval:
            retrieval_result = run_retrieval_phase(retriever, question, k)
            q_report["retrieval"] = retrieval_result

        if do_generation:
            generation_result = run_generation_phase(retriever, question, k)
            q_report["generation"] = generation_result
            if generation_result.get("parse_error"):
                parse_error_count += 1

        per_question_reports.append(q_report)

    retrieval_block = None
    if do_retrieval:
        retrieval_metric_results = [
            {"subtopic": q["subtopic"], "metrics": q["retrieval"]}
            for q in per_question_reports
        ]
        retrieval_agg = aggregate_metrics(retrieval_metric_results, RETRIEVAL_METRIC_KEYS)
        # Use aggregate_mrr explicitly for the overall MRR aggregation, per spec.
        retrieval_agg["overall"]["mrr"] = aggregate_mrr(
            [q["retrieval"]["mrr"] for q in per_question_reports]
        )
        retrieval_block = {
            "aggregate": retrieval_agg,
        }

    generation_block = None
    if do_generation:
        generation_metric_results = [
            {"subtopic": q["subtopic"], "metrics": q["generation"]}
            for q in per_question_reports
        ]
        generation_agg = aggregate_metrics(generation_metric_results, JUDGE_METRIC_KEYS)
        generation_block = {
            "aggregate": generation_agg,
        }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    report = {
        "timestamp": timestamp,
        "mode": args.mode,
        "k": k,
        "dataset_path": str(args.dataset),
        "question_id_filter": args.question_id,
        "num_questions": len(per_question_reports),
        "judge_parse_error_count": parse_error_count,
        "retrieval": retrieval_block,
        "generation": generation_block,
        "questions": per_question_reports,
    }

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"eval_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nWrote JSON report to {json_path}")

    if not args.no_markdown:
        markdown_path = output_dir / f"eval_{timestamp}.md"
        markdown_content = build_markdown_report(report)
        with open(markdown_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        print(f"Wrote Markdown report to {markdown_path}")

    print_console_summary(
        mode=args.mode,
        num_questions=len(per_question_reports),
        retrieval_agg=retrieval_block["aggregate"] if retrieval_block else None,
        generation_agg=generation_block["aggregate"] if generation_block else None,
        parse_error_count=parse_error_count,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())

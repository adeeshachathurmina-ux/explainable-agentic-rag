"""
Automated Empirical Benchmark Runner for Deliverable D1 (Section IV) & D2.

Executes the 10 Ground-Truth queries from `data/ground_truth.json` against the
Explainable Agentic RAG System, computes Context Relevance, Groundedness
(Faithfulness), and Answer Relevance, and exports both a JSON audit report and
a formatted Markdown evaluation table.
"""

import json
from pathlib import Path
from typing import List

from src.ingestion import PDFDocumentProcessor
from src.math_retriever import HybridMathRetriever
from src.agent import AcademicPolicyAgent
from src.evaluator import RAGEvaluator
from src.interfaces import EvaluationMetrics


def run_benchmark(
    ground_truth_path: str = "data/ground_truth.json",
    output_json_path: str = "data/evaluation_results.json",
    output_md_path: str = "data/evaluation_table.md",
) -> None:
    print("\n" + "=" * 85)
    print("PHASE 4: EMPIRICAL EVALUATION BENCHMARK (10 GROUND-TRUTH QUERIES)")
    print("=" * 85)

    gt_file = Path(ground_truth_path)
    if not gt_file.exists():
        raise FileNotFoundError(f"Ground-truth dataset not found at: {ground_truth_path}")

    with open(gt_file, "r", encoding="utf-8") as f:
        ground_truth_samples = json.load(f)

    # Initialize pipeline components
    processor = PDFDocumentProcessor(chunk_size=180, chunk_overlap=40)
    chunks = processor.ingest_directory("data/pdfs")

    retriever = HybridMathRetriever(
        persist_directory="data/chroma_db",
        collection_name="academic_policy_store",
        alpha=0.65,
    )
    retriever.index_chunks(chunks)

    agent = AcademicPolicyAgent(retriever=retriever, max_steps=3, top_k_per_step=2)
    evaluator = RAGEvaluator()

    all_metrics: List[EvaluationMetrics] = []
    md_rows: List[str] = [
        "| ID | Ground-Truth Query | Context Relevance | Groundedness (Faithfulness) | Answer Relevance |",
        "| :--- | :--- | :---: | :---: | :---: |",
    ]

    print(
        f"{'ID':<7} | {'Context Rel.':<14} | {'Groundedness':<14} | {'Answer Rel.':<14} | Query"
    )
    print("-" * 85)

    for item in ground_truth_samples:
        qid = item["id"]
        question = item["question"]
        gt_answer = item["ground_truth_answer"]
        expected_kw = item.get("expected_context_keywords", [])

        state = agent.run(question)
        metrics = evaluator.evaluate_sample(
            query=question,
            ground_truth=gt_answer,
            agent_state=state,
            expected_keywords=expected_kw,
        )
        all_metrics.append(metrics)

        print(
            f"{qid:<7} | {metrics.context_relevance:<14.4f} | "
            f"{metrics.groundedness:<14.4f} | {metrics.answer_relevance:<14.4f} | "
            f"{question[:42]}..."
        )

        short_q = question.replace("|", "/")
        md_rows.append(
            f"| {qid} | {short_q} | {metrics.context_relevance:.4f} | "
            f"{metrics.groundedness:.4f} | {metrics.answer_relevance:.4f} |"
        )

    mean_ctx = sum(m.context_relevance for m in all_metrics) / len(all_metrics)
    mean_grd = sum(m.groundedness for m in all_metrics) / len(all_metrics)
    mean_ans = sum(m.answer_relevance for m in all_metrics) / len(all_metrics)

    print("-" * 85)
    print(
        f"{'MEAN':<7} | {mean_ctx:<14.4f} | {mean_grd:<14.4f} | {mean_ans:<14.4f} | "
        f"Across {len(all_metrics)} Ground-Truth Queries"
    )
    print("=" * 85 + "\n")

    md_rows.append(
        f"| **Mean** | **Aggregate System Performance (10 Queries)** | "
        f"**{mean_ctx:.4f}** | **{mean_grd:.4f}** | **{mean_ans:.4f}** |"
    )

    # Save JSON and Markdown artifacts
    summary_payload = {
        "aggregate_metrics": {
            "mean_context_relevance": round(mean_ctx, 4),
            "mean_groundedness": round(mean_grd, 4),
            "mean_answer_relevance": round(mean_ans, 4),
            "total_queries": len(all_metrics),
        },
        "samples": [m.to_dict() for m in all_metrics],
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write("# Empirical Evaluation Results (RAG Triad Metrics)\n\n")
        f.write("\n".join(md_rows) + "\n")

    print(f"[Exported] Full JSON evaluation audit saved to: {output_json_path}")
    print(f"[Exported] Markdown table for Technical Report saved to: {output_md_path}\n")


if __name__ == "__main__":
    run_benchmark()
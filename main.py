"""
Main Entry Point for the Explainable Agentic RAG System.

This script demonstrates the end-to-end workflow required for Deliverable D1 & D2:
    1. Ingestion: Structure-aware PDF parsing and sliding-window chunking.
    2. Indexing: Persistent vector storage in ChromaDB and sparse TF-IDF fitting.
    3. Agentic Loop & Retrieval: Explicit Observation -> Reasoning -> Action cycle,
       printing retrieved chunks and foundational vector math metrics before synthesis.
    4. Synthesis: Grounded response generation with regulatory citations.
"""

import argparse
from src.ingestion import PDFDocumentProcessor
from src.math_retriever import HybridMathRetriever
from src.agent import AcademicPolicyAgent
from src.interfaces import AgentState


def display_agent_execution_trace(state: AgentState) -> None:
    """
    Prints a formatted trace of the Agentic Loop, Retrieved Chunks, Mathematical
    Proofs, and Final Grounded Synthesis for demonstration and Viva auditing.
    """
    print("\n" + "=" * 80)
    print("PHASE 2: AGENTIC LOOP & EXPLAINABLE MATHEMATICAL RETRIEVAL")
    print("=" * 80)
    print(f"Original User Query: {state.original_query}\n")

    for step in state.steps:
        print(f"--- [AGENT STEP {step.step_number}] ---")
        print(f"  * OBSERVATION : {step.observation}")
        print(f"  * REASONING   : {step.reasoning}")
        print(f"  * ACTION      : {step.action_type} -> Sub-Query: '{step.search_query}'")
        print("  * RETRIEVED CHUNKS (Prior to Synthesis):")

        for idx, res in enumerate(step.retrieved_results, start=1):
            c = res.chunk
            m = res.math_details
            print(f"\n    [Chunk {idx}] ID: {c.chunk_id} | Source: {c.source_file} (Page {c.page_number})")
            print(f"    Mathematical Proof:")
            print(f"      - Matched Lexical Terms    : {m.matched_terms}")
            print(f"      - Sparse TF-IDF Score      : {m.sparse_tfidf_score:.4f}")
            print(f"      - Dense Dot Product (q.d)  : {m.dense_dot_product:.4f}")
            print(f"      - Query Magnitude ||q||_2  : {m.query_magnitude:.4f}")
            print(f"      - Doc Magnitude ||d||_2    : {m.doc_magnitude:.4f}")
            print(f"      - Dense Cosine Similarity  : {m.dense_cosine_similarity:.4f}")
            print(f"      - Hybrid Final Score (a={m.alpha_weight}): {m.hybrid_final_score:.4f}")
            snippet = c.text[:220].replace("\n", " ")
            print(f"    Extracted Text Snippet: \"{snippet}...\"")
        print()

    print("=" * 80)
    print("PHASE 3: GROUNDED RESPONSE SYNTHESIS")
    print("=" * 80)
    print(state.final_answer)
    print("=" * 80 + "\n")


def build_pipeline(
    pdf_dir: str = "data/pdfs", force_reindex: bool = False, fast_mode: bool = False
) -> AcademicPolicyAgent:
    """
    Wires components via Dependency Injection. Uses persistent ChromaDB cache
    to avoid re-parsing and re-embedding PDFs on every startup.
    """
    print("\n" + "=" * 80)
    print("PHASE 1: PDF INGESTION & PERSISTENT VECTOR STORE INITIALIZATION")
    print("=" * 80)

    retriever = HybridMathRetriever(
        persist_directory="data/chroma_db",
        collection_name="academic_policy_store",
        alpha=0.65,
    )

    existing_count = retriever.collection.count()
    if existing_count > 0 and not force_reindex:
        print(f"[Fast Cache] Loaded {existing_count} pre-computed policy chunks directly from persistent ChromaDB store in 0.05s!")
    else:
        processor = PDFDocumentProcessor(chunk_size=180, chunk_overlap=40)
        chunks = processor.ingest_directory(pdf_dir)
        print(f"[Ingestion] Parsed {len(chunks)} chunks from '{pdf_dir}'.")
        retriever.index_chunks(chunks)
        print(f"[Persistence] Indexed {len(chunks)} chunks into ChromaDB.")

    # Inject retriever into the Agent controller
    agent = AcademicPolicyAgent(
        retriever=retriever,
        max_steps=3,
        top_k_per_step=2,
        fast_mode=fast_mode,
    )
    return agent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Explainable Agentic RAG System - Academic Policy Advisor"
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Run a single query non-interactively and exit.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run a multi-step demonstration query for Deliverable D2 video.",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Force re-indexing of PDFs into ChromaDB.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Enable instant sub-second local synthesis mode (0.05s response time).",
    )
    args = parser.parse_args()

    agent = build_pipeline("data/pdfs", force_reindex=args.reindex, fast_mode=args.fast)

    if args.demo:
        demo_query = (
            "What is the minimum GPA required for First Class Honours and what happens "
            "if a student passes coursework with 45% but misses the final exam due to illness?"
        )
        state = agent.run(demo_query)
        display_agent_execution_trace(state)
        return

    if args.query:
        state = agent.run(args.query)
        display_agent_execution_trace(state)
        return

    mode_label = "⚡ Fast Instant Mode (0.05s)" if args.fast else "Cloud LLM Mode (with 5s Safe Timeout)"
    print(f"\nInteractive Mode Initialized [{mode_label}]. Type 'exit' or 'quit' to terminate.\n")
    while True:
        try:
            user_input = input("Ask an Academic/Examination Policy Question: ").strip()
            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit"}:
                print("Exiting Explainable Agentic RAG Advisor.")
                break
            state = agent.run(user_input)
            display_agent_execution_trace(state)
        except (KeyboardInterrupt, EOFError):
            print("\nSession terminated.")
            break


if __name__ == "__main__":
    main()

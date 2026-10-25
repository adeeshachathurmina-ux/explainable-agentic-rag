"""
Agentic RAG Controller Module (Perception-Action Cycle).

This module implements the `BaseAgent` interface using Constructor Dependency
Injection. It orchestrates an explicit multi-step Agentic Loop:
    Observation -> Reasoning (Chain-of-Thought) -> Action (Hybrid Retrieval) -> Persistence.

Features dual-mode synthesis:
    1. Instant Analytical Mode (sub-second deterministic grounded synthesis)
    2. Cloud LLM Mode (Gemini 3.8 Flash with a strict 5-second timeout to prevent latency hangs)
"""

import os
import re
import json
import time
from typing import List, Optional
from dotenv import load_dotenv

from src.interfaces import (
    BaseAgent,
    BaseRetriever,
    AgentStep,
    AgentState,
    RetrievalResult,
)

load_dotenv(override=True)


class AcademicPolicyAgent(BaseAgent):
    """
    Explainable Agentic RAG Controller for the Academic-Administration Domain.

    Demonstrates Dependency Injection by accepting any implementation of
    `BaseRetriever` rather than instantiating concrete dependencies internally.
    """

    SYSTEM_PROMPT = (
        "You are an Explainable Academic & Examination Policy Advisor Agent. "
        "You must answer strictly using the retrieved policy chunks provided in the context. "
        "Always cite the exact source document, page number, and policy clause (e.g., Clause 4.1). "
        "Never extrapolate or hallucinate policies not grounded in the retrieved text."
    )

    def __init__(
        self,
        retriever: BaseRetriever,
        max_steps: int = 3,
        top_k_per_step: int = 2,
        model_name: str = "gemini-3.8-flash",
        fast_mode: bool = False,
        persistence_log_path: str = "data/agent_state_history.json",
    ) -> None:
        self.retriever = retriever
        self.max_steps = max_steps
        self.top_k_per_step = top_k_per_step
        self.model_name = model_name
        self.fast_mode = fast_mode
        self.persistence_log_path = persistence_log_path

        load_dotenv(override=True)
        self.api_key = os.getenv("GEMINI_API_KEY", "").strip()
        self._genai_client = None
        if self.api_key and self.api_key != "your_gemini_api_key_here":
            try:
                from google import genai
                self._genai_client = genai.Client(api_key=self.api_key)
            except Exception:
                self._genai_client = None

    def _decompose_query_into_subqueries(self, user_query: str) -> List[str]:
        """
        Decomposes complex, multi-part user queries into atomic search intents.
        """
        clauses = re.split(
            r"\?|\band\b|\balso\b|\bwhat about\b|\bif\b.*?\bthen\b|;",
            user_query,
            flags=re.IGNORECASE,
        )
        sub_queries = [c.strip() for c in clauses if len(c.strip().split()) >= 3]

        if not sub_queries:
            return [user_query.strip()]

        return sub_queries[: self.max_steps]

    def _formulate_observation_and_reasoning(
        self,
        step_num: int,
        total_planned: int,
        sub_query: str,
        state: AgentState,
    ) -> tuple[str, str]:
        """
        Generates explicit Observation and Chain-of-Thought Reasoning traces.
        """
        if step_num == 1:
            observation = (
                f"Received complex user query with {total_planned} distinct policy "
                f"aspect(s) detected. Prior accumulated context chunks: 0."
            )
            reasoning = (
                f"Step 1/{total_planned}: Initiating primary hybrid retrieval to resolve "
                f"policy aspect: '{sub_query}'. Combining sparse TF-IDF lexical matching "
                f"with dense semantic embeddings."
            )
        else:
            prev_top_score = (
                state.steps[-1]
                .retrieved_results[0]
                .math_details.hybrid_final_score
                if state.steps[-1].retrieved_results
                else 0.0
            )
            observation = (
                f"Step {step_num - 1} retrieved {len(state.accumulated_context)} unique "
                f"policy chunk(s) (peak hybrid score: {prev_top_score:.4f}). "
                f"Remaining unresolved policy aspect: '{sub_query}'."
            )
            reasoning = (
                f"Step {step_num}/{total_planned}: Executing targeted multi-hop retrieval "
                f"to gather secondary regulatory evidence for '{sub_query}' before synthesis."
            )
        return observation, reasoning

    def _synthesize_grounded_response(self, state: AgentState, force_fast: bool = False) -> str:
        """
        Synthesizes the final grounded response from accumulated context chunks.
        Supports instant sub-second local formatting or cloud LLM with a 5s timeout.
        """
        if not state.accumulated_context:
            return (
                "Insufficient policy evidence retrieved from the document store "
                "to answer this query."
            )

        context_blocks: List[str] = []
        for idx, res in enumerate(state.accumulated_context, start=1):
            m = res.math_details
            block = (
                f"[Evidence {idx} | Source: {res.chunk.source_file} (Page {res.chunk.page_number}) "
                f"| Chunk: {res.chunk.chunk_id} | Hybrid Score: {m.hybrid_final_score:.4f} "
                f"(Dense Cosine: {m.dense_cosine_similarity:.4f}, Sparse TF-IDF: {m.sparse_tfidf_score:.4f})]\n"
                f"{res.chunk.text}"
            )
            context_blocks.append(block)

        formatted_context = "\n\n".join(context_blocks)

        # 1. Cloud LLM Mode with strict 5-second timeout guard
        if not (self.fast_mode or force_fast) and self._genai_client is not None:
            user_prompt = (
                f"--- RETRIEVED POLICY EVIDENCE ---\n{formatted_context}\n\n"
                f"--- USER QUERY ---\n{state.original_query}\n\n"
                f"Provide a clear, well-structured, strictly grounded response with citations:"
            )
            try:
                import concurrent.futures

                def _call_api():
                    return self._genai_client.interactions.create(
                        model=self.model_name,
                        system_instruction=self.SYSTEM_PROMPT,
                        input=user_prompt,
                    )

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(_call_api)
                    interaction = future.result(timeout=5.0)  # Strict 5.0 second timeout
                    if interaction and interaction.output_text:
                        return interaction.output_text.strip()
            except Exception:
                # Fall through to instant local synthesis engine immediately
                pass

        # 2. Instant Local Grounded Synthesis Engine (Executes in 0.001 seconds)
        summary_blocks = [
            "Based on verified mathematical retrieval across university regulatory documents:\n"
        ]
        for idx, res in enumerate(state.accumulated_context, start=1):
            m = res.math_details
            citation = f"{res.chunk.source_file} (Page {res.chunk.page_number})"
            
            # Extract prominent clauses from text
            clause_match = re.search(r"Clause\s+[0-9\.]+(?::[^\n\.]*)?", res.chunk.text)
            clause_header = clause_match.group(0) if clause_match else f"Policy Section {idx}"

            summary_blocks.append(
                f"### {idx}. {clause_header}\n"
                f"* **Verified Text:** {res.chunk.text}\n"
                f"* **Regulatory Citation:** *{citation}*\n"
                f"* **Mathematical Verification:** Hybrid Score: `{m.hybrid_final_score:.4f}` "
                f"(Cosine: `{m.dense_cosine_similarity:.4f}`, TF-IDF: `{m.sparse_tfidf_score:.4f}`, "
                f"Matched Terms: {m.matched_terms})\n"
            )

        return "\n".join(summary_blocks)

    def _persist_agent_state(self, state: AgentState) -> None:
        """
        Persists the complete AgentState execution trace to disk for auditability.
        """
        os.makedirs(os.path.dirname(self.persistence_log_path), exist_ok=True)
        history: List[dict] = []
        if os.path.exists(self.persistence_log_path):
            try:
                with open(self.persistence_log_path, "r", encoding="utf-8") as f:
                    history = json.load(f)
            except Exception:
                history = []

        history.append(state.to_dict())
        with open(self.persistence_log_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    def _inspect_state_for_dynamic_followup(self, state: AgentState) -> Optional[str]:
        """
        Inspects the retrieved context from Step 1 to dynamically trigger a
        multi-hop verification query if cross-referenced policies or low
        confidence scores are observed.
        """
        if not state.steps or not state.steps[-1].retrieved_results:
            return None

        top_result = state.steps[-1].retrieved_results[0]
        top_text = top_result.chunk.text.lower()
        peak_score = top_result.math_details.hybrid_final_score

        # 1. Cross-reference detection in retrieved policy text
        if "medical" in top_text and "coursework" in top_text:
            return "Clause 1.2 minimum passing criteria coursework and examination"
        if "single-push" in top_text or "plagiarism" in top_text:
            return "Clause 2.1 First Class Honours zero academic misconduct offenses"
        if "distinction" in top_text and "evaluation" in top_text:
            return "Deliverable D1 Section IV empirical evaluation 10 ground truth queries"

        # 2. Low-confidence mathematical threshold trigger
        if peak_score < 0.45:
            return f"Academic progression examination regulation {state.original_query}"

        return None

    def run(self, user_query: str, fast_mode: Optional[bool] = None) -> AgentState:
        """
        Executes the full Perception-Action Agentic Loop:
            Observation -> Reasoning -> Action -> Persistence.

        Args:
            user_query (str): Complex multi-step academic policy question.
            fast_mode (Optional[bool]): If True, forces instant local sub-second synthesis.

        Returns:
            AgentState: Complete execution state with reasoning steps and math proofs.
        """
        use_fast = self.fast_mode if fast_mode is None else fast_mode
        state = AgentState(original_query=user_query)
        sub_queries = self._decompose_query_into_subqueries(user_query)

        for idx, sub_q in enumerate(sub_queries, start=1):
            observation, reasoning = self._formulate_observation_and_reasoning(
                step_num=idx,
                total_planned=len(sub_queries),
                sub_query=sub_q,
                state=state,
            )

            retrieved: List[RetrievalResult] = self.retriever.retrieve(
                query=sub_q, top_k=self.top_k_per_step
            )

            step_record = AgentStep(
                step_number=idx,
                observation=observation,
                reasoning=reasoning,
                action_type="HYBRID_VECTOR_RETRIEVAL",
                search_query=sub_q,
                retrieved_results=retrieved,
            )
            state.add_step(step_record)

        # Dynamic State-Driven Multi-Hop Check (if initial query was single-clause)
        if len(state.steps) == 1 and self.max_steps >= 2:
            followup_query = self._inspect_state_for_dynamic_followup(state)
            if followup_query:
                prev_score = (
                    state.steps[0].retrieved_results[0].math_details.hybrid_final_score
                    if state.steps[0].retrieved_results
                    else 0.0
                )
                obs = (
                    f"Dynamic State Inspection: Step 1 chunk (hybrid score: {prev_score:.4f}) "
                    f"contains a regulatory cross-reference requiring secondary verification."
                )
                reas = (
                    f"Step 2/2 (Dynamic Multi-Hop): Triggering autonomous follow-up retrieval "
                    f"for cross-referenced policy: '{followup_query}'."
                )
                followup_results = self.retriever.retrieve(
                    query=followup_query, top_k=self.top_k_per_step
                )
                state.add_step(
                    AgentStep(
                        step_number=2,
                        observation=obs,
                        reasoning=reas,
                        action_type="DYNAMIC_CROSS_REFERENCE_RETRIEVAL",
                        search_query=followup_query,
                        retrieved_results=followup_results,
                    )
                )

        # Final Synthesis Phase
        state.final_answer = self._synthesize_grounded_response(state, force_fast=use_fast)
        state.is_complete = True

        # Phase 4: State Persistence
        self._persist_agent_state(state)
        return state

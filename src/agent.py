"""
Agentic RAG Controller Module (Perception-Action Cycle).

This module implements the `BaseAgent` interface using Constructor Dependency
Injection. It orchestrates an explicit multi-step Agentic Loop:
    Observation -> Reasoning (Chain-of-Thought) -> Action (Hybrid Retrieval) -> Persistence.

It supports Google Gemini (`google-genai`) for LLM reasoning and grounded synthesis,
while providing a deterministic analytical fallback engine so the pipeline remains
fully reproducible in offline or unauthenticated environments.
"""

import os
import re
import json
from typing import List, Optional
from dotenv import load_dotenv

from src.interfaces import (
    BaseAgent,
    BaseRetriever,
    AgentStep,
    AgentState,
    RetrievalResult,
)

load_dotenv()


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
        model_name: str = "gemini-2.5-flash",
        persistence_log_path: str = "data/agent_state_history.json",
    ) -> None:
        self.retriever = retriever
        self.max_steps = max_steps
        self.top_k_per_step = top_k_per_step
        self.model_name = model_name
        self.persistence_log_path = persistence_log_path

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

        # Cap sub-queries to max_steps to bound the Agentic Loop
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

    def _synthesize_grounded_response(self, state: AgentState) -> str:
        """
        Synthesizes the final grounded response from accumulated context chunks.
        Uses Gemini LLM when configured, or falls back to deterministic synthesis.
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

        # 1. Try LLM synthesis if Gemini API key is configured
        if self._genai_client is not None:
            prompt = (
                f"{self.SYSTEM_PROMPT}\n\n"
                f"--- RETRIEVED POLICY EVIDENCE ---\n{formatted_context}\n\n"
                f"--- USER QUERY ---\n{state.original_query}\n\n"
                f"Provide a clear, well-structured, strictly grounded response with citations:"
            )
            try:
                response = self._genai_client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                )
                if response and response.text:
                    return response.text.strip()
            except Exception:
                pass

        # 2. Deterministic Grounded Synthesis Engine (Reproducible Offline Mode)
        summary_lines = [
            "Based on the mathematical retrieval and multi-step policy verification, "
            "the following regulations apply:\n"
        ]
        for idx, res in enumerate(state.accumulated_context, start=1):
            m = res.math_details
            citation = f"[{res.chunk.source_file}, Page {res.chunk.page_number}]"
            summary_lines.append(
                f"{idx}. {res.chunk.text}\n"
                f"   -> Verified via {citation} "
                f"(Hybrid Score: {m.hybrid_final_score:.4f}, "
                f"Matched Terms: {m.matched_terms})"
            )

        return "\n\n".join(summary_lines)

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

    def run(self, user_query: str) -> AgentState:
        """
        Executes the full Perception-Action Agentic Loop:
            Observation -> Reasoning -> Action -> Persistence.

        Args:
            user_query (str): Complex multi-step academic policy question.

        Returns:
            AgentState: Complete execution state with reasoning steps and math proofs.
        """
        state = AgentState(original_query=user_query)
        sub_queries = self._decompose_query_into_subqueries(user_query)

        for idx, sub_q in enumerate(sub_queries, start=1):
            # Phase 1 & 2: Observation & Reasoning (Chain-of-Thought)
            observation, reasoning = self._formulate_observation_and_reasoning(
                step_num=idx,
                total_planned=len(sub_queries),
                sub_query=sub_q,
                state=state,
            )

            # Phase 3: Action (Explainable Hybrid Retrieval)
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

        # Final Synthesis Phase
        state.final_answer = self._synthesize_grounded_response(state)
        state.is_complete = True

        # Phase 4: State Persistence
        self._persist_agent_state(state)
        return state
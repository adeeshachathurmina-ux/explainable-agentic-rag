"""
Automated Empirical RAG Evaluation Framework.

This module implements the `BaseEvaluator` interface to programmatically score
the Explainable Agentic RAG pipeline across three canonical RAG Triad metrics:
    1. Context Relevance: Alignment between the query/expected policy terms and
       the retrieved document chunks.
    2. Groundedness (Faithfulness): Degree to which the generated response is
       strictly entailed by and anchored in the retrieved context chunks without
       extrinsic hallucination.
    3. Answer Relevance: Semantic and lexical alignment between the generated
       response, the user query, and the ground-truth reference answer.
"""

import re
from typing import List, Optional, Set
import numpy as np
from sentence_transformers import SentenceTransformer

from src.interfaces import BaseEvaluator, AgentState, EvaluationMetrics


class RAGEvaluator(BaseEvaluator):
    """
    Programmatic evaluator computing the RAG Triad metrics using dense semantic
    cosine similarity and lexical token verification.
    """

    STOPWORDS: Set[str] = {
        "the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "of",
        "with", "by", "from", "is", "are", "was", "were", "be", "been", "what",
        "how", "why", "when", "where", "who", "which", "that", "this", "it",
        "as", "if", "under", "based", "following", "apply", "according",
    }

    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2") -> None:
        self.embedding_model = SentenceTransformer(embedding_model_name)

    def _cosine_similarity(self, text_a: str, text_b: str) -> float:
        """Computes L2-normalized cosine similarity between two text strings."""
        if not text_a.strip() or not text_b.strip():
            return 0.0
        vecs = self.embedding_model.encode(
            [text_a, text_b], normalize_embeddings=True, show_progress_bar=False
        )
        score = float(np.dot(vecs[0], vecs[1]))
        return max(0.0, min(1.0, score))

    @classmethod
    def _extract_content_tokens(cls, text: str) -> Set[str]:
        """Extracts normalized alphanumeric content tokens excluding stopwords."""
        tokens = re.findall(r"[a-z0-9]+(?:\.[0-9]+)?%?", text.lower())
        return {t for t in tokens if t not in cls.STOPWORDS and len(t) > 1}

    def compute_context_relevance(
        self,
        query: str,
        retrieved_texts: List[str],
        expected_keywords: Optional[List[str]] = None,
    ) -> float:
        """
        Calculates Context Relevance in [0.0, 1.0].

        Combines dense semantic similarity between the query and retrieved chunks
        with expected ground-truth keyword recall in the retrieved context.
        """
        if not retrieved_texts:
            return 0.0

        combined_context = " ".join(retrieved_texts)
        semantic_sim = self._cosine_similarity(query, combined_context)

        if expected_keywords:
            context_lower = combined_context.lower()
            hits = sum(
                1 for kw in expected_keywords if kw.lower() in context_lower
            )
            keyword_recall = hits / len(expected_keywords)
            score = (0.45 * semantic_sim) + (0.55 * keyword_recall)
        else:
            query_tokens = self._extract_content_tokens(query)
            context_tokens = self._extract_content_tokens(combined_context)
            overlap = (
                len(query_tokens.intersection(context_tokens)) / len(query_tokens)
                if query_tokens
                else 0.0
            )
            score = (0.60 * semantic_sim) + (0.40 * overlap)

        return round(min(1.0, max(0.0, score)), 4)

    def compute_groundedness(
        self, generated_answer: str, retrieved_texts: List[str]
    ) -> float:
        """
        Calculates Groundedness / Faithfulness in [0.0, 1.0].

        Quantifies the proportion of factual content tokens and semantic claims
        in the generated answer that are directly verifiable in the retrieved chunks.
        """
        if not generated_answer.strip() or not retrieved_texts:
            return 0.0

        combined_context = " ".join(retrieved_texts)
        answer_tokens = self._extract_content_tokens(generated_answer)
        context_tokens = self._extract_content_tokens(combined_context)

        if not answer_tokens:
            return 0.0

        # Lexical token precision against retrieved source context
        supported_tokens = answer_tokens.intersection(context_tokens)
        token_precision = len(supported_tokens) / len(answer_tokens)

        # Semantic entailment similarity between generated answer and retrieved context
        semantic_alignment = self._cosine_similarity(
            generated_answer, combined_context
        )

        groundedness = (0.55 * token_precision) + (0.45 * semantic_alignment)
        return round(min(1.0, max(0.0, groundedness)), 4)

    def compute_answer_relevance(
        self, query: str, generated_answer: str, ground_truth_answer: str
    ) -> float:
        """
        Calculates Answer Relevance in [0.0, 1.0].

        Combines semantic alignment with the ground-truth answer, semantic alignment
        with the user query, and key factual token recall.
        """
        if not generated_answer.strip():
            return 0.0

        sim_to_gt = self._cosine_similarity(generated_answer, ground_truth_answer)
        sim_to_query = self._cosine_similarity(generated_answer, query)

        gt_tokens = self._extract_content_tokens(ground_truth_answer)
        ans_tokens = self._extract_content_tokens(generated_answer)
        token_recall = (
            len(gt_tokens.intersection(ans_tokens)) / len(gt_tokens)
            if gt_tokens
            else 0.0
        )

        score = (0.50 * sim_to_gt) + (0.20 * sim_to_query) + (0.30 * token_recall)
        return round(min(1.0, max(0.0, score)), 4)

    def evaluate_sample(
        self,
        query: str,
        ground_truth: str,
        agent_state: AgentState,
        expected_keywords: Optional[List[str]] = None,
    ) -> EvaluationMetrics:
        """
        Evaluates a full AgentState execution against a ground-truth sample.
        """
        retrieved_texts = [
            res.chunk.text for res in agent_state.accumulated_context
        ]
        generated_answer = agent_state.final_answer

        context_rel = self.compute_context_relevance(
            query=query,
            retrieved_texts=retrieved_texts,
            expected_keywords=expected_keywords,
        )
        groundedness = self.compute_groundedness(
            generated_answer=generated_answer,
            retrieved_texts=retrieved_texts,
        )
        answer_rel = self.compute_answer_relevance(
            query=query,
            generated_answer=generated_answer,
            ground_truth_answer=ground_truth,
        )

        explanation = (
            f"Evaluated over {len(retrieved_texts)} retrieved chunk(s) across "
            f"{len(agent_state.steps)} agentic step(s)."
        )

        return EvaluationMetrics(
            query=query,
            ground_truth_answer=ground_truth,
            generated_answer=generated_answer,
            context_relevance=context_rel,
            groundedness=groundedness,
            answer_relevance=answer_rel,
            explanation=explanation,
        )
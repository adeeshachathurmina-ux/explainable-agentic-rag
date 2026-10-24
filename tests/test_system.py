"""
Automated Unit Test Suite for the Explainable Agentic RAG System.

Validates foundational vector space mathematics (Dot Product, Euclidean L2 Norm,
Cosine Similarity), AgentState deduplication, and PDF sliding-window chunking
to support Continuous Integration (CI/CD).
"""

import unittest
import numpy as np

from src.interfaces import (
    DocumentChunk,
    MathExplanation,
    RetrievalResult,
    AgentStep,
    AgentState,
)
from src.math_retriever import HybridMathRetriever
from src.ingestion import PDFDocumentProcessor


class TestMathematicalFoundations(unittest.TestCase):
    """Verifies linear algebra operations exposed in the retrieval pipeline."""

    def test_orthogonal_vectors_have_zero_cosine_similarity(self) -> None:
        q = np.array([3.0, 0.0], dtype=np.float64)
        d = np.array([0.0, 4.0], dtype=np.float64)
        dot, q_mag, d_mag, cos_sim = HybridMathRetriever.compute_vector_mathematics(q, d)

        self.assertAlmostEqual(dot, 0.0, places=5)
        self.assertAlmostEqual(q_mag, 3.0, places=5)
        self.assertAlmostEqual(d_mag, 4.0, places=5)
        self.assertAlmostEqual(cos_sim, 0.0, places=5)

    def test_collinear_vectors_have_unit_cosine_similarity(self) -> None:
        q = np.array([1.0, 2.0, 2.0], dtype=np.float64)  # L2 norm = 3.0
        d = np.array([2.0, 4.0, 4.0], dtype=np.float64)  # L2 norm = 6.0
        dot, q_mag, d_mag, cos_sim = HybridMathRetriever.compute_vector_mathematics(q, d)

        self.assertAlmostEqual(dot, 18.0, places=5)
        self.assertAlmostEqual(q_mag, 3.0, places=5)
        self.assertAlmostEqual(d_mag, 6.0, places=5)
        self.assertAlmostEqual(cos_sim, 1.0, places=5)


class TestAgentStatePersistence(unittest.TestCase):
    """Verifies state management and context deduplication across agent steps."""

    def test_agent_state_deduplicates_retrieved_chunks(self) -> None:
        chunk = DocumentChunk(
            chunk_id="test_c1",
            text="Clause 2.1 requires a 3.70 GPA for First Class Honours.",
            source_file="policy.pdf",
            page_number=1,
        )
        math_proof = MathExplanation(
            matched_terms=["gpa"],
            sparse_tfidf_score=0.5,
            dense_dot_product=0.8,
            query_magnitude=1.0,
            doc_magnitude=1.0,
            dense_cosine_similarity=0.8,
            alpha_weight=0.65,
            hybrid_final_score=0.695,
        )
        result = RetrievalResult(chunk=chunk, math_details=math_proof)

        state = AgentState(original_query="Test query")
        step1 = AgentStep(1, "Obs 1", "Reason 1", "RETRIEVE", "q1", [result])
        step2 = AgentStep(2, "Obs 2", "Reason 2", "RETRIEVE", "q2", [result])

        state.add_step(step1)
        state.add_step(step2)

        self.assertEqual(len(state.steps), 2)
        self.assertEqual(len(state.accumulated_context), 1)


class TestStructuralChunking(unittest.TestCase):
    """Verifies sliding-window chunking and overlap validation."""

    def test_invalid_overlap_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            PDFDocumentProcessor(chunk_size=50, chunk_overlap=50)

    def test_sliding_window_creates_chunks(self) -> None:
        processor = PDFDocumentProcessor(chunk_size=10, chunk_overlap=3)
        sample_text = (
            "Sentence one has five words. Sentence two also has five words. "
            "Sentence three adds more words to trigger a second chunk."
        )
        chunks = processor._create_sliding_window_chunks(
            text=sample_text,
            source_file="sample.pdf",
            page_number=1,
            has_table=False,
        )
        self.assertGreaterEqual(len(chunks), 2)


if __name__ == "__main__":
    unittest.main()
"""
Core Architectural Interfaces and Data Models.

This module defines the Abstract Base Classes (ABCs) and immutable/structured
data models for the Explainable Agentic RAG System. It enforces the Strategy
Pattern and Dependency Injection across the ingestion, hybrid mathematical
retrieval, agentic reasoning, and evaluation pipelines.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional


# ============================================================================
# 1. DOMAIN DATA MODELS (State Management & Mathematical Traceability)
# ============================================================================

@dataclass
class DocumentChunk:
    """Represents a processed, semantically coherent segment of a source PDF."""
    chunk_id: str
    text: str
    source_file: str
    page_number: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the chunk into a dictionary for persistence and logging."""
        return asdict(self)


@dataclass
class MathExplanation:
    """
    Encapsulates the mathematical mechanics of the hybrid retrieval phase.

    Exposes both sparse lexical vector mathematics (TF-IDF) and dense vector
    space operations (Dot Product, Euclidean Magnitude/L2 Norm, Cosine Similarity).
    """
    matched_terms: List[str]
    sparse_tfidf_score: float
    dense_dot_product: float
    query_magnitude: float
    doc_magnitude: float
    dense_cosine_similarity: float
    alpha_weight: float
    hybrid_final_score: float

    def to_dict(self) -> Dict[str, Any]:
        """Serializes mathematical metrics for traceability logs."""
        return asdict(self)


@dataclass
class RetrievalResult:
    """Pairs a retrieved document chunk with its mathematical explainability proof."""
    chunk: DocumentChunk
    math_details: MathExplanation

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk": self.chunk.to_dict(),
            "math_details": self.math_details.to_dict(),
        }


@dataclass
class AgentStep:
    """Represents a single iteration of the Perception-Action (Agentic) Cycle."""
    step_number: int
    observation: str
    reasoning: str
    action_type: str
    search_query: str
    retrieved_results: List[RetrievalResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_number": self.step_number,
            "observation": self.observation,
            "reasoning": self.reasoning,
            "action_type": self.action_type,
            "search_query": self.search_query,
            "retrieved_results": [r.to_dict() for r in self.retrieved_results],
        }


@dataclass
class AgentState:
    """
    Maintains the complete state across the multi-step Agentic Loop.

    Provides state persistence across Observation -> Reasoning -> Action transitions.
    """
    original_query: str
    steps: List[AgentStep] = field(default_factory=list)
    accumulated_context: List[RetrievalResult] = field(default_factory=list)
    final_answer: str = ""
    is_complete: bool = False

    def add_step(self, step: AgentStep) -> None:
        """Appends a new reasoning step and deduplicates accumulated context."""
        self.steps.append(step)
        existing_ids = {res.chunk.chunk_id for res in self.accumulated_context}
        for res in step.retrieved_results:
            if res.chunk.chunk_id not in existing_ids:
                self.accumulated_context.append(res)
                existing_ids.add(res.chunk.chunk_id)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the full agent execution trace for auditing and persistence."""
        return {
            "original_query": self.original_query,
            "steps": [step.to_dict() for step in self.steps],
            "final_answer": self.final_answer,
            "is_complete": self.is_complete,
        }


@dataclass
class EvaluationMetrics:
    """Holds empirical RAG evaluation scores for a single ground-truth test case."""
    query: str
    ground_truth_answer: str
    generated_answer: str
    context_relevance: float
    groundedness: float
    answer_relevance: float
    explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# 2. ABSTRACT BASE CLASSES (Interfaces for Dependency Injection)
# ============================================================================

class BaseDocumentProcessor(ABC):
    """Interface for PDF ingestion and structural text chunking strategies."""

    @abstractmethod
    def process_pdf(self, file_path: str) -> List[DocumentChunk]:
        """Extracts and chunks text from a single PDF document."""
        pass

    @abstractmethod
    def ingest_directory(self, directory_path: str) -> List[DocumentChunk]:
        """Processes all PDF documents within a target directory."""
        pass


class BaseRetriever(ABC):
    """Interface for persistent vector storage and explainable hybrid retrieval."""

    @abstractmethod
    def index_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Computes embeddings/weights and persists chunks to the vector store."""
        pass

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 3) -> List[RetrievalResult]:
        """Executes a hybrid search and returns chunks with mathematical proofs."""
        pass


class BaseAgent(ABC):
    """Interface for the Agentic RAG controller implementing the Perception-Action loop."""

    @abstractmethod
    def run(self, user_query: str) -> AgentState:
        """Executes the Observation -> Reasoning -> Action cycle until resolution."""
        pass


class BaseEvaluator(ABC):
    """Interface for automated empirical evaluation of RAG outputs."""

    @abstractmethod
    def evaluate_sample(
        self,
        query: str,
        ground_truth: str,
        agent_state: AgentState,
    ) -> EvaluationMetrics:
        """Calculates Context Relevance, Groundedness, and Answer Relevance."""
        pass
"""
Mathematical Foundations and Persistent Hybrid Retrieval Module.

This module implements the `BaseRetriever` interface, bridging foundational
lexical vector space mathematics (TF-IDF, Vocabulary Overlap) with modern dense
embedding retrieval. It explicitly computes and logs raw Dot Products, Euclidean
Vector Magnitudes (L2 Norms), explicit L2 Normalization, Cosine Similarity, and
Convex Hybrid Score Fusion while persisting vectors in ChromaDB with in-memory
vectorized matrix caching for sub-millisecond retrieval latency.
"""

import os
from typing import List, Dict, Any, Tuple
import numpy as np
import chromadb
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

# Disable verbose network warnings to optimize execution latency
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from src.interfaces import (
    BaseRetriever,
    DocumentChunk,
    MathExplanation,
    RetrievalResult,
)


class HybridMathRetriever(BaseRetriever):
    """
    Explainable Hybrid Retriever combining Sparse TF-IDF and Dense Embeddings.

    Attributes:
        persist_directory (str): Directory path for ChromaDB disk persistence.
        collection_name (str): Name of the persistent vector collection.
        alpha (float): Convex combination weight in [0, 1] balancing dense
            semantic similarity (alpha) and sparse lexical TF-IDF (1 - alpha).
    """

    def __init__(
        self,
        persist_directory: str = "data/chroma_db",
        collection_name: str = "academic_policy_store",
        embedding_model_name: str = "all-MiniLM-L6-v2",
        alpha: float = 0.65,
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("Hybrid weighting parameter alpha must be in [0.0, 1.0].")

        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.alpha = alpha

        # Initialize persistent vector store on disk
        self.chroma_client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Persistent store for Explainable Agentic RAG"},
        )

        # Load dense bi-encoder model with local caching
        self.embedding_model = SentenceTransformer(
            embedding_model_name,
            device="cpu",
        )
        # Remove internal Normalize layer to expose raw unnormalized vector magnitudes
        if len(self.embedding_model) > 2 and "Normalize" in type(self.embedding_model[-1]).__name__:
            self.embedding_model._modules.pop(str(len(self.embedding_model) - 1), None)

        # Initialize sparse TF-IDF vectorizer (unigrams and bigrams)
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            norm="l2",
            sublinear_tf=True,
        )
        self._tfidf_matrix = None
        self._corpus_chunks: List[DocumentChunk] = []
        self._cached_dense_embeddings: np.ndarray = np.empty((0, 384), dtype=np.float64)

        # Restore sparse index and in-memory dense matrix if collection already exists
        self._restore_from_persistent_store()

    def _restore_from_persistent_store(self) -> None:
        """Restores in-memory TF-IDF corpus state and vector cache from ChromaDB."""
        existing_count = self.collection.count()
        if existing_count > 0:
            stored = self.collection.get(include=["documents", "metadatas", "embeddings"])
            restored_chunks: List[DocumentChunk] = []
            for chunk_id, text, meta in zip(
                stored["ids"], stored["documents"], stored["metadatas"]
            ):
                restored_chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        text=text,
                        source_file=str(meta.get("source_file", "unknown.pdf")),
                        page_number=int(meta.get("page_number", 1)),
                        metadata=meta,
                    )
                )
            self._corpus_chunks = restored_chunks
            texts = [c.text for c in self._corpus_chunks]
            self._tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)

            if stored.get("embeddings") is not None and len(stored["embeddings"]) > 0:
                self._cached_dense_embeddings = np.asarray(
                    stored["embeddings"], dtype=np.float64
                )

    def index_chunks(self, chunks: List[DocumentChunk]) -> None:
        """
        Computes unnormalized dense embeddings and TF-IDF matrices, persisting to disk.

        Args:
            chunks (List[DocumentChunk]): Structured document chunks to index.
        """
        if not chunks:
            raise ValueError("Cannot index an empty list of DocumentChunks.")

        texts = [c.text for c in chunks]
        ids = [c.chunk_id for c in chunks]

        # Encode without automatic normalization so raw vector magnitudes can be audited
        dense_vectors = self.embedding_model.encode(
            texts, normalize_embeddings=False, show_progress_bar=False, batch_size=32
        )

        metadatas: List[Dict[str, Any]] = []
        for c in chunks:
            meta = {
                "source_file": c.source_file,
                "page_number": int(c.page_number),
                "word_count": int(c.metadata.get("word_count", 0)),
                "contains_table": bool(c.metadata.get("contains_table", False)),
            }
            metadatas.append(meta)

        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=dense_vectors.tolist(),
            metadatas=metadatas,
        )

        # Re-synchronize in-memory cache and TF-IDF representation
        self._restore_from_persistent_store()

    @staticmethod
    def compute_vector_mathematics(
        query_vec: np.ndarray, doc_vec: np.ndarray
    ) -> Tuple[float, float, float, float]:
        """
        Explicitly computes foundational linear algebra operations in vector space.

        Formulae:
            Dot Product: q . d = sum(q_i * d_i)
            Query Magnitude (L2 Norm): ||q||_2 = sqrt(sum(q_i^2))
            Doc Magnitude (L2 Norm): ||d||_2 = sqrt(sum(d_i^2))
            Cosine Similarity: (q . d) / (||q||_2 * ||d||_2)

        Returns:
            Tuple[float, float, float, float]:
                (dot_product, query_magnitude, doc_magnitude, cosine_similarity)
        """
        q = np.asarray(query_vec, dtype=np.float64)
        d = np.asarray(doc_vec, dtype=np.float64)

        dot_product = float(np.dot(q, d))
        query_magnitude = float(np.linalg.norm(q))
        doc_magnitude = float(np.linalg.norm(d))

        denominator = query_magnitude * doc_magnitude
        if denominator == 0.0:
            cosine_similarity = 0.0
        else:
            cosine_similarity = float(dot_product / denominator)

        return dot_product, query_magnitude, doc_magnitude, cosine_similarity

    def _compute_sparse_tfidf_scores(
        self, query: str
    ) -> Tuple[np.ndarray, List[List[str]]]:
        """
        Computes sparse TF-IDF cosine similarities and extracts matched vocabulary terms.
        """
        if self._tfidf_matrix is None or not self._corpus_chunks:
            raise RuntimeError("Retriever index is empty. Call index_chunks() first.")

        query_tfidf = self.tfidf_vectorizer.transform([query])
        feature_names = self.tfidf_vectorizer.get_feature_names_out()

        # Since TF-IDF vectors are L2-normalized, matrix dot product equals cosine similarity
        sparse_scores = (self._tfidf_matrix @ query_tfidf.T).toarray().ravel()

        query_nonzero_indices = set(query_tfidf.nonzero()[1])
        matched_terms_per_doc: List[List[str]] = []

        for doc_idx in range(self._tfidf_matrix.shape[0]):
            doc_nonzero_indices = set(self._tfidf_matrix[doc_idx].nonzero()[1])
            overlap_indices = sorted(query_nonzero_indices.intersection(doc_nonzero_indices))
            overlap_terms = [str(feature_names[i]) for i in overlap_indices]
            matched_terms_per_doc.append(overlap_terms)

        return sparse_scores, matched_terms_per_doc

    def retrieve(self, query: str, top_k: int = 3) -> List[RetrievalResult]:
        """
        Executes vectorized hybrid retrieval with sub-millisecond in-memory operations.

        Args:
            query (str): Natural language or keyword search query.
            top_k (int): Number of highest-scoring chunks to return.

        Returns:
            List[RetrievalResult]: Ranked results with mathematical explainability.
        """
        if not self._corpus_chunks or self._cached_dense_embeddings.size == 0:
            self._restore_from_persistent_store()
        if not self._corpus_chunks:
            raise RuntimeError("Vector store is empty. Ingest and index PDFs first.")

        # 1. Sparse TF-IDF computation
        sparse_scores, matched_terms_list = self._compute_sparse_tfidf_scores(query)

        # 2. Dense Embedding vector computation
        query_dense = self.embedding_model.encode(
            [query], normalize_embeddings=False, show_progress_bar=False
        )[0].astype(np.float64)

        query_mag = float(np.linalg.norm(query_dense))

        # 3. Vectorized Matrix Multiplication for all chunks simultaneously
        # Compute dot products for all documents in a single BLAS call (sub-millisecond)
        dot_products = self._cached_embeddings_dot_product(query_dense)
        doc_magnitudes = np.linalg.norm(self._cached_dense_embeddings, axis=1)

        denominators = query_mag * doc_magnitudes
        denominators[denominators == 0.0] = 1e-9
        dense_cosines = dot_products / denominators

        # 4. Vectorized Convex Combination
        hybrid_scores = (self.alpha * dense_cosines) + ((1.0 - self.alpha) * sparse_scores)

        # 5. Extract top_k results with full mathematical proofs
        top_indices = np.argsort(hybrid_scores)[::-1][:top_k]

        results: List[RetrievalResult] = []
        for idx in top_indices:
            chunk = self._corpus_chunks[idx]
            math_proof = MathExplanation(
                matched_terms=matched_terms_list[idx],
                sparse_tfidf_score=round(float(sparse_scores[idx]), 4),
                dense_dot_product=round(float(dot_products[idx]), 4),
                query_magnitude=round(query_mag, 4),
                doc_magnitude=round(float(doc_magnitudes[idx]), 4),
                dense_cosine_similarity=round(float(dense_cosines[idx]), 4),
                alpha_weight=self.alpha,
                hybrid_final_score=round(float(hybrid_scores[idx]), 4),
            )
            results.append(RetrievalResult(chunk=chunk, math_details=math_proof))

        return results

    def _cached_embeddings_dot_product(self, query_dense: np.ndarray) -> np.ndarray:
        """Fast vectorized dot product over in-memory cached embeddings matrix."""
        return self._cached_dense_embeddings @ query_dense

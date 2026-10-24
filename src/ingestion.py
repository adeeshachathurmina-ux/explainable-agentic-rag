"""
Advanced PDF Ingestion and Structure-Aware Chunking Module.

This module implements the `BaseDocumentProcessor` interface using PyMuPDF (fitz)
to handle complex PDF layouts, including embedded tables and multi-block sections.
It applies a boundary-aware sliding-window chunking strategy with configurable
lexical overlap to preserve semantic continuity across chunks.
"""

import os
import re
import hashlib
from pathlib import Path
from typing import List, Tuple
import pymupdf as fitz  # PyMuPDF

from src.interfaces import BaseDocumentProcessor, DocumentChunk


class PDFDocumentProcessor(BaseDocumentProcessor):
    """
    Production-grade PDF processor with table extraction and sliding-window chunking.

    Attributes:
        chunk_size (int): Target maximum number of words per chunk.
        chunk_overlap (int): Number of overlapping words between consecutive chunks
            to preserve cross-boundary context.
    """

    def __init__(self, chunk_size: int = 180, chunk_overlap: int = 40) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size.")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _extract_tables_as_markdown(self, page: fitz.Page) -> Tuple[str, bool]:
        """
        Detects and extracts structured tables from a PDF page into Markdown format.

        Args:
            page (fitz.Page): A PyMuPDF page instance.

        Returns:
            Tuple[str, bool]: Formatted markdown table string and a boolean flag
                indicating whether any tables were detected.
        """
        table_texts: List[str] = []
        has_tables = False
        try:
            tables = page.find_tables()
            if tables and tables.tables:
                has_tables = True
                for idx, table in enumerate(tables.tables, start=1):
                    extracted = table.extract()
                    if not extracted:
                        continue
                    rows_md = [f"\n[Structured Table {idx}]"]
                    for row in extracted:
                        cleaned_cells = [
                            re.sub(r"\s+", " ", str(cell or "")).strip()
                            for cell in row
                        ]
                        if any(cleaned_cells):
                            rows_md.append(" | ".join(cleaned_cells))
                    table_texts.append("\n".join(rows_md))
        except Exception:
            # Fallback gracefully if layout table finder encounters non-standard objects
            pass

        return "\n\n".join(table_texts), has_tables

    def _clean_text(self, raw_text: str) -> str:
        """Normalizes unicode artifacts, ligatures, and excessive whitespace."""
        text = raw_text.replace("\u00a0", " ").replace("\ufffd", " ")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _create_sliding_window_chunks(
        self,
        text: str,
        source_file: str,
        page_number: int,
        has_table: bool,
    ) -> List[DocumentChunk]:
        """
        Splits page text into overlapping semantic chunks respecting sentence boundaries.
        """
        cleaned_text = self._clean_text(text)
        if not cleaned_text:
            return []

        # Split into sentence-like segments first to avoid mid-sentence truncation
        sentences = re.split(r"(?<=[.!?])\s+|\n+", cleaned_text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks: List[DocumentChunk] = []
        current_words: List[str] = []
        chunk_index = 0

        for sentence in sentences:
            sentence_words = sentence.split()
            if not sentence_words:
                continue

            if len(current_words) + len(sentence_words) <= self.chunk_size:
                current_words.extend(sentence_words)
            else:
                if current_words:
                    chunk_text = " ".join(current_words)
                    chunk_id = self._generate_chunk_id(
                        source_file, page_number, chunk_index, chunk_text
                    )
                    chunks.append(
                        DocumentChunk(
                            chunk_id=chunk_id,
                            text=chunk_text,
                            source_file=source_file,
                            page_number=page_number,
                            metadata={
                                "word_count": len(current_words),
                                "char_count": len(chunk_text),
                                "contains_table": has_table,
                                "chunk_index": chunk_index,
                            },
                        )
                    )
                    chunk_index += 1

                    # Retain trailing words for sliding-window overlap
                    if self.chunk_overlap > 0:
                        current_words = current_words[-self.chunk_overlap :]
                    else:
                        current_words = []

                current_words.extend(sentence_words)

        # Flush remaining words into a final chunk for the page
        if current_words:
            chunk_text = " ".join(current_words)
            chunk_id = self._generate_chunk_id(
                source_file, page_number, chunk_index, chunk_text
            )
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    source_file=source_file,
                    page_number=page_number,
                    metadata={
                        "word_count": len(current_words),
                        "char_count": len(chunk_text),
                        "contains_table": has_table,
                        "chunk_index": chunk_index,
                    },
                )
            )

        return chunks

    @staticmethod
    def _generate_chunk_id(
        source_file: str, page_number: int, chunk_index: int, text: str
    ) -> str:
        """Generates a deterministic, unique hash identifier for a document chunk."""
        content_hash = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
        stem = Path(source_file).stem
        return f"{stem}_p{page_number}_c{chunk_index}_{content_hash}"

    def process_pdf(self, file_path: str) -> List[DocumentChunk]:
        """
        Extracts layout blocks and tables from a PDF and returns structured chunks.

        Args:
            file_path (str): Path to the target PDF file.

        Returns:
            List[DocumentChunk]: List of extracted and overlapping chunks.
        """
        path = Path(file_path)
        if not path.exists() or path.suffix.lower() != ".pdf":
            raise FileNotFoundError(f"Valid PDF file not found at: {file_path}")

        all_chunks: List[DocumentChunk] = []
        doc = fitz.open(str(path))

        try:
            for page_idx in range(len(doc)):
                page = doc[page_idx]
                page_number = page_idx + 1

                # Extract standard text blocks ordered by reading layout
                raw_text = page.get_text("text", sort=True)
                # Extract structured tables if present on the page
                table_md, has_table = self._extract_tables_as_markdown(page)

                combined_text = raw_text
                if table_md:
                    combined_text = f"{raw_text}\n\n{table_md}"

                page_chunks = self._create_sliding_window_chunks(
                    text=combined_text,
                    source_file=path.name,
                    page_number=page_number,
                    has_table=has_table,
                )
                all_chunks.extend(page_chunks)
        finally:
            doc.close()

        return all_chunks

    def ingest_directory(self, directory_path: str) -> List[DocumentChunk]:
        """
        Iterates over all PDF files in a directory and ingests them into chunks.

        Args:
            directory_path (str): Path to the directory containing PDF files.

        Returns:
            List[DocumentChunk]: Aggregated list of chunks across all PDFs.
        """
        dir_path = Path(directory_path)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        pdf_files = sorted(dir_path.glob("*.pdf"))
        if not pdf_files:
            raise ValueError(f"No PDF documents found in directory: {directory_path}")

        aggregated_chunks: List[DocumentChunk] = []
        for pdf_file in pdf_files:
            file_chunks = self.process_pdf(str(pdf_file))
            aggregated_chunks.extend(file_chunks)

        return aggregated_chunks
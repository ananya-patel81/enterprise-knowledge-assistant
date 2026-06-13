"""
qa_pipeline.py
==============

Orchestrates the end-to-end Retrieval-Augmented Generation (RAG)
pipeline:

    User Question
        -> Embed Query
        -> Retrieve Relevant Chunks (vector search and/or BM25)
        -> Send Context + Question to LLM
        -> Generate Response (with citations)

WHY RAG instead of just asking the LLM directly?
-----------------------------------------------------
LLMs are trained on a fixed snapshot of public data and know nothing
about a specific company's internal documents. RAG solves this *without*
retraining the model:

  1. We retrieve the most relevant pieces of the company's own documents
     for a given question.
  2. We paste those pieces directly into the prompt as "context".
  3. The LLM answers using that context, dramatically reducing
     hallucination and allowing the answer to be grounded with
     citations back to the source documents.

This module also implements the document ingestion pipeline (the other
half of RAG): turning an uploaded file into chunks + embeddings stored in
FAISS and SQLite.
"""

import os
import time
import uuid
from typing import List

from backend.config import settings
from backend.logger import get_logger
from database import db
from rag import bm25_search
from rag.chunking import chunk_text
from rag.embeddings import embed_query, embed_texts
from rag.extraction import extract_text
from rag.llm import generate_answer
from rag.vector_store import get_vector_store

logger = get_logger(__name__)


# ----------------------------------------------------------------------
# Ingestion pipeline
# ----------------------------------------------------------------------

def ingest_document(file_bytes: bytes, filename: str) -> dict:
    """Process an uploaded file end-to-end: save -> extract -> chunk ->
    embed -> index -> persist metadata.

    Args:
        file_bytes: Raw file content.
        filename: Original filename (used to determine file type and
            for display/citations).

    Returns:
        A summary dict with document id, number of chunks, etc.
    """
    file_type = filename.rsplit(".", 1)[-1].lower()

    # 1. Save the raw file to disk (so it can be re-processed or
    #    downloaded later).
    safe_name = f"{uuid.uuid4().hex}_{filename}"
    file_path = os.path.join(settings.uploads_dir, safe_name)
    with open(file_path, "wb") as f:
        f.write(file_bytes)
    logger.info("Saved upload to %s", file_path)

    # 2. Extract + clean text.
    text = extract_text(file_path, file_type)
    if not text:
        raise ValueError("No extractable text found in document.")

    # 3. Record the document in SQLite.
    document_id = db.insert_document(filename, file_type, len(text))

    # 4. Chunk the text using the configured strategy.
    chunks = chunk_text(text)
    logger.info("Document %r split into %d chunks (strategy=%s)",
                 filename, len(chunks), settings.chunk_strategy)

    # 5. Embed all chunks in a single batch call (much faster than
    #    embedding one at a time).
    vectors = embed_texts(chunks)

    # 6. Add vectors to FAISS, then persist the chunk text + faiss id
    #    mapping in SQLite.
    store = get_vector_store()
    faiss_ids = store.add(vectors)

    for idx, (chunk, faiss_id) in enumerate(zip(chunks, faiss_ids)):
        db.insert_chunk(document_id, idx, chunk, faiss_id)

    db.update_document_chunk_count(document_id, len(chunks))
    store.save()

    return {
        "document_id": document_id,
        "filename": filename,
        "num_chunks": len(chunks),
        "char_count": len(text),
    }


# ----------------------------------------------------------------------
# Question answering
# ----------------------------------------------------------------------

def answer_question(question: str, retrieval_mode: str = "vector", top_k: int = None) -> dict:
    """Run the full RAG pipeline for a user question.

    Args:
        question: The user's natural-language question.
        retrieval_mode: "vector" (FAISS semantic search) or "bm25"
            (keyword search). Used by the retrieval-comparison feature.
        top_k: Number of chunks to retrieve.

    Returns:
        A dict containing the generated answer, the retrieved chunks
        (for displaying citations/context), the retrieval mode, and
        latency in milliseconds.
    """
    top_k = top_k or settings.top_k
    start = time.perf_counter()

    if retrieval_mode == "vector":
        retrieved_chunks = _retrieve_vector(question, top_k)
    elif retrieval_mode == "bm25":
        retrieved_chunks = _retrieve_bm25(question, top_k)
    else:
        raise ValueError(f"Unknown retrieval_mode: {retrieval_mode!r}")

    if not retrieved_chunks:
        answer = "No documents have been uploaded yet, so I have no context to answer from."
    else:
        answer = generate_answer(question, retrieved_chunks)

    latency_ms = (time.perf_counter() - start) * 1000
    db.log_query(question, retrieval_mode, latency_ms)

    return {
        "answer": answer,
        "retrieved_chunks": retrieved_chunks,
        "retrieval_mode": retrieval_mode,
        "latency_ms": round(latency_ms, 2),
    }


def _retrieve_vector(question: str, top_k: int) -> List[dict]:
    """Semantic retrieval: embed the query, search FAISS, then resolve
    each FAISS id back to its chunk text via SQLite.
    """
    store = get_vector_store()
    query_vector = embed_query(question)
    results = store.search(query_vector, top_k=top_k)

    chunks = []
    for faiss_id, score in results:
        row = db.get_chunk_by_faiss_id(faiss_id)
        if row is None:
            continue
        chunks.append({
            "document_name": row["document_name"],
            "chunk_index": row["chunk_index"],
            "content": row["content"],
            "score": round(score, 4),
        })
    return chunks


def _retrieve_bm25(question: str, top_k: int) -> List[dict]:
    """Lexical retrieval using BM25 over all chunks."""
    results = bm25_search.bm25_search(question, top_k=top_k)
    chunks = []
    for row, score in results:
        chunks.append({
            "document_name": row["document_name"],
            "chunk_index": row["chunk_index"],
            "content": row["content"],
            "score": round(score, 4),
        })
    return chunks


def compare_retrieval(question: str, top_k: int = None) -> dict:
    """Run both vector and BM25 retrieval for the same question, without
    calling the LLM. Used by the Retrieval Comparison page.
    """
    top_k = top_k or settings.top_k
    return {
        "vector_results": _retrieve_vector(question, top_k),
        "bm25_results": _retrieve_bm25(question, top_k),
    }

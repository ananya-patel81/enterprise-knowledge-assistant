"""
vector_store.py
================

A thin wrapper around FAISS for storing and searching chunk embeddings.

WHAT is FAISS?
----------------
FAISS (Facebook AI Similarity Search) is a library for efficient
similarity search over dense vectors. Given a query vector, it quickly
finds the N most similar vectors in an index containing potentially
millions of vectors.

WHY FAISS over a hosted vector database (Pinecone, Weaviate, etc.)?
------------------------------------------------------------------------
- FAISS runs **in-process**, with no network calls, no external service,
  and no monthly cost -- ideal for a student project on free hosting.
- For the data volumes here (hundreds to low-thousands of chunks), an
  in-memory FAISS index is extremely fast (sub-millisecond search) and
  far simpler to deploy and debug.
- It's still the *same underlying algorithm family* (approximate /
  exact nearest-neighbour search) that powers production vector
  databases, so the concepts transfer directly to an interview
  discussion about scaling (see docs/interview_prep.md).

WHICH FAISS INDEX TYPE?
--------------------------
We use `IndexFlatIP` (Flat = exhaustive, IP = Inner Product).

- "Flat" means FAISS compares the query against *every* stored vector
  (brute-force). This is O(N) per search but, for N in the thousands,
  takes well under a millisecond -- so the simplicity is "free" at this
  scale.
- "IP" (inner product) is used because our embeddings are L2-normalized
  (see embeddings.py). For unit vectors, inner product == cosine
  similarity. A higher score means "more similar".

COSINE SIMILARITY, briefly:
------------------------------
Cosine similarity measures the angle between two vectors, ignoring
their magnitude. Two vectors pointing in the same direction (angle = 0)
have a cosine similarity of 1 (most similar); orthogonal vectors have a
similarity of 0; opposite vectors have -1. Because embeddings encode
*direction* (meaning) more than magnitude, cosine similarity is the
standard metric for comparing them.

PERSISTENCE
--------------
FAISS indexes can be written to / read from disk (`faiss.write_index` /
`faiss.read_index`). We additionally keep a small "metadata" pickle file
that maps each FAISS row index -> chunk id, so we can look up the
original text/source in SQLite after a search.
"""

import os
import pickle
from typing import List, Tuple

import faiss
import numpy as np

from backend.config import settings
from backend.logger import get_logger
from typing import Optional

logger = get_logger(__name__)


class VectorStore:
    """Wraps a FAISS `IndexFlatIP` index plus a simple id mapping."""

    def __init__(self, dim: int = None):
        self.dim = dim or settings.embedding_dim
        self.index = faiss.IndexFlatIP(self.dim)
        # next_id is the FAISS row index that the *next* inserted vector
        # will receive. FAISS's IndexFlatIP doesn't support arbitrary
        # ids, so we rely on insertion order == row index, and store
        # this counter ourselves for bookkeeping/persistence.
        self.next_id = 0

        self._load_if_exists()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_if_exists(self) -> None:
        if os.path.exists(settings.faiss_index_path) and os.path.exists(
            settings.faiss_metadata_path
        ):
            logger.info("Loading existing FAISS index from %s", settings.faiss_index_path)
            self.index = faiss.read_index(settings.faiss_index_path)
            with open(settings.faiss_metadata_path, "rb") as f:
                meta = pickle.load(f)
            self.next_id = meta["next_id"]

    def save(self) -> None:
        """Persist the index and metadata to disk."""
        faiss.write_index(self.index, settings.faiss_index_path)
        with open(settings.faiss_metadata_path, "wb") as f:
            pickle.dump({"next_id": self.next_id}, f)
        logger.info("Saved FAISS index (%d vectors) to %s", self.index.ntotal, settings.faiss_index_path)

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def add(self, vectors: np.ndarray) -> List[int]:
        """Add vectors to the index.

        Args:
            vectors: numpy array of shape (N, dim), dtype float32.

        Returns:
            The list of FAISS ids assigned to the new vectors (in
            insertion order). These ids should be stored alongside the
            corresponding chunk rows in SQLite (`chunks.faiss_id`).
        """
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"Expected vectors of shape (N, {self.dim}), got {vectors.shape}")

        start_id = self.next_id
        self.index.add(vectors)
        new_ids = list(range(start_id, start_id + vectors.shape[0]))
        self.next_id += vectors.shape[0]
        return new_ids

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query_vector: np.ndarray, top_k: int = None) -> List[Tuple[int, float]]:
        """Return the top-k most similar vectors to `query_vector`.

        Args:
            query_vector: 1D numpy array of shape (dim,).
            top_k: number of results to return.

        Returns:
            A list of (faiss_id, similarity_score) tuples, sorted by
            descending similarity. similarity_score is the cosine
            similarity (since vectors are normalized), in range [-1, 1].
        """
        top_k = top_k or settings.top_k
        if self.index.ntotal == 0:
            return []

        top_k = min(top_k, self.index.ntotal)
        query = query_vector.reshape(1, -1).astype("float32")
        scores, ids = self.index.search(query, top_k)

        results = []
        for faiss_id, score in zip(ids[0], scores[0]):
            if faiss_id == -1:
                continue
            results.append((int(faiss_id), float(score)))
        return results

    @property
    def total_vectors(self) -> int:
        return self.index.ntotal


# A module-level singleton so the whole app shares one in-memory index.
# Loaded once at startup via init_vector_store() in backend/main.py.
_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store

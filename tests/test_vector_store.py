"""
test_vector_store.py
=====================

Unit tests for rag/vector_store.py.

We use a small, fixed dimensionality and synthetic vectors (not real
embeddings) so the test is fast and deterministic, while still
exercising the real FAISS index logic.
"""

import numpy as np
import pytest

from rag.vector_store import VectorStore


@pytest.fixture
def store():
    return VectorStore(dim=4)


def _normalize(vec: np.ndarray) -> np.ndarray:
    return vec / np.linalg.norm(vec)


class TestVectorStore:
    def test_empty_store_search_returns_empty(self, store):
        query = _normalize(np.array([1, 0, 0, 0], dtype="float32"))
        assert store.search(query, top_k=3) == []

    def test_add_assigns_sequential_ids(self, store):
        vectors = np.array(
            [
                _normalize(np.array([1, 0, 0, 0], dtype="float32")),
                _normalize(np.array([0, 1, 0, 0], dtype="float32")),
            ],
            dtype="float32",
        )
        ids = store.add(vectors)
        assert ids == [0, 1]
        assert store.total_vectors == 2

    def test_search_returns_most_similar_first(self, store):
        # Three orthogonal-ish vectors.
        v0 = _normalize(np.array([1, 0, 0, 0], dtype="float32"))
        v1 = _normalize(np.array([0, 1, 0, 0], dtype="float32"))
        v2 = _normalize(np.array([0.9, 0.1, 0, 0], dtype="float32"))  # close to v0

        store.add(np.stack([v0, v1, v2]))

        # Querying with v0 should rank v0 (id=0) first, then v2 (id=2,
        # which is close to v0), then v1 (id=1, orthogonal).
        results = store.search(v0, top_k=3)
        ids_in_order = [r[0] for r in results]

        assert ids_in_order[0] == 0
        assert ids_in_order[1] == 2
        assert ids_in_order[2] == 1

    def test_search_scores_are_cosine_similarity_range(self, store):
        v0 = _normalize(np.array([1, 0, 0, 0], dtype="float32"))
        store.add(np.stack([v0]))

        results = store.search(v0, top_k=1)
        faiss_id, score = results[0]

        assert faiss_id == 0
        # Identical normalized vectors -> cosine similarity ~= 1.
        assert 0.99 <= score <= 1.0

    def test_add_rejects_wrong_dimension(self, store):
        bad_vector = np.zeros((1, 5), dtype="float32")  # store.dim is 4
        with pytest.raises(ValueError):
            store.add(bad_vector)

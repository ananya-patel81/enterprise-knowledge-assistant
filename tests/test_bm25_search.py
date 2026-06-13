"""
test_bm25_search.py
====================

Unit tests for rag/bm25_search.py.

We monkeypatch `database.db.get_all_chunks` so these tests don't depend
on an actual SQLite database -- a common pattern for isolating unit
tests from infrastructure (a good interview talking point about test
design / dependency injection).
"""

from rag import bm25_search


FAKE_CHUNKS = [
    {
        "id": 1,
        "document_id": 1,
        "chunk_index": 0,
        "content": "The password reset procedure requires visiting the identity portal.",
        "faiss_id": 0,
        "document_name": "it_security_policy.txt",
    },
    {
        "id": 2,
        "document_id": 1,
        "chunk_index": 1,
        "content": "Multi-factor authentication is mandatory for production databases.",
        "faiss_id": 1,
        "document_name": "it_security_policy.txt",
    },
    {
        "id": 3,
        "document_id": 2,
        "chunk_index": 0,
        "content": "Employees accrue eighteen days of paid annual leave per year.",
        "faiss_id": 2,
        "document_name": "employee_handbook.md",
    },
]


def test_bm25_search_returns_empty_for_empty_corpus(monkeypatch):
    monkeypatch.setattr(bm25_search.db, "get_all_chunks", lambda: [])
    assert bm25_search.bm25_search("anything") == []


def test_bm25_search_ranks_keyword_match_highest(monkeypatch):
    monkeypatch.setattr(bm25_search.db, "get_all_chunks", lambda: FAKE_CHUNKS)

    results = bm25_search.bm25_search("password reset", top_k=3)

    assert len(results) == 3
    top_chunk, top_score = results[0]
    assert top_chunk["document_name"] == "it_security_policy.txt"
    assert top_chunk["chunk_index"] == 0
    assert top_score > 0


def test_bm25_search_respects_top_k(monkeypatch):
    monkeypatch.setattr(bm25_search.db, "get_all_chunks", lambda: FAKE_CHUNKS)

    results = bm25_search.bm25_search("policy", top_k=2)
    assert len(results) == 2


def test_tokenize_lowercases_and_splits():
    tokens = bm25_search._tokenize("Hello World Test")
    assert tokens == ["hello", "world", "test"]

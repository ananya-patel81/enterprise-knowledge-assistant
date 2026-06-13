"""
test_chunking.py
=================

Unit tests for rag/chunking.py.

These tests verify both chunking strategies behave correctly on edge
cases (empty text, text shorter than chunk size, overlap behaviour) --
the kind of tests an interviewer would expect to see for a "core
algorithm" module.
"""

import pytest

from rag.chunking import chunk_text, fixed_size_chunks, recursive_chunks


class TestFixedSizeChunks:
    def test_empty_text_returns_no_chunks(self):
        assert fixed_size_chunks("") == []
        assert fixed_size_chunks("   ") == []

    def test_text_shorter_than_chunk_size_returns_single_chunk(self):
        text = "Hello world"
        result = fixed_size_chunks(text, chunk_size=100, overlap=10)
        assert result == ["Hello world"]

    def test_basic_splitting_with_overlap(self):
        text = "ABCDEFGHIJ"
        result = fixed_size_chunks(text, chunk_size=4, overlap=1)
        # Each chunk after the first overlaps the previous by 1 char.
        assert result == ["ABCD", "DEFG", "GHIJ"]

    def test_overlap_must_be_smaller_than_chunk_size(self):
        with pytest.raises(ValueError):
            fixed_size_chunks("some text", chunk_size=5, overlap=5)

    def test_chunk_size_must_be_positive(self):
        with pytest.raises(ValueError):
            fixed_size_chunks("some text", chunk_size=0, overlap=0)


class TestRecursiveChunks:
    def test_empty_text_returns_no_chunks(self):
        assert recursive_chunks("") == []

    def test_short_text_returns_single_chunk(self):
        text = "This is a short paragraph."
        result = recursive_chunks(text, chunk_size=100, overlap=10)
        assert len(result) == 1
        assert result[0] == text

    def test_splits_on_paragraph_boundaries(self):
        text = (
            "Paragraph one is here.\n\n"
            "Paragraph two is here and is also reasonably short.\n\n"
            "Paragraph three rounds things out nicely."
        )
        result = recursive_chunks(text, chunk_size=40, overlap=5)

        # No chunk should wildly exceed chunk_size + overlap.
        for chunk in result:
            assert len(chunk) <= 40 + 5 + 10  # small slack for boundary text

        # All paragraphs' core content should be represented somewhere.
        joined = " ".join(result)
        assert "Paragraph one" in joined
        assert "Paragraph two" in joined
        assert "Paragraph three" in joined

    def test_handles_text_with_no_separators(self):
        # A long string with no spaces/newlines must still be split.
        text = "A" * 1000
        result = recursive_chunks(text, chunk_size=100, overlap=10)
        assert len(result) > 1
        for chunk in result:
            assert len(chunk) <= 100


class TestChunkTextDispatch:
    def test_dispatch_fixed(self):
        result = chunk_text("ABCDEFGHIJ", strategy="fixed")
        assert isinstance(result, list)

    def test_dispatch_recursive(self):
        result = chunk_text("Hello. World.", strategy="recursive")
        assert isinstance(result, list)

    def test_dispatch_unknown_strategy_raises(self):
        with pytest.raises(ValueError):
            chunk_text("text", strategy="banana")

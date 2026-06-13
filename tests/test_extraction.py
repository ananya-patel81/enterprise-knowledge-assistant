"""
test_extraction.py
===================

Unit tests for rag/extraction.py.
"""

import pytest

from rag.extraction import clean_text, extract_text


class TestCleanText:
    def test_collapses_multiple_blank_lines(self):
        text = "Line one\n\n\n\n\nLine two"
        cleaned = clean_text(text)
        assert "\n\n\n" not in cleaned
        assert "Line one" in cleaned and "Line two" in cleaned

    def test_strips_trailing_whitespace_per_line(self):
        text = "Line one   \nLine two\t\t"
        cleaned = clean_text(text)
        for line in cleaned.splitlines():
            assert line == line.rstrip()

    def test_collapses_multiple_spaces(self):
        text = "Too    many     spaces"
        cleaned = clean_text(text)
        assert "  " not in cleaned


class TestExtractText:
    def test_txt_file(self, tmp_path):
        file_path = tmp_path / "sample.txt"
        file_path.write_text("Hello world\n\n\nGoodbye")

        result = extract_text(str(file_path), "txt")
        assert "Hello world" in result
        assert "Goodbye" in result

    def test_md_file(self, tmp_path):
        file_path = tmp_path / "sample.md"
        file_path.write_text("# Title\n\nSome **bold** content.")

        result = extract_text(str(file_path), "md")
        assert "Title" in result
        assert "content" in result

    def test_unsupported_type_raises(self, tmp_path):
        file_path = tmp_path / "sample.docx"
        file_path.write_text("irrelevant")

        with pytest.raises(ValueError):
            extract_text(str(file_path), "docx")

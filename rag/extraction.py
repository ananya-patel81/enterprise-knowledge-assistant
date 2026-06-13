"""
extraction.py
=============

Extracts plain text from uploaded files (PDF, TXT, Markdown) and applies
light cleaning before chunking.

WHY a separate extraction step?
-----------------------------------
Different file formats store text differently (PDFs encode text as
positioned glyphs on a page; Markdown has formatting syntax). By
normalizing everything to plain text *before* chunking and embedding, the
rest of the pipeline (chunking, embeddings, retrieval) doesn't need to
know anything about file formats -- a clean separation of concerns.
"""

import re

from pypdf import PdfReader

from backend.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = {"pdf", "txt", "md"}


def extract_text(file_path: str, file_type: str) -> str:
    """Extract raw text from a file on disk.

    Args:
        file_path: Path to the uploaded file.
        file_type: One of "pdf", "txt", "md".

    Returns:
        The extracted (and lightly cleaned) text.

    Raises:
        ValueError: if `file_type` is not supported.
    """
    if file_type not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {file_type!r}")

    if file_type == "pdf":
        text = _extract_pdf(file_path)
    else:  # txt or md
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

    return clean_text(text)


def _extract_pdf(file_path: str) -> str:
    """Extract text from every page of a PDF and join with newlines."""
    reader = PdfReader(file_path)
    pages = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        pages.append(page_text)
    return "\n\n".join(pages)


def clean_text(text: str) -> str:
    """Apply light, format-agnostic cleaning.

    - Collapse runs of 3+ blank lines into a single blank line.
    - Strip trailing whitespace from each line.
    - Collapse multiple spaces into one.

    We deliberately keep cleaning minimal: aggressive cleaning (e.g.
    removing all newlines) can destroy the paragraph structure that
    `recursive_chunks` relies on.
    """
    # Strip trailing whitespace per line.
    lines = [line.rstrip() for line in text.splitlines()]
    text = "\n".join(lines)

    # Collapse 3+ consecutive newlines into 2 (i.e. one blank line).
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse multiple spaces/tabs into a single space.
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()

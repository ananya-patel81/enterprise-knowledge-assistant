"""
chunking.py
===========

Splits raw document text into smaller "chunks" before embedding.

WHY do we chunk at all?
------------------------
1. Embedding models and LLMs have limited context windows -- we cannot
   embed or feed an entire 50-page PDF in one go.
2. Smaller chunks produce more *focused* embeddings. A chunk about
   "refund policy" should not be diluted by also containing the
   "shipping policy" section -- otherwise a query about refunds may not
   rank it highly.
3. Citations become meaningful: we can point to the exact paragraph
   that answered the question, not just "somewhere in this 50-page PDF".

This module implements two strategies, as required by the project spec.

FIXED-SIZE CHUNKING
-------------------
Splits text into chunks of a fixed number of characters, with an overlap
between consecutive chunks.

Pros:
  - Extremely simple and predictable; easy to reason about memory/cost.
  - Works on any text, even text with no clear structure (logs, code).
Cons:
  - Can split a sentence (or even a word) in half, which can hurt both
    embedding quality and answer quality.

RECURSIVE CHUNKING
-------------------
Tries to split on "natural" boundaries first (paragraphs, then
sentences, then words), recursively, until each chunk is within
`chunk_size`. This is the strategy popularized by LangChain's
`RecursiveCharacterTextSplitter` (we re-implement a small version here
so the student fully understands -- and can explain -- the algorithm).

Pros:
  - Chunks tend to be semantically coherent (whole paragraphs/sentences).
  - Generally produces better retrieval quality for prose documents.
Cons:
  - Slightly more complex.
  - Chunk sizes vary more than fixed-size chunking.

Both strategies share the same overlap idea: overlap means consecutive
chunks share some text, which helps preserve context that spans a chunk
boundary (e.g. a sentence that starts at the end of chunk N and
continues into chunk N+1).
"""

from typing import List

from backend.config import settings


def fixed_size_chunks(text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
    """Split `text` into fixed-size character chunks with overlap.

    Args:
        text: The full document text.
        chunk_size: Number of characters per chunk. Defaults to
            `settings.chunk_size`.
        overlap: Number of overlapping characters between consecutive
            chunks. Defaults to `settings.chunk_overlap`.

    Returns:
        A list of text chunks.

    Example:
        >>> fixed_size_chunks("ABCDEFGHIJ", chunk_size=4, overlap=1)
        ['ABCD', 'DEFG', 'GHIJ']
    """
    chunk_size = chunk_size if chunk_size is not None else settings.chunk_size
    overlap = overlap if overlap is not None else settings.chunk_overlap

    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    text = text.strip()
    if not text:
        return []

    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(text), step):
        chunk = text[start:start + chunk_size]
        if chunk.strip():
            chunks.append(chunk)
        if start + chunk_size >= len(text):
            break

    return chunks


def recursive_chunks(text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
    """Split `text` recursively on natural boundaries.

    The algorithm tries separators in order of "size":
        1. "\\n\\n" (paragraph breaks)
        2. "\\n"   (line breaks)
        3. ". "    (sentence breaks)
        4. " "     (word breaks)
        5. ""      (last resort: hard character split)

    For each piece produced by splitting on a separator, if the piece
    still exceeds `chunk_size`, we recursively split it using the *next*
    separator. Pieces that fit are then greedily merged back together up
    to `chunk_size`, so we don't end up with hundreds of tiny chunks.

    Args:
        text: The full document text.
        chunk_size: Maximum number of characters per chunk.
        overlap: Number of overlapping characters between consecutive
            merged chunks.

    Returns:
        A list of text chunks.
    """
    chunk_size = chunk_size if chunk_size is not None else settings.chunk_size
    overlap = overlap if overlap is not None else settings.chunk_overlap

    text = text.strip()
    if not text:
        return []

    separators = ["\n\n", "\n", ". ", " ", ""]
    pieces = _split_recursive(text, separators, chunk_size)
    return _merge_pieces(pieces, chunk_size, overlap)


def _split_recursive(text: str, separators: List[str], chunk_size: int) -> List[str]:
    """Recursively split `text` until every piece is <= chunk_size."""
    if len(text) <= chunk_size:
        return [text]

    if not separators:
        # Last resort: hard split by character count.
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    sep, rest_separators = separators[0], separators[1:]

    if sep == "":
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    parts = text.split(sep)
    # Re-attach the separator (except possibly the last part) so we don't
    # lose punctuation/whitespace structure.
    parts = [p + sep if i < len(parts) - 1 else p for i, p in enumerate(parts)]

    results: List[str] = []
    for part in parts:
        if not part:
            continue
        if len(part) <= chunk_size:
            results.append(part)
        else:
            # This piece is still too big -- recurse with the next,
            # finer-grained separator.
            results.extend(_split_recursive(part, rest_separators, chunk_size))

    return results


def _merge_pieces(pieces: List[str], chunk_size: int, overlap: int) -> List[str]:
    """Greedily merge small pieces into chunks close to `chunk_size`,
    inserting `overlap` characters of context between consecutive
    chunks.
    """
    chunks: List[str] = []
    current = ""

    for piece in pieces:
        if len(current) + len(piece) <= chunk_size:
            current += piece
        else:
            if current.strip():
                chunks.append(current.strip())
            # Start the next chunk with the tail of the previous chunk
            # (overlap) plus the new piece. If the piece itself is
            # already a full chunk (e.g. from a hard character split
            # with no separators), skip the overlap so we don't exceed
            # chunk_size.
            if overlap > 0 and len(piece) < chunk_size:
                overlap_text = current[-overlap:]
            else:
                overlap_text = ""
            current = overlap_text + piece

    if current.strip():
        chunks.append(current.strip())

    return chunks


def chunk_text(text: str, strategy: str = None) -> List[str]:
    """Dispatch to the configured chunking strategy.

    Args:
        text: The full document text.
        strategy: "fixed" or "recursive". Defaults to
            `settings.chunk_strategy`.

    Returns:
        A list of text chunks.
    """
    strategy = strategy or settings.chunk_strategy
    if strategy == "fixed":
        return fixed_size_chunks(text)
    elif strategy == "recursive":
        return recursive_chunks(text)
    else:
        raise ValueError(f"Unknown chunk strategy: {strategy!r}")

"""
bm25_search.py
================

Implements BM25-based keyword retrieval as an alternative to vector
search, used in the "Retrieval Comparison" feature.

WHAT is BM25?
---------------
BM25 ("Best Match 25") is a classic information-retrieval ranking
function -- the algorithm behind traditional search engines (and a
refinement of TF-IDF). It scores how relevant a document is to a query
based on:

1. Term Frequency (TF): how often query words appear in the document
   (with diminishing returns -- the 10th occurrence of a word matters
   much less than the 2nd).
2. Inverse Document Frequency (IDF): words that appear in *few*
   documents are more informative than words that appear in *most*
   documents (e.g. "the" is in every document and is not useful for
   ranking; "depreciation" might appear in only a few).
3. Document length normalization: a word appearing once in a short
   document is weighted more heavily than once in a very long document.

VECTOR SEARCH vs BM25 -- WHY COMPARE THEM?
----------------------------------------------
- BM25 (lexical/keyword search) excels at exact-term matches: product
  codes, acronyms, names, numbers -- things embeddings can sometimes
  blur together.
- Vector search (semantic search) excels at *paraphrased* or
  conceptually-related queries, where the exact words don't match.

Providing both, side-by-side, lets the user (and, in an interview, the
student) directly *see* these trade-offs on real data -- which is far
more convincing than describing them abstractly. In production RAG
systems, this comparison motivates "hybrid retrieval" (combining both
scores), which is listed as a Future Improvement in the README.

IMPLEMENTATION NOTE
----------------------
We use the `rank_bm25` library's `BM25Okapi` implementation. Because the
corpus is small (at most a few thousand chunks for this project), we
simply rebuild the BM25 index from SQLite on each request. This keeps
the implementation simple; for a larger corpus, this index would be
cached/persisted like the FAISS index.
"""

from typing import List, Tuple

from rank_bm25 import BM25Okapi

from database import db


def _tokenize(text: str) -> List[str]:
    """Very simple whitespace + lowercase tokenizer.

    A production system might use a proper tokenizer (e.g. spaCy, NLTK)
    with stop-word removal and stemming, but a simple tokenizer is
    sufficient to demonstrate the BM25 algorithm and keeps dependencies
    minimal.
    """
    return text.lower().split()


def bm25_search(query: str, top_k: int = 4) -> List[Tuple[dict, float]]:
    """Run BM25 keyword search over all chunks in the database.

    Args:
        query: The user's question.
        top_k: Number of top results to return.

    Returns:
        A list of (chunk_row_as_dict, bm25_score) tuples, sorted by
        descending score. Returns an empty list if there are no chunks.
    """
    chunks = db.get_all_chunks()
    if not chunks:
        return []

    corpus = [_tokenize(row["content"]) for row in chunks]
    bm25 = BM25Okapi(corpus)

    tokenized_query = _tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    # Pair each chunk with its score, then sort descending.
    scored = list(zip(chunks, scores))
    scored.sort(key=lambda pair: pair[1], reverse=True)

    top_k = min(top_k, len(scored))
    return [(dict(row), float(score)) for row, score in scored[:top_k]]

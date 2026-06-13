"""
embeddings.py
=============

Generates vector embeddings for text using Sentence-Transformers.

WHAT is an embedding?
-----------------------
An embedding is a fixed-length vector of floating-point numbers (here,
384 dimensions) that represents the *meaning* of a piece of text.
Texts with similar meaning end up "close together" in this 384-
dimensional space (measured by cosine similarity / Euclidean distance),
even if they don't share the same words.

Example: "How do I reset my password?" and "I forgot my login
credentials" would have embeddings that are close together, even though
they share almost no words in common.

WHY do we need embeddings for RAG?
-------------------------------------
Traditional keyword search (e.g. SQL `LIKE` or BM25) matches on exact
words. Embedding-based ("semantic" / "vector") search matches on
*meaning*, which lets users phrase questions naturally and still find
relevant document chunks -- this is the core idea behind
Retrieval-Augmented Generation (RAG).

WHY Sentence-Transformers + `all-MiniLM-L6-v2` specifically?
----------------------------------------------------------------
- It's a small (~80MB), fast, CPU-friendly model that produces strong
  semantic embeddings for English text.
- Running locally means no per-call API cost and no external dependency
  for the embedding step (only the final LLM call needs an API key).
- 384-dimensional vectors are small enough that FAISS indexing and
  search are essentially instant for thousands of chunks -- perfect for
  a student project deployed on free-tier infrastructure.
"""

from functools import lru_cache
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from backend.config import settings
from backend.logger import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """Load (and cache) the Sentence-Transformers model.

    The model is loaded lazily and cached so that:
      - Importing this module doesn't trigger a slow model download/load.
      - The (somewhat expensive) model load happens only once per
        process, not once per request.
    """
    logger.info("Loading embedding model: %s", settings.embedding_model_name)
    return SentenceTransformer(settings.embedding_model_name)


def embed_texts(texts: List[str]) -> np.ndarray:
    """Embed a list of strings into a 2D numpy array of shape
    (len(texts), embedding_dim).

    The vectors are L2-normalized so that cosine similarity can be
    computed as a simple dot product (this is what
    `faiss.IndexFlatIP` -- Inner Product -- relies on; see
    vector_store.py).

    Args:
        texts: List of text chunks (or a single query wrapped in a
            list).

    Returns:
        A numpy array of shape (N, embedding_dim), dtype float32.
    """
    model = get_model()
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,  # unit-length vectors -> cosine == dot product
        show_progress_bar=False,
    )
    return embeddings.astype("float32")


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string, returning a 1D vector of shape
    (embedding_dim,).
    """
    return embed_texts([query])[0]

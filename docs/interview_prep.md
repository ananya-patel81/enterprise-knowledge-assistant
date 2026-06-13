# Interview Preparation — Enterprise Knowledge Assistant

This document contains 30 likely interview questions about this project,
with detailed answers. Read through this before any internship interview
where you list this project on your resume. The goal is not to
memorize — it's to genuinely understand each answer well enough to
explain it in your own words and handle follow-up questions.

---

## Section 1: Project Overview & Motivation

### 1. What does this project do, in one sentence?

It's a Retrieval-Augmented Generation (RAG) system: users upload
documents (PDF/TXT/Markdown), and can then ask natural-language
questions that are answered using the content of those documents, with
citations to the exact source chunk.

### 2. Why did you build this project?

Enterprise knowledge search is a real, widely-deployed use case (Glean,
Notion AI, Microsoft Copilot all do versions of this). I wanted to build
a project that demonstrates the full RAG pipeline end-to-end — document
ingestion, chunking, embeddings, vector search, and LLM generation —
using a small, understandable tech stack so I could defend every design
decision rather than relying on a framework that hides the details.

### 3. What problem does RAG solve that a plain LLM doesn't?

LLMs are trained on a static snapshot of data and have no knowledge of
private/internal documents. They also hallucinate — generating
plausible-sounding but false information — especially when asked about
things outside their training data. RAG solves both: it retrieves
relevant passages from the actual documents at query time and gives them
to the LLM as context, so answers are grounded in real, verifiable
text, with citations.

### 4. Who is the target user for this system?

Employees at a company who need to find information buried in policy
documents, manuals, or reports — e.g. "What's our WFH policy?" or "How
do I report a security incident?" — without manually searching through
PDFs.

---

## Section 2: Chunking

### 5. What is "chunking" and why is it necessary?

Chunking is splitting a large document into smaller pieces before
embedding. It's necessary because (a) embedding models and LLMs have
limited input sizes, and (b) smaller, focused chunks produce more
precise embeddings — a chunk about "refund policy" won't get diluted by
unrelated content from elsewhere in the document, so it ranks correctly
for refund-related queries.

### 6. Explain the difference between fixed-size and recursive chunking.

Fixed-size chunking splits text into chunks of N characters with some
overlap, regardless of sentence/paragraph boundaries — simple and
predictable, but can cut a sentence in half. Recursive chunking tries to
split on natural boundaries first (paragraphs, then sentences, then
words), recursing into smaller separators only if a piece is still too
large, then greedily merges small pieces back up toward the target size.
This produces more semantically coherent chunks at the cost of slightly
more complexity and variable chunk sizes.

### 7. What is "overlap" and why does it matter?

Overlap is the number of characters shared between consecutive chunks.
Without it, a sentence or idea that spans a chunk boundary could be
split such that neither chunk contains the full thought, hurting both
embedding quality and retrieval. Overlap ensures some context carries
over from one chunk to the next.

### 8. How did you choose your chunk size (500 characters)?

It's a balance: too small, and chunks lack enough context to be useful
on their own (and you generate many more embeddings/vectors); too large,
and a chunk may cover multiple topics, diluting its embedding and making
retrieval less precise, plus increasing the token cost when sent to the
LLM. 500 characters (~roughly 100 tokens) is a common starting point for
prose documents; it would be tuned based on evaluation in a real system.

### 9. What would happen if you used a chunk size of 10,000 characters?

Each chunk would likely span multiple topics/sections, so its embedding
would be a "blurred average" of all those topics — making it less likely
to rank highly for a specific question, and also more expensive (in LLM
tokens) once retrieved. You'd likely need fewer chunks overall (less
infrastructure), but retrieval *precision* would suffer.

---

## Section 3: Embeddings

### 10. What is an embedding, in plain terms?

An embedding is a list of numbers (a vector) that represents the meaning
of a piece of text. Texts with similar meaning produce vectors that are
close together in that numerical space, even if they use completely
different words.

### 11. How are embeddings generated in your project?

Using a Sentence-Transformers model (`all-MiniLM-L6-v2`), which runs
locally on CPU. Each chunk of text is passed through the model to
produce a 384-dimensional vector. Vectors are L2-normalized (scaled to
unit length) so that cosine similarity can be computed as a simple dot
product.

### 12. Why did you choose `all-MiniLM-L6-v2` specifically?

It's small (~80MB), fast on CPU, free to run (no API costs), and
produces strong general-purpose semantic embeddings for English text —
a good fit for a project that needs to run on free hosting tiers without
a GPU.

### 13. What's the difference between embedding a document and embedding a query?

Mechanically, nothing — the same model and function (`embed_texts`) is
used for both. Conceptually, the query embedding represents "what the
user is looking for" and document chunk embeddings represent "what each
piece of text is about." Retrieval finds chunk embeddings closest to the
query embedding.

### 14. What happens if you embed text in a different language than the documents?

`all-MiniLM-L6-v2` is primarily trained on English; cross-lingual
performance would degrade significantly. For a multilingual system,
you'd choose a multilingual embedding model (e.g.
`paraphrase-multilingual-MiniLM-L12-v2`) — this is a good "I know the
limitation and how I'd fix it" answer.

---

## Section 4: Vector Search / FAISS

### 15. What is FAISS and what role does it play here?

FAISS (Facebook AI Similarity Search) is a library for fast similarity
search over vectors. In this project, it stores the embedding vector for
every chunk and, given a query vector, returns the IDs of the most
similar chunks.

### 16. Explain cosine similarity.

Cosine similarity measures the angle between two vectors, ignoring their
magnitude — it answers "do these vectors point in the same direction?"
A value of 1 means identical direction (very similar meaning), 0 means
unrelated (orthogonal), and -1 means opposite. Because embeddings encode
meaning primarily through direction, cosine similarity is the standard
metric for comparing them.

### 17. Why does your code use `IndexFlatIP` (inner product) instead of an L2/Euclidean index?

Because the embeddings are normalized to unit length before being added
to the index, the inner product of two unit vectors is mathematically
equal to their cosine similarity. So `IndexFlatIP` on normalized vectors
gives us cosine similarity search "for free", which is simpler than
computing cosine similarity manually or using an L2 index and converting
distances.

### 18. What does "Flat" mean in `IndexFlatIP`, and what's the trade-off?

"Flat" means the index does an exhaustive (brute-force) comparison of
the query against every stored vector — O(N) per search. The trade-off
is search time grows linearly with the number of vectors. For this
project's scale (hundreds to low-thousands of chunks), this is still
sub-millisecond, so the simplicity is worth it. At millions of vectors,
you'd switch to an approximate index like `IndexHNSWFlat` or
`IndexIVFFlat`, trading a small amount of accuracy for much faster
search.

### 19. How would this system scale to millions of documents?

Several changes: (1) switch from `IndexFlatIP` to an approximate index
(HNSW or IVF) for sub-linear search time, (2) move FAISS to a dedicated
service or a managed vector DB if memory becomes a constraint (FAISS
indexes are in-memory), (3) move SQLite to PostgreSQL for concurrent
writes, (4) move document ingestion to an async background job
queue, and (5) potentially shard the index across multiple machines.

### 20. Why not just use a managed vector database like Pinecone from the start?

For a project of this scale, an in-process FAISS index has zero
infrastructure cost, zero network latency, and is simpler to deploy on
free hosting. A managed vector DB becomes valuable when you need
durability across restarts at scale, multi-node search, or built-in
metadata filtering at scale — none of which are needed here. (Note:
FAISS index + SQLite metadata is persisted to disk in this project, so
durability across restarts is already handled simply.)

---

## Section 5: Retrieval Comparison (Vector vs BM25)

### 21. What is BM25?

BM25 is a classic keyword-based ranking algorithm (an evolution of
TF-IDF) used by traditional search engines. It scores documents based on
term frequency (how often query words appear), inverse document
frequency (how rare/informative those words are across the corpus), and
document length normalization.

### 22. When would BM25 outperform vector search?

When the query contains exact terms that matter precisely — product
codes, acronyms, specific names, or numbers — where an embedding model
might treat semantically "close" but lexically different terms as more
similar than they should be, potentially missing an exact match.

### 23. When would vector search outperform BM25?

When the query is phrased differently than the document's wording —
e.g. "How do I get my money back?" vs. a document that says "Refund
Policy" — vector search can match based on meaning, while BM25 would
find zero keyword overlap.

### 24. What is "hybrid retrieval" and why didn't you implement it?

Hybrid retrieval combines vector and BM25 scores (e.g. via weighted sum
or Reciprocal Rank Fusion) to get the benefits of both. I didn't
implement it to keep the project focused and the comparison page
*explicit* — showing both result sets side-by-side is itself a strong
demonstration of understanding the trade-offs, and hybrid retrieval is
a natural "next step" I can discuss.

---

## Section 6: LLM & Prompting

### 25. How do you prevent the LLM from hallucinating?

By constructing a prompt that explicitly instructs the model to answer
*only* using the provided context chunks, and to say "I could not find
this in the provided documents" if the answer isn't present. This
"grounding" instruction, combined with actually providing relevant
context, is the core hallucination-reduction mechanism of RAG (though
it's not a 100% guarantee — LLMs can still occasionally ignore
instructions).

### 26. Why support both Gemini and OpenAI?

To demonstrate the adapter/strategy pattern — the rest of the codebase
calls a single `generate_answer()` function and doesn't know which
provider is used. Swapping providers is a one-line config change, not a
code change. Practically, it also gives flexibility based on which
API has available free credits.

### 27. What would you do if the LLM's answer contradicted the retrieved context?

In the current system, that would indicate either (a) the prompt's
grounding instruction wasn't strong enough, or (b) the retrieved context
wasn't actually relevant (a retrieval problem, not a generation
problem). I'd first check the retrieved chunks shown in the UI — if
they're irrelevant, the fix is in retrieval (better chunking, more
top-k, or reranking); if they're relevant but the LLM ignored them,
I'd strengthen the prompt or try a different/larger model.

---

## Section 7: System Design & Engineering

### 28. Walk me through what happens when a user uploads a PDF.

The file is sent to `POST /documents/upload`. The backend saves the raw
file to disk, extracts text using `pypdf`, cleans it (collapsing
whitespace), records a `documents` row in SQLite, splits the text into
chunks using the configured strategy (fixed or recursive), embeds all
chunks in one batch call to the Sentence-Transformers model, adds the
resulting vectors to the FAISS index, inserts a `chunks` row per chunk
(linking it to its FAISS id), updates the document's chunk count, and
persists the FAISS index to disk.

### 29. Walk me through what happens when a user asks a question.

The frontend sends `POST /ask` with the question and retrieval mode. The
backend embeds the question (if vector mode) or tokenizes it (if BM25
mode), retrieves the top-k most relevant chunks, resolves each chunk's
text and source document from SQLite, builds a prompt containing the
question plus the retrieved chunks as labeled "sources", sends it to the
configured LLM, logs the query (for analytics), and returns the answer
plus the retrieved chunks (so the frontend can display citations).

### 30. What would you change if you had another month to work on this?

I'd implement reranking (a cross-encoder model that re-scores the top-N
retrieved chunks for higher precision), add hybrid retrieval combining
vector and BM25 scores, move to PostgreSQL to support multiple
concurrent users, add basic authentication so different teams see only
their own documents, and add an evaluation harness (a set of test
questions with expected source chunks) to measure retrieval quality
quantitatively rather than just eyeballing results.

---

## Bonus: Quick-Fire Definitions

Be ready to define these in one sentence each:

- **RAG**: Retrieval-Augmented Generation — retrieving relevant context
  and giving it to an LLM to ground its answer.
- **Embedding**: A numerical vector representation of text's meaning.
- **Cosine similarity**: A measure of how similar two vectors' directions
  are, ignoring magnitude.
- **Chunking**: Splitting documents into smaller pieces before embedding.
- **BM25**: A keyword-based relevance ranking algorithm.
- **Vector store / index**: A data structure optimized for finding the
  most similar vectors to a query vector.
- **Citation/grounding**: Linking an LLM's answer back to the specific
  source text it was based on.
- **Hybrid retrieval**: Combining keyword and semantic search results.
- **Reranking**: A second-stage model that re-scores initially retrieved
  results for higher precision.

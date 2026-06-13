# Resume & Pitch Materials — Enterprise Knowledge Assistant

## Resume Bullet Points

Use 2–4 of these depending on space, tailoring verbs/metrics to the role
(SWE vs AI/ML internship).

1. **Built a full-stack Retrieval-Augmented Generation (RAG) system**
   (FastAPI + Streamlit + FAISS + SQLite) enabling natural-language Q&A
   over uploaded PDF/TXT/Markdown documents, with source citations for
   every answer.

2. **Designed and implemented two configurable document-chunking
   strategies** (fixed-size and recursive boundary-aware) and a vector
   search pipeline using Sentence-Transformers embeddings and FAISS
   cosine-similarity search.

3. **Implemented a retrieval comparison feature** benchmarking semantic
   (vector/FAISS) search against lexical (BM25) search on the same
   queries, surfacing trade-offs between the two retrieval strategies.

4. **Containerized the application with Docker and docker-compose**,
   set up a GitHub Actions CI pipeline running pytest on every push, and
   wrote unit tests covering chunking, vector search, and BM25 retrieval
   logic.

---

## Elevator Pitch (30 seconds)

"I built an AI-powered enterprise knowledge assistant — think of it as
an internal ChatGPT for company documents. Users upload PDFs or
Markdown files, and can then ask questions in plain English. The system
uses Retrieval-Augmented Generation: it finds the most relevant chunks
of the documents using vector search with FAISS, sends them to an LLM
as context, and returns an answer with citations showing exactly which
document and chunk it came from — so you can verify it's not making
things up."

---

## 2-Minute Project Explanation

"The core problem is that companies have huge amounts of internal
documentation — policies, manuals, reports — that's hard to search
because employees phrase questions differently than how the documents
are written. A plain keyword search often misses relevant content, and
asking an LLM directly doesn't work because it has no knowledge of
private documents and tends to hallucinate.

My solution implements the full RAG pipeline. On the ingestion side,
when a document is uploaded, I extract its text, clean it, and split it
into chunks — I implemented two chunking strategies, fixed-size and a
recursive one that tries to split on paragraph and sentence boundaries
first, so chunks stay semantically coherent. Each chunk is then embedded
into a 384-dimensional vector using a local Sentence-Transformers model,
and stored in a FAISS index for fast similarity search, with the chunk
text and metadata stored in SQLite.

On the query side, when a user asks a question, I embed the question
the same way, search FAISS for the most similar chunks using cosine
similarity, and build a prompt that includes those chunks as context.
The LLM — I support both Gemini and OpenAI — is instructed to answer
only from that context and to say if it can't find an answer, which
keeps the responses grounded. The UI shows the answer alongside the
exact source chunks used, as citations.

I also built a retrieval-comparison page that runs the same query
through vector search and through BM25 — a classic keyword-ranking
algorithm — side by side, which makes the trade-offs between semantic
and lexical search very concrete. And there's a small analytics
dashboard tracking documents, chunks, queries, and latency.

Everything's containerized with Docker, has a CI pipeline running unit
tests on the core retrieval and chunking logic, and the codebase is
deliberately organized so the RAG logic is completely independent of the
web framework — it's a separate `rag` package that could be reused
anywhere."

---

## 5-Minute Deep Technical Explanation

Use the above 2-minute explanation as the foundation, then expand on
these areas as follow-ups arise:

**On chunking**: "I implemented fixed-size chunking as a baseline — it
splits text into N-character windows with overlap, which is simple but
can cut sentences in half. Recursive chunking is more involved: it
tries separators in order of size — paragraph breaks, then line breaks,
then sentence breaks, then word breaks — recursively splitting any piece
that's still too large, then greedily merging the resulting pieces back
up toward the target chunk size. The overlap is added by carrying the
tail of the previous chunk into the next one, but I had to special-case
the situation where a piece is itself already at the maximum chunk
size — like a long string with no spaces — to avoid the overlap pushing
it over the limit. That edge case is actually caught by one of my unit
tests."

**On embeddings and FAISS**: "I chose `all-MiniLM-L6-v2` because it's
small enough to run on CPU with no GPU and no API cost, while still
giving good semantic embeddings. I normalize the embeddings to unit
length, which means I can use FAISS's `IndexFlatIP` — inner product —
as an exact cosine-similarity search, since the inner product of two
unit vectors equals their cosine similarity. `Flat` means it's an exact,
brute-force search rather than approximate, which is fine at the scale
of a few thousand chunks — sub-millisecond — but at millions of vectors
I'd switch to something like HNSW for sub-linear search time."

**On the LLM integration**: "I built a small adapter layer so the rest
of the app calls one function, `generate_answer`, regardless of whether
Gemini or OpenAI is configured — that's the strategy pattern, and it
means swapping providers is a config change, not a code change. The
prompt explicitly tells the model to answer only from the provided
context and to say it doesn't know if the answer isn't there, which is
the main lever for reducing hallucination in RAG systems."

**On retrieval comparison**: "BM25 is a TF-IDF-based ranking algorithm
that scores documents by how often query terms appear, weighted by how
rare those terms are across the whole corpus, normalized by document
length. It's purely lexical — no understanding of meaning — so it's
great for exact terms like product codes but misses paraphrases. Vector
search is the opposite: great for paraphrases, but can sometimes overlook
an exact, important keyword match. Showing both side by side on the same
query makes that trade-off tangible, and naturally leads into discussing
hybrid retrieval as a future improvement."

**On architecture/engineering**: "I deliberately kept the `rag` package
free of any FastAPI or Streamlit imports — it's pure Python functions
operating on text, vectors, and dicts. `backend/main.py` is a thin
HTTP layer over it, and `frontend/app.py` is a thin UI layer that only
talks to the backend over HTTP. That separation means the RAG logic is
independently testable — which is why my pytest suite can test chunking
and FAISS search without spinning up a web server — and could be reused
in, say, a CLI tool or a different frontend without changes."

---

## Anticipated Tough Follow-Ups

- *"Have you measured retrieval quality?"* — Be honest: "Not with a
  formal evaluation set yet — that's one of my planned next steps, an
  evaluation harness with test questions and expected source chunks to
  measure recall@k."
- *"What happens with a 500-page PDF?"* — "It would produce many chunks
  and take longer to embed and index on upload, since it's currently
  synchronous. For very large documents, I'd move ingestion to a
  background task queue."
- *"Is this production-ready?"* — "It's a focused portfolio
  implementation of the core RAG pattern. For production, I'd add
  authentication, PostgreSQL for concurrency, reranking, and an
  approximate FAISS index for scale — all listed explicitly as Future
  Improvements in the README."

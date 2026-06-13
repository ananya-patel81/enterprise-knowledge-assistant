# Implementation Plan — Enterprise Knowledge Assistant

This is the phased build plan used to construct this project. It's
useful both as a roadmap if you're building this yourself from scratch,
and as material for interview discussions about how you approach
breaking down a project.

The plan is split into 7 phases, each independently runnable/testable —
a deliberate choice so that at every stage there's a working (if
incomplete) system, rather than a big-bang integration at the end.

---

## Phase 0 — Project Scaffolding (Day 1)

**Goal:** A repository that runs, even if it does nothing useful yet.

- [ ] Create the directory structure (`backend/`, `frontend/`, `rag/`,
      `database/`, `tests/`, `docs/`, `assets/`).
- [ ] Add `requirements.txt`, `.env.example`, `.gitignore`.
- [ ] Add `backend/config.py` (centralized settings) and
      `backend/logger.py` (logging setup).
- [ ] Stand up a minimal FastAPI app with a `/health` endpoint.
- [ ] Stand up a minimal Streamlit page that calls `/health`.

**Checkpoint:** `uvicorn backend.main:app` and `streamlit run
frontend/app.py` both start; Streamlit shows the health check result.

---

## Phase 1 — Document Storage & Database (Day 1–2)

**Goal:** Documents can be recorded with metadata.

- [ ] Write `database/schema.sql` (documents, chunks, query_log tables).
- [ ] Write `database/db.py` with `init_db()`, `insert_document()`,
      `list_documents()`, etc.
- [ ] Add `POST /documents/upload` (saves file to disk, inserts a
      `documents` row — no text extraction yet).
- [ ] Add `GET /documents`.
- [ ] Streamlit "Upload Documents" page: file uploader + table of
      uploaded documents.

**Checkpoint:** Upload a file via Streamlit, see it listed with
filename/upload date/type.

---

## Phase 2 — Text Extraction, Cleaning & Chunking (Day 2–3)

**Goal:** Uploaded documents are converted into clean text chunks.

- [ ] Write `rag/extraction.py`: `extract_text()` for PDF (pypdf), TXT,
      and MD; `clean_text()` for whitespace normalization.
- [ ] Write `rag/chunking.py`: implement `fixed_size_chunks()` first
      (simpler), then `recursive_chunks()`.
- [ ] Write `tests/test_chunking.py` and `tests/test_extraction.py`
      covering edge cases (empty text, short text, no separators, etc.)
- [ ] Wire extraction + chunking into the upload endpoint (still no
      embeddings yet — just print/log chunk counts).

**Checkpoint:** `pytest tests/test_chunking.py
tests/test_extraction.py` passes. Uploading a document logs the correct
number of chunks for both strategies (toggle via `.env`).

---

## Phase 3 — Embeddings & Vector Search (Day 3–5)

**Goal:** Chunks are embedded and searchable by similarity.

- [ ] Write `rag/embeddings.py`: load Sentence-Transformers model
      (cached), `embed_texts()`, `embed_query()`.
- [ ] Write `rag/vector_store.py`: `VectorStore` wrapping
      `faiss.IndexFlatIP`, with `add()`, `search()`, `save()`/load.
- [ ] Write `tests/test_vector_store.py` using synthetic vectors (no
      need to load the real embedding model in tests — fast & isolated).
- [ ] Update `chunks` table inserts to include `faiss_id`.
- [ ] Finish the upload pipeline: extract → chunk → embed → FAISS.add →
      SQLite insert → FAISS.save.

**Checkpoint:** After uploading a document, `data/faiss.index` and
`data/knowledge.db` both contain the expected number of entries.

---

## Phase 4 — RAG Question Answering (Day 5–7)

**Goal:** Users can ask questions and get grounded, cited answers.

- [ ] Write `rag/llm.py`: prompt construction (`_build_prompt`) +
      provider adapters for Gemini and OpenAI.
- [ ] Write `rag/qa_pipeline.py`: `answer_question()` — embed query,
      FAISS search, resolve chunks via SQLite, call LLM, log query.
- [ ] Add `POST /ask` endpoint.
- [ ] Streamlit "Ask a Question" page: question input, answer display,
      expandable citations showing source document + chunk number +
      retrieved text.

**Checkpoint:** Upload `assets/sample_docs/employee_handbook.md`, ask
"How many days of annual leave do I get?", get an answer citing the
correct chunk.

---

## Phase 5 — Retrieval Comparison (BM25) (Day 7–8)

**Goal:** Side-by-side comparison of vector vs. keyword retrieval.

- [ ] Write `rag/bm25_search.py`: `bm25_search()` using `rank_bm25`,
      pulling the full corpus from SQLite via `db.get_all_chunks()`.
- [ ] Write `tests/test_bm25_search.py` (mock `db.get_all_chunks`).
- [ ] Add `POST /retrieval/compare` returning both vector and BM25
      results for the same query.
- [ ] Streamlit "Retrieval Comparison" page: two-column layout showing
      both result sets with scores.

**Checkpoint:** A query like "password reset" returns the IT security
chunk highly-ranked by BM25 (exact keyword match) and a comparable rank
from vector search (semantic match).

---

## Phase 6 — Analytics, Docker, CI, Docs (Day 8–10)

**Goal:** Production-readiness polish and resume/interview materials.

- [ ] Add `query_log` writes in `answer_question()`; add `GET
      /analytics` and `db.get_analytics()`.
- [ ] Streamlit "Analytics" page: metric cards + recent queries table.
- [ ] Write `Dockerfile`, `Dockerfile.frontend`, `docker-compose.yml`.
- [ ] Write `.github/workflows/ci.yml` (pytest on push/PR).
- [ ] Write `README.md` (architecture diagrams, design decisions,
      install/usage instructions).
- [ ] Write `docs/interview_prep.md` and `docs/resume_points.md`.

**Checkpoint:** `docker compose up --build` brings up both services;
`pytest tests/` passes; GitHub Actions runs green on push.

---

## Suggested Time Allocation (for a 2-week build)

| Phase | Days | Focus |
|-------|------|-------|
| 0 | 1 | Scaffolding |
| 1 | 1–2 | Storage & DB |
| 2 | 2–3 | Extraction & chunking |
| 3 | 3–5 | Embeddings & FAISS |
| 4 | 5–7 | RAG Q&A |
| 5 | 7–8 | Retrieval comparison |
| 6 | 8–10 | Analytics, Docker, CI, docs |
| Buffer | 10–14 | Testing, polish, screenshots, deployment |

---

## Risk Areas & Mitigations

- **PDF text extraction quality varies** (scanned PDFs with no
  selectable text won't extract). Mitigation: document this limitation
  in the README; stick to text-based PDFs for the demo/sample dataset.
- **LLM API costs/rate limits** during development. Mitigation: cache
  test prompts manually during development; use Gemini's free tier;
  keep `top_k` small to limit token usage.
- **FAISS index growing unbounded in memory.** Mitigation: documented
  explicitly as a scaling consideration (see README Future
  Improvements and interview prep Q19).

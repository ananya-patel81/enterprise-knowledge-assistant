"""
main.py
=======

FastAPI application exposing the Enterprise Knowledge Assistant API.

WHY FastAPI?
---------------
- Automatic interactive API docs (Swagger UI at /docs) -- great for
  demoing the project and for the frontend team (Streamlit) to discover
  endpoints.
- Built-in request validation via Pydantic models -> fewer bugs, clear
  contracts between frontend and backend.
- Async support out of the box, which matters if this were scaled to
  handle concurrent requests (see Future Improvements in README).

ENDPOINT OVERVIEW
--------------------
  POST /documents/upload        - upload + ingest a document
  GET  /documents                - list uploaded documents
  POST /ask                      - ask a question (RAG pipeline)
  POST /retrieval/compare        - compare vector vs BM25 retrieval
  GET  /analytics                - dashboard stats
  GET  /health                   - simple health check
"""

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.logger import get_logger
from database import db
from rag import qa_pipeline
from rag.extraction import SUPPORTED_EXTENSIONS
from typing import Optional

logger = get_logger(__name__)

app = FastAPI(
    title="Enterprise Knowledge Assistant API",
    description=(
        "A Retrieval-Augmented Generation (RAG) API for asking questions "
        "over uploaded company documents."
    ),
    version="1.0.0",
)

# Allow the Streamlit frontend (running on a different port/host) to
# call this API. In a real enterprise deployment this would be locked
# down to specific origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    """Initialize the SQLite database and load (or create) the FAISS
    index when the server starts.
    """
    db.init_db()
    qa_pipeline.get_vector_store()
    logger.info("Application startup complete.")


# ----------------------------------------------------------------------
# Pydantic request/response models
# ----------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str
    retrieval_mode: str = "vector"  # "vector" | "bm25"
    top_k: Optional[int] = None


class CompareRequest(BaseModel):
    question: str
    top_k: Optional[int] = None


# ----------------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)) -> dict:
    """Upload and ingest a document (PDF, TXT, or Markdown).

    The file is saved, text is extracted, split into chunks, embedded,
    and indexed in FAISS -- all synchronously. For larger files this
    would be moved to a background task/queue (see Future Improvements).
    """
    extension = file.filename.rsplit(".", 1)[-1].lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '.{extension}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}",
        )

    file_bytes = await file.read()
    try:
        result = qa_pipeline.ingest_document(file_bytes, file.filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return result


@app.get("/documents")
def list_documents() -> list[dict]:
    """Return metadata for all uploaded documents."""
    return [dict(row) for row in db.list_documents()]


@app.post("/ask")
def ask_question(request: AskRequest) -> dict:
    """Run the full RAG pipeline for a question."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    try:
        return qa_pipeline.answer_question(
            question=request.question,
            retrieval_mode=request.retrieval_mode,
            top_k=request.top_k,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        # Raised by rag/llm.py for missing API keys / provider errors.
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/retrieval/compare")
def compare_retrieval(request: CompareRequest) -> dict:
    """Return both vector-search and BM25 results for the same query,
    without calling the LLM. Powers the Retrieval Comparison page.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    return qa_pipeline.compare_retrieval(request.question, top_k=request.top_k)


@app.get("/analytics")
def analytics() -> dict:
    """Return aggregate stats for the analytics dashboard."""
    return db.get_analytics()

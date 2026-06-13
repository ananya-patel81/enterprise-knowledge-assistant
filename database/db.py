"""
db.py
=====

Thin data-access layer over SQLite.

WHY a hand-written DAL instead of an ORM (e.g. SQLAlchemy)?
-------------------------------------------------------------
For a project of this size, raw SQL via the stdlib `sqlite3` module is:
  - Easier to explain line-by-line in an interview (no ORM "magic").
  - Zero extra dependencies.
  - Plenty fast for the data volumes this app handles.

If the project grew (multiple users, complex relationships), migrating
to SQLAlchemy + Alembic migrations would be a natural "Future
Improvement" (see README).
"""

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator, Optional

from backend.config import settings
from backend.logger import get_logger

logger = get_logger(__name__)

SCHEMA_PATH = "database/schema.sql"


def init_db() -> None:
    """Create all tables if they do not already exist.

    Called once at application startup (see backend/main.py).
    """
    with get_connection() as conn:
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            conn.executescript(f.read())
    logger.info("Database initialized at %s", settings.sqlite_path)


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    """Yield a SQLite connection with row access by column name.

    Using a context manager guarantees the connection is closed (and
    committed) even if an exception occurs -- a common interview talking
    point about resource management.
    """
    conn = sqlite3.connect(settings.sqlite_path)
    conn.row_factory = sqlite3.Row
    # Enforce foreign key constraints (off by default in SQLite).
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ----------------------------------------------------------------------
# Documents
# ----------------------------------------------------------------------

def insert_document(filename: str, file_type: str, char_count: int) -> int:
    """Insert a new document row and return its generated id."""
    with get_connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO documents (filename, file_type, upload_date, num_chunks, char_count)
            VALUES (?, ?, ?, 0, ?)
            """,
            (filename, file_type, datetime.now(timezone.utc).isoformat(), char_count),
        )
        return cur.lastrowid


def update_document_chunk_count(document_id: int, num_chunks: int) -> None:
    with get_connection() as conn:
        conn.execute(
            "UPDATE documents SET num_chunks = ? WHERE id = ?",
            (num_chunks, document_id),
        )


def list_documents() -> list[sqlite3.Row]:
    with get_connection() as conn:
        return conn.execute(
            "SELECT * FROM documents ORDER BY upload_date DESC"
        ).fetchall()


# ----------------------------------------------------------------------
# Chunks
# ----------------------------------------------------------------------

def insert_chunk(document_id: int, chunk_index: int, content: str, faiss_id: int) -> int:
    with get_connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO chunks (document_id, chunk_index, content, faiss_id)
            VALUES (?, ?, ?, ?)
            """,
            (document_id, chunk_index, content, faiss_id),
        )
        return cur.lastrowid


def get_chunk_by_faiss_id(faiss_id: int) -> Optional[sqlite3.Row]:
    with get_connection() as conn:
        return conn.execute(
            """
            SELECT chunks.*, documents.filename AS document_name
            FROM chunks
            JOIN documents ON documents.id = chunks.document_id
            WHERE chunks.faiss_id = ?
            """,
            (faiss_id,),
        ).fetchone()


def get_all_chunks() -> list[sqlite3.Row]:
    """Return every chunk with its parent document name.

    Used by the BM25 retriever, which needs the full corpus in memory.
    """
    with get_connection() as conn:
        return conn.execute(
            """
            SELECT chunks.*, documents.filename AS document_name
            FROM chunks
            JOIN documents ON documents.id = chunks.document_id
            ORDER BY chunks.faiss_id ASC
            """
        ).fetchall()


def count_chunks() -> int:
    with get_connection() as conn:
        return conn.execute("SELECT COUNT(*) AS c FROM chunks").fetchone()["c"]


# ----------------------------------------------------------------------
# Query log (analytics)
# ----------------------------------------------------------------------

def log_query(question: str, retrieval_mode: str, latency_ms: float) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO query_log (question, retrieval_mode, latency_ms, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (question, retrieval_mode, latency_ms, datetime.now(timezone.utc).isoformat()),
        )


def get_analytics() -> dict:
    """Aggregate stats used by the Streamlit analytics dashboard."""
    with get_connection() as conn:
        num_documents = conn.execute("SELECT COUNT(*) AS c FROM documents").fetchone()["c"]
        num_chunks = conn.execute("SELECT COUNT(*) AS c FROM chunks").fetchone()["c"]
        num_queries = conn.execute("SELECT COUNT(*) AS c FROM query_log").fetchone()["c"]
        avg_latency = conn.execute(
            "SELECT AVG(latency_ms) AS a FROM query_log"
        ).fetchone()["a"]

        recent_queries = conn.execute(
            "SELECT question, retrieval_mode, latency_ms, created_at "
            "FROM query_log ORDER BY id DESC LIMIT 10"
        ).fetchall()

    return {
        "num_documents": num_documents,
        "num_chunks": num_chunks,
        "num_queries": num_queries,
        "avg_latency_ms": round(avg_latency, 2) if avg_latency else 0.0,
        "recent_queries": [dict(r) for r in recent_queries],
    }

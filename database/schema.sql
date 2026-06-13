-- schema.sql
-- ===========
--
-- WHY SQLite?
-- -----------
-- SQLite is a zero-configuration, file-based relational database. For a
-- single-user / small-scale knowledge assistant it gives us:
--   - Real SQL (joins, indexes, constraints) for structured metadata.
--   - No separate database server to install, configure, or deploy.
--   - A single file that can be backed up or shipped inside a Docker
--     volume.
--
-- The actual document TEXT and VECTOR embeddings live in FAISS + on
-- disk; SQLite stores the *metadata* that ties everything together
-- (which document a chunk came from, when it was uploaded, query logs
-- for analytics, etc.)

-- Documents uploaded by the user.
CREATE TABLE IF NOT EXISTS documents (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    filename      TEXT NOT NULL,
    file_type     TEXT NOT NULL,            -- pdf | txt | md
    upload_date   TEXT NOT NULL,            -- ISO-8601 timestamp
    num_chunks    INTEGER NOT NULL DEFAULT 0,
    char_count    INTEGER NOT NULL DEFAULT 0
);

-- Individual text chunks produced from documents. Each chunk corresponds
-- 1:1 with a vector stored in the FAISS index (faiss_id is the row's
-- position in the FAISS index).
CREATE TABLE IF NOT EXISTS chunks (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id   INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index   INTEGER NOT NULL,         -- order within the document
    content       TEXT NOT NULL,
    faiss_id      INTEGER NOT NULL UNIQUE   -- position in FAISS index
);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);

-- Log of every question asked, used for the analytics dashboard and for
-- demonstrating "observability" thinking in interviews.
CREATE TABLE IF NOT EXISTS query_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    question        TEXT NOT NULL,
    retrieval_mode  TEXT NOT NULL,          -- "vector" | "bm25"
    latency_ms      REAL NOT NULL,
    created_at      TEXT NOT NULL
);

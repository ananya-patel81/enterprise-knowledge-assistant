"""
config.py
=========

Centralized configuration for the Enterprise Knowledge Assistant.

WHY a config module?
---------------------
Hard-coding values (chunk size, model names, file paths) throughout the
codebase makes the system brittle and hard to tune. By centralizing all
"knobs" in one place, we can:

1. Change behaviour (e.g. chunk size, embedding model) without touching
   business logic.
2. Keep secrets (API keys) out of source code via environment variables.
3. Make the system easier to reason about in an interview: "Where does
   X come from?" -> "config.py".

All values can be overridden via a `.env` file (see `.env.example`).
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load variables from a local .env file if present. In production
# (Docker/Render/etc.) these would instead be set as real environment
# variables, but python-dotenv makes local development painless.
load_dotenv()


@dataclass(frozen=True)
class Settings:
    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    # Root directory where all persistent data (uploaded files, the
    # SQLite DB, and the FAISS index) is stored. Keeping everything under
    # one "data" directory makes it trivial to mount as a Docker volume.
    data_dir: str = os.getenv("DATA_DIR", "data")
    uploads_dir: str = field(init=False)
    sqlite_path: str = field(init=False)
    faiss_index_path: str = field(init=False)
    faiss_metadata_path: str = field(init=False)

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------
    # CHUNK_STRATEGY controls how documents are split before embedding.
    #   "fixed"     -> fixed-size character chunks with overlap.
    #   "recursive" -> recursively split on paragraph/sentence boundaries
    #                   until chunks fit within chunk_size.
    chunk_strategy: str = os.getenv("CHUNK_STRATEGY", "recursive")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------
    # We use a small, fast Sentence-Transformers model. It runs on CPU,
    # which keeps the project deployable on free-tier hosting (no GPU
    # required) -- an important consideration when defending design
    # choices in an interview.
    embedding_model_name: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "384"))

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    top_k: int = int(os.getenv("TOP_K", "4"))

    # ------------------------------------------------------------------
    # LLM
    # ------------------------------------------------------------------
    # Supports either Gemini or OpenAI. Only the configured provider's
    # API key needs to be set.
    llm_provider: str = os.getenv("LLM_PROVIDER", "gemini")  # "gemini" | "openai"
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    def __post_init__(self):
        # dataclass is frozen, so we use object.__setattr__ to set the
        # derived paths after the dataclass initializes.
        object.__setattr__(self, "uploads_dir", os.path.join(self.data_dir, "uploads"))
        object.__setattr__(self, "sqlite_path", os.path.join(self.data_dir, "knowledge.db"))
        object.__setattr__(self, "faiss_index_path", os.path.join(self.data_dir, "faiss.index"))
        object.__setattr__(self, "faiss_metadata_path", os.path.join(self.data_dir, "faiss_meta.pkl"))

        os.makedirs(self.uploads_dir, exist_ok=True)


# Single shared settings instance, imported everywhere as:
#   from backend.config import settings
settings = Settings()

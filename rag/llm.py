"""
llm.py
======

Wraps calls to either the Gemini API or the OpenAI API behind one
function: `generate_answer(question, context_chunks)`.

WHY support both Gemini and OpenAI?
---------------------------------------
- Gives the student flexibility based on which free-tier credits they
  have available (Gemini's API has a generous free tier, which is
  useful for a project that must run on a student budget).
- Demonstrates the "adapter" / "strategy" design pattern: the rest of
  the codebase (rag/qa_pipeline.py) doesn't care which provider is used
  -- it just calls `generate_answer(...)`. Swapping providers is a
  config change (`LLM_PROVIDER` in .env), not a code change. This is a
  good interview talking point about extensibility.

PROMPT DESIGN
----------------
The prompt explicitly instructs the model to:
  1. Answer **only** using the provided context.
  2. Say "I don't know" if the answer isn't in the context.
  3. Avoid hallucination by not relying on its own world knowledge.

This "grounding" instruction is the core of what makes RAG reduce
hallucinations compared to asking an LLM a question directly.
"""

from typing import List

from backend.config import settings
from backend.logger import get_logger

logger = get_logger(__name__)


SYSTEM_PROMPT = (
    "You are an enterprise knowledge assistant. Answer the user's question "
    "using ONLY the information provided in the context below. "
    "If the answer cannot be found in the context, say "
    "\"I could not find this in the provided documents.\" "
    "Do not use any outside knowledge. Be concise and cite which "
    "document/chunk numbers you used."
)


def _build_prompt(question: str, context_chunks: List[dict]) -> str:
    """Assemble the final prompt sent to the LLM.

    Args:
        question: The user's question.
        context_chunks: List of dicts, each with keys
            "document_name", "chunk_index", and "content".

    Returns:
        A formatted prompt string.
    """
    context_blocks = []
    for i, chunk in enumerate(context_chunks, start=1):
        context_blocks.append(
            f"[Source {i}: {chunk['document_name']}, chunk #{chunk['chunk_index']}]\n"
            f"{chunk['content']}"
        )
    context_text = "\n\n".join(context_blocks)

    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"--- CONTEXT ---\n{context_text}\n\n"
        f"--- QUESTION ---\n{question}\n\n"
        f"--- ANSWER ---"
    )


def generate_answer(question: str, context_chunks: List[dict]) -> str:
    """Generate an answer to `question` grounded in `context_chunks`.

    Dispatches to the configured LLM provider (`settings.llm_provider`).

    Args:
        question: The user's question.
        context_chunks: Retrieved chunks (see _build_prompt).

    Returns:
        The LLM's text response.
    """
    prompt = _build_prompt(question, context_chunks)

    if settings.llm_provider == "gemini":
        return _call_gemini(prompt)
    elif settings.llm_provider == "openai":
        return _call_openai(prompt)
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {settings.llm_provider!r}")


def _call_gemini(prompt: str) -> str:
    """Call the Gemini API. Imports the SDK lazily so the rest of the
    app works even if `google-generativeai` isn't installed and OpenAI
    is used instead.
    """
    import google.generativeai as genai

    # DEBUG: Log the settings to diagnose configuration issues
    logger.debug(f"DEBUG: GEMINI_API_KEY present: {bool(settings.gemini_api_key)}")
    logger.debug(f"DEBUG: GEMINI_API_KEY value (first 20 chars): {settings.gemini_api_key[:20] if settings.gemini_api_key else 'NOT SET'}")
    logger.debug(f"DEBUG: settings.gemini_model = {settings.gemini_model}")
    
    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is not set. See .env.example")

    genai.configure(api_key=settings.gemini_api_key)
    logger.debug(f"DEBUG: Creating GenerativeModel with model name: '{settings.gemini_model}'")
    model = genai.GenerativeModel(settings.gemini_model)
    logger.debug(f"DEBUG: GenerativeModel created successfully")

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as exc:
        logger.exception("Gemini API call failed")
        raise RuntimeError(f"Gemini API call failed: {exc}") from exc


def _call_openai(prompt: str) -> str:
    """Call the OpenAI API. Imports the SDK lazily."""
    from openai import OpenAI

    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. See .env.example")

    client = OpenAI(api_key=settings.openai_api_key)

    try:
        response = client.chat.completions.create(
            model=settings.openai_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return response.choices[0].message.content
    except Exception as exc:
        logger.exception("OpenAI API call failed")
        raise RuntimeError(f"OpenAI API call failed: {exc}") from exc

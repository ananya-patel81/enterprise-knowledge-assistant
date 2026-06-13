"""
app.py
======

Streamlit frontend for the Enterprise Knowledge Assistant.

WHY Streamlit?
-----------------
- Lets a backend-focused student build a usable, presentable UI in a
  few hundred lines of pure Python -- no separate JS/React build step.
- Streamlit apps are trivial to deploy on free platforms (Streamlit
  Community Cloud, HuggingFace Spaces, Render), which matters for the
  demo link on a resume.

ARCHITECTURE NOTE
--------------------
This frontend is a thin client: it holds NO business logic. Every
action (upload, ask, compare, analytics) is an HTTP call to the FastAPI
backend (`backend/main.py`). This separation means:
  - The same backend could power a different frontend (mobile app, CLI,
    Slack bot) without changes.
  - It's a clean, classic "client/server" architecture that's easy to
    draw and explain in a system-design interview.
"""

import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Enterprise Knowledge Assistant", layout="wide")

st.sidebar.title("📚 Knowledge Assistant")
page = st.sidebar.radio(
    "Navigate",
    ["Ask a Question", "Upload Documents", "Retrieval Comparison", "Analytics"],
)

st.sidebar.markdown("---")
st.sidebar.caption(f"Backend: `{API_URL}`")


# ----------------------------------------------------------------------
# Page: Ask a Question
# ----------------------------------------------------------------------

if page == "Ask a Question":
    st.title("Ask a Question About Your Documents")
    st.caption(
        "Questions are answered using Retrieval-Augmented Generation (RAG): "
        "relevant chunks are retrieved from your uploaded documents and "
        "passed to the LLM as context."
    )

    retrieval_mode = st.selectbox(
        "Retrieval mode",
        ["vector", "bm25"],
        help=(
            "'vector' uses semantic (embedding) search via FAISS. "
            "'bm25' uses classic keyword search. "
            "See the 'Retrieval Comparison' page to compare both."
        ),
    )
    question = st.text_input("Your question", placeholder="e.g. What is the refund policy?")

    if st.button("Ask", type="primary") and question:
        with st.spinner("Retrieving context and generating answer..."):
            try:
                resp = requests.post(
                    f"{API_URL}/ask",
                    json={"question": question, "retrieval_mode": retrieval_mode},
                    timeout=60,
                )
                resp.raise_for_status()
                data = resp.json()
            except requests.RequestException as exc:
                st.error(f"Request failed: {exc}")
                data = None

        if data:
            st.subheader("Answer")
            st.write(data["answer"])
            st.caption(f"Retrieval mode: {data['retrieval_mode']} | Latency: {data['latency_ms']} ms")

            st.subheader("Sources / Citations")
            if not data["retrieved_chunks"]:
                st.info("No matching chunks were found. Try uploading documents first.")
            for i, chunk in enumerate(data["retrieved_chunks"], start=1):
                with st.expander(
                    f"Source {i}: {chunk['document_name']} (chunk #{chunk['chunk_index']}, "
                    f"score={chunk['score']})"
                ):
                    st.write(chunk["content"])


# ----------------------------------------------------------------------
# Page: Upload Documents
# ----------------------------------------------------------------------

elif page == "Upload Documents":
    st.title("Upload Documents")
    st.caption("Supported formats: PDF, TXT, Markdown (.md)")

    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt", "md"])

    if uploaded_file is not None and st.button("Upload & Process", type="primary"):
        with st.spinner("Extracting text, chunking, and embedding..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                resp = requests.post(f"{API_URL}/documents/upload", files=files, timeout=120)
                resp.raise_for_status()
                result = resp.json()
            except requests.RequestException as exc:
                st.error(f"Upload failed: {exc}")
                result = None

        if result:
            st.success(
                f"Uploaded '{result['filename']}': "
                f"{result['num_chunks']} chunks created from {result['char_count']} characters."
            )

    st.markdown("---")
    st.subheader("Uploaded Documents")
    try:
        docs = requests.get(f"{API_URL}/documents", timeout=10).json()
    except requests.RequestException as exc:
        st.error(f"Could not fetch documents: {exc}")
        docs = []

    if docs:
        st.dataframe(
            [
                {
                    "Filename": d["filename"],
                    "Type": d["file_type"],
                    "Chunks": d["num_chunks"],
                    "Characters": d["char_count"],
                    "Uploaded": d["upload_date"],
                }
                for d in docs
            ],
            use_container_width=True,
        )
    else:
        st.info("No documents uploaded yet.")


# ----------------------------------------------------------------------
# Page: Retrieval Comparison
# ----------------------------------------------------------------------

elif page == "Retrieval Comparison":
    st.title("Retrieval Comparison: Vector Search vs BM25")
    st.caption(
        "This page runs the SAME query through both a semantic (vector/FAISS) "
        "retriever and a lexical (BM25/keyword) retriever, so you can directly "
        "compare which chunks each strategy considers most relevant -- without "
        "involving the LLM."
    )

    question = st.text_input("Query", placeholder="e.g. password reset procedure")

    if st.button("Compare", type="primary") and question:
        with st.spinner("Running both retrievers..."):
            try:
                resp = requests.post(
                    f"{API_URL}/retrieval/compare", json={"question": question}, timeout=30
                )
                resp.raise_for_status()
                data = resp.json()
            except requests.RequestException as exc:
                st.error(f"Request failed: {exc}")
                data = None

        if data:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Vector Search (Semantic)")
                if not data["vector_results"]:
                    st.info("No results.")
                for i, chunk in enumerate(data["vector_results"], start=1):
                    st.markdown(
                        f"**#{i} — {chunk['document_name']} (chunk {chunk['chunk_index']}, "
                        f"score={chunk['score']})**"
                    )
                    st.caption(chunk["content"][:300] + ("..." if len(chunk["content"]) > 300 else ""))

            with col2:
                st.subheader("BM25 (Keyword)")
                if not data["bm25_results"]:
                    st.info("No results.")
                for i, chunk in enumerate(data["bm25_results"], start=1):
                    st.markdown(
                        f"**#{i} — {chunk['document_name']} (chunk {chunk['chunk_index']}, "
                        f"score={chunk['score']})**"
                    )
                    st.caption(chunk["content"][:300] + ("..." if len(chunk["content"]) > 300 else ""))


# ----------------------------------------------------------------------
# Page: Analytics
# ----------------------------------------------------------------------

elif page == "Analytics":
    st.title("Analytics Dashboard")

    try:
        data = requests.get(f"{API_URL}/analytics", timeout=10).json()
    except requests.RequestException as exc:
        st.error(f"Could not fetch analytics: {exc}")
        data = None

    if data:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Documents", data["num_documents"])
        col2.metric("Chunks", data["num_chunks"])
        col3.metric("Queries", data["num_queries"])
        col4.metric("Avg Latency (ms)", data["avg_latency_ms"])

        st.markdown("---")
        st.subheader("Recent Queries")
        if data["recent_queries"]:
            st.dataframe(data["recent_queries"], use_container_width=True)
        else:
            st.info("No queries logged yet.")

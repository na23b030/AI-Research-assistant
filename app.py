"""Streamlit frontend for the Evidence-Grounded AI Research Assistant."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
REQUEST_TIMEOUT = 180

st.set_page_config(
    page_title="RAG-Powered AI Research Assistant",
    page_icon="📚",
    layout="wide",
)


def api_request(method: str, path: str, **kwargs: Any) -> Any:
    """Call the backend and turn API failures into readable UI errors."""
    url = f"{BACKEND_URL}{path}"
    try:
        response = requests.request(method, url, timeout=REQUEST_TIMEOUT, **kwargs)
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Could not connect to the backend at {BACKEND_URL}. Start the FastAPI server first."
        ) from exc

    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except (ValueError, AttributeError):
            detail = response.text
        raise RuntimeError(f"API returned HTTP {response.status_code}: {detail}")
    try:
        return response.json()
    except ValueError as exc:
        raise RuntimeError("The backend returned a response that was not valid JSON.") from exc


if "messages" not in st.session_state:
    st.session_state.messages = []
if "eval_examples" not in st.session_state:
    st.session_state.eval_examples = []

st.title("📚 RAG-Powered AI Document Intelligence and Research Assistant")
st.caption("Upload documents, ask questions, inspect retrieved evidence, and evaluate retrieval quality.")

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Retrieved passages (top-k)", min_value=1, max_value=10, value=5)
    st.caption("Top-k controls how many passages are sent to the answer-generation step, subject to source diversity and context limits.")
    st.divider()
    st.subheader("Backend status")
    try:
        health = api_request("GET", "/health")
        st.success("Backend connected")
        if health.get("api_key_configured"):
            st.caption(f"Chat model: `{health.get('chat_model')}`")
            st.caption(f"Embedding model: `{health.get('embedding_model')}`")
        else:
            st.warning("OPENAI_API_KEY is not configured in the backend environment.")
    except RuntimeError as exc:
        st.error(str(exc))
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# 1. Document upload and inventory
st.header("1. Documents")
col_upload, col_inventory = st.columns([1, 1.25], gap="large")

with col_upload:
    st.subheader("Upload source files")
    files = st.file_uploader(
        "Choose PDF, TXT, or Markdown files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        help="Scanned PDFs without selectable text are not supported in this first version.",
    )
    if st.button("Index selected files", type="primary", disabled=not files, use_container_width=True):
        upload_payload = [
            ("files", (file.name, file.getvalue(), file.type or "application/octet-stream"))
            for file in files
        ]
        with st.spinner("Extracting text and creating embeddings..."):
            try:
                result = api_request("POST", "/api/documents/upload", files=upload_payload)
                st.success(result.get("message", "Upload request completed."))
                for item in result.get("uploaded", []):
                    st.write(f"✅ **{item['filename']}** — {item['chunk_count']} chunks (document ID {item['id']})")
                for item in result.get("skipped", []):
                    st.info(f"Skipped **{item['filename']}**: {item['reason']}")
                for item in result.get("errors", []):
                    st.error(f"{item['filename']}: {item['error']}")
            except RuntimeError as exc:
                st.error(str(exc))

with col_inventory:
    st.subheader("Indexed documents")
    documents: list[dict[str, Any]] = []
    try:
        documents = api_request("GET", "/api/documents").get("documents", [])
    except RuntimeError as exc:
        st.warning(str(exc))

    if not documents:
        st.info("No documents indexed yet. Upload a sample file to begin.")
    else:
        inventory = pd.DataFrame(
            [
                {
                    "ID": document["id"],
                    "Filename": document["filename"],
                    "Chunks": document["chunk_count"],
                    "Characters": document["char_count"],
                }
                for document in documents
            ]
        )
        st.dataframe(inventory, hide_index=True, use_container_width=True)
        deletion_map = {f"{doc['id']} · {doc['filename']}": doc["id"] for doc in documents}
        selected_delete = st.selectbox(
            "Select a document to remove",
            options=["(Select a document)"] + list(deletion_map),
            key="delete_document_selection",
        )
        if st.button("Delete selected document", disabled=selected_delete == "(Select a document)"):
            try:
                document_id = deletion_map[selected_delete]
                api_request("DELETE", f"/api/documents/{document_id}")
                st.success("Document and its stored chunks were deleted.")
                st.rerun()
            except RuntimeError as exc:
                st.error(str(exc))

st.divider()

# 2. RAG chat
st.header("2. Ask questions about your documents")
st.caption("Answers are generated from retrieved passages. Expand Sources below each response to inspect the evidence.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("Retrieved evidence and similarity scores"):
                for source in message["sources"]:
                    st.markdown(
                        f"**{source['citation']} — {source['filename']}** · "
                        f"chunk {source['chunk_index']} · cosine similarity `{source['similarity']}`"
                    )
                    st.write(source["text"])
        if message["role"] == "assistant" and message.get("metadata"):
            meta = message["metadata"]
            st.caption(
                f"Model: {meta.get('model', 'n/a')} · {meta.get('latency_ms', 'n/a')} ms · "
                f"Input/output tokens: {meta.get('usage', {}).get('input_tokens', 'n/a')}/"
                f"{meta.get('usage', {}).get('output_tokens', 'n/a')}"
            )

user_prompt = st.chat_input("Ask a question about the indexed documents...")
if user_prompt:
    if not documents:
        st.warning("Upload and index at least one document before asking questions.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)
        history = st.session_state.messages[:-1][-6:]
        with st.chat_message("assistant"):
            with st.spinner("Retrieving evidence and generating an answer..."):
                try:
                    response = api_request(
                        "POST",
                        "/api/chat",
                        json={"question": user_prompt, "top_k": top_k, "history": history},
                    )
                    answer = response.get("answer", "No answer was returned.")
                    st.markdown(answer)
                    sources = response.get("sources", [])
                    if sources:
                        with st.expander("Retrieved evidence and similarity scores"):
                            for source in sources:
                                st.markdown(
                                    f"**{source['citation']} — {source['filename']}** · "
                                    f"chunk {source['chunk_index']} · cosine similarity `{source['similarity']}`"
                                )
                                st.write(source["text"])
                    meta = {
                        "model": response.get("model"),
                        "latency_ms": response.get("latency_ms"),
                        "usage": response.get("usage", {}),
                    }
                    st.caption(
                        f"Model: {meta['model']} · {meta['latency_ms']} ms · "
                        f"Input/output tokens: {meta['usage'].get('input_tokens', 'n/a')}/"
                        f"{meta['usage'].get('output_tokens', 'n/a')}"
                    )
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer, "sources": sources, "metadata": meta}
                    )
                except RuntimeError as exc:
                    error_text = f"I couldn't complete that request. {exc}"
                    st.error(error_text)
                    st.session_state.messages.append({"role": "assistant", "content": error_text})

st.divider()

# 3. Retrieval evaluation
st.header("3. Retrieval evaluation")
st.markdown(
    "Create a small labelled test set. For each question, select the document(s) that "
    "**should** be retrieved. The metrics measure source-document retrieval, not answer quality."
)

if not documents:
    st.info("Index documents first to create retrieval evaluation examples.")
    st.session_state.eval_examples = []
else:
    doc_by_id = {int(doc["id"]): doc for doc in documents}
    with st.form("add_eval_example_form", clear_on_submit=True):
        eval_question = st.text_input("Evaluation question", placeholder="What does the document say about RAG?")
        relevant_ids = st.multiselect(
            "Expected relevant document(s)",
            options=list(doc_by_id.keys()),
            format_func=lambda doc_id: f"{doc_id} · {doc_by_id[doc_id]['filename']}",
        )
        add_example = st.form_submit_button("Add example")
    if add_example:
        if not eval_question.strip():
            st.error("Enter a question before adding the example.")
        elif not relevant_ids:
            st.error("Select at least one expected relevant document.")
        else:
            st.session_state.eval_examples.append(
                {"question": eval_question.strip(), "relevant_document_ids": relevant_ids}
            )
            st.success("Evaluation example added.")

    if st.session_state.eval_examples:
        st.write(f"**{len(st.session_state.eval_examples)} labelled example(s)**")
        eval_display = []
        for index, example in enumerate(st.session_state.eval_examples, start=1):
            readable = [
                f"{doc_id}: {doc_by_id[doc_id]['filename']}"
                for doc_id in example["relevant_document_ids"]
                if doc_id in doc_by_id
            ]
            eval_display.append({"#": index, "Question": example["question"], "Expected sources": "; ".join(readable)})
        st.dataframe(pd.DataFrame(eval_display), hide_index=True, use_container_width=True)
        eval_col1, eval_col2 = st.columns(2)
        with eval_col1:
            if st.button("Run retrieval evaluation", type="primary", use_container_width=True):
                with st.spinner("Embedding evaluation questions and measuring retrieval..."):
                    try:
                        result = api_request(
                            "POST",
                            "/api/evaluate/retrieval",
                            json={"examples": st.session_state.eval_examples, "top_k": top_k},
                        )
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Hit rate @ k", f"{result['hit_rate_at_k']:.1%}")
                        m2.metric("Mean recall @ k", f"{result['mean_recall_at_k']:.1%}")
                        m3.metric("MRR @ k", f"{result['mean_reciprocal_rank_at_k']:.3f}")
                        st.caption(result["metric_note"])
                        st.json(result["details"])
                    except RuntimeError as exc:
                        st.error(str(exc))
        with eval_col2:
            if st.button("Clear evaluation examples", use_container_width=True):
                st.session_state.eval_examples = []
                st.rerun()
st.caption("Local prototype · Authentication and per-user document isolation are not included. Use non-sensitive documents.")

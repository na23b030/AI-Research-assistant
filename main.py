"""FastAPI application exposing document ingestion, RAG chat, and evaluation."""
from __future__ import annotations

import hashlib
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from backend.config import (
    CHUNK_OVERLAP_WORDS,
    CHUNK_SIZE_WORDS,
    DEFAULT_TOP_K,
    EMBEDDING_MODEL,
    MAX_FILE_MB,
    MAX_FILES_PER_UPLOAD,
    MAX_TOP_K,
    OPENAI_API_KEY,
    OPENAI_MODEL,
)
from backend.database import (
    delete_document,
    document_exists,
    find_document_by_hash,
    initialize_database,
    list_documents,
    save_document,
)
from backend.evaluation import evaluate_retrieval
from backend.llm_client import MissingAPIKeyError, OpenAIError, embed_texts, require_api_key
from backend.rag import answer_question
from backend.schemas import ChatRequest, RetrievalEvaluationRequest
from backend.text_processing import SUPPORTED_EXTENSIONS, chunk_text, extract_text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("research_assistant")


@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_database()
    logger.info("Database initialized.")
    yield


app = FastAPI(
    title="RAG-Powered AI Document Intelligence and Research Assistant API",
    description=(
        "Upload PDF and text documents, retrieve relevant passages, generate cited answers, "
        "and evaluate source-document retrieval."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


def _require_configured_api_key() -> None:
    try:
        require_api_key()
    except MissingAPIKeyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _api_error(exc: OpenAIError) -> HTTPException:
    logger.exception("OpenAI API request failed", exc_info=exc)
    return HTTPException(
        status_code=502,
        detail=(
            f"The model provider request failed ({type(exc).__name__}). "
            "Check API-key validity, API billing/quota, rate limits, and provider status."
        ),
    )


@app.get("/health", tags=["System"])
def health() -> dict[str, Any]:
    """Basic health endpoint. It never reveals the secret API key."""
    return {
        "status": "ok",
        "api_key_configured": bool(OPENAI_API_KEY),
        "chat_model": OPENAI_MODEL,
        "embedding_model": EMBEDDING_MODEL,
    }


@app.get("/api/documents", tags=["Documents"])
def get_documents() -> dict[str, Any]:
    documents = list_documents()
    return {"count": len(documents), "documents": documents}


@app.post("/api/documents/upload", tags=["Documents"])
def upload_documents(files: list[UploadFile] = File(...)) -> dict[str, Any]:
    """Extract, chunk, embed, and persist one or more files."""
    _require_configured_api_key()
    if not files:
        raise HTTPException(status_code=400, detail="Choose at least one file to upload.")
    if len(files) > MAX_FILES_PER_UPLOAD:
        raise HTTPException(
            status_code=413,
            detail=f"Upload at most {MAX_FILES_PER_UPLOAD} files per request.",
        )

    max_bytes = MAX_FILE_MB * 1024 * 1024
    uploaded: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for upload in files:
        filename = Path(upload.filename or "unnamed").name
        suffix = Path(filename).suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            errors.append({"filename": filename, "error": "Only PDF, TXT, and MD files are supported."})
            continue

        try:
            content = upload.file.read(max_bytes + 1)
            if len(content) > max_bytes:
                errors.append({"filename": filename, "error": f"File exceeds the {MAX_FILE_MB} MB limit."})
                continue
            if not content:
                errors.append({"filename": filename, "error": "The file is empty."})
                continue

            digest = hashlib.sha256(content).hexdigest()
            existing = find_document_by_hash(digest)
            if existing and existing["embedding_model"] == EMBEDDING_MODEL:
                skipped.append(
                    {"id": existing["id"], "filename": existing["filename"], "reason": "Duplicate file content."}
                )
                continue
            if existing:
                # Embeddings from different models aren't directly comparable; re-index this file.
                delete_document(int(existing["id"]))

            text = extract_text(filename, content)
            chunks = chunk_text(text, CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS)
            if not chunks:
                errors.append({"filename": filename, "error": "No text could be chunked from this file."})
                continue

            vectors = embed_texts(chunks)
            document = save_document(filename, digest, text, chunks, vectors)
            uploaded.append(document)
            logger.info("Indexed document id=%s filename=%s chunks=%s", document["id"], filename, len(chunks))
        except ValueError as exc:
            errors.append({"filename": filename, "error": str(exc)})
        except OpenAIError as exc:
            raise _api_error(exc) from exc
        finally:
            upload.file.close()

    return {
        "uploaded": uploaded,
        "skipped": skipped,
        "errors": errors,
        "message": f"Indexed {len(uploaded)} document(s); skipped {len(skipped)} duplicate(s); {len(errors)} error(s).",
    }


@app.delete("/api/documents/{document_id}", tags=["Documents"])
def remove_document(document_id: int) -> dict[str, Any]:
    if not document_exists(document_id):
        raise HTTPException(status_code=404, detail="Document not found.")
    delete_document(document_id)
    return {"deleted": True, "document_id": document_id}


@app.post("/api/chat", tags=["RAG Chat"])
def chat(request: ChatRequest) -> dict[str, Any]:
    """Answer a question using retrieved document passages."""
    _require_configured_api_key()
    if not list_documents():
        raise HTTPException(status_code=400, detail="Upload at least one document before asking a question.")

    started = time.perf_counter()
    history = [turn.model_dump() for turn in request.history]
    try:
        result = answer_question(request.question, request.top_k, history)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OpenAIError as exc:
        raise _api_error(exc) from exc

    result["model"] = OPENAI_MODEL
    result["top_k"] = request.top_k
    result["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
    return result


@app.post("/api/evaluate/retrieval", tags=["Evaluation"])
def evaluate(request: RetrievalEvaluationRequest) -> dict[str, Any]:
    """Measure source-document retrieval against manually labeled examples."""
    _require_configured_api_key()
    if not list_documents():
        raise HTTPException(status_code=400, detail="Upload at least one document before running evaluation.")

    examples = [example.model_dump() for example in request.examples]
    started = time.perf_counter()
    try:
        result = evaluate_retrieval(examples, request.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OpenAIError as exc:
        raise _api_error(exc) from exc

    result["embedding_model"] = EMBEDDING_MODEL
    result["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
    return result


@app.exception_handler(Exception)
def unexpected_error_handler(request, exc: Exception):
    """Log unexpected errors while returning a generic message to clients."""
    logger.exception("Unhandled application error: %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "An unexpected server error occurred. Check the backend terminal logs."})

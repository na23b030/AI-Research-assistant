"""Baseline dense retrieval using cosine similarity over stored embeddings."""
from __future__ import annotations

import json
from typing import Any

import numpy as np

from backend.config import EMBEDDING_MODEL
from backend.database import get_all_chunks


def cosine_similarity(vector_a: list[float] | np.ndarray, vector_b: list[float] | np.ndarray) -> float:
    """Return cosine similarity; return 0 for zero-length vectors."""
    a = np.asarray(vector_a, dtype=np.float32)
    b = np.asarray(vector_b, dtype=np.float32)
    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape:
        raise ValueError("Cosine similarity requires two one-dimensional vectors of equal length.")
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(a, b) / denominator)


def retrieve_chunks(
    query_embedding: list[float],
    top_k: int = 5,
    max_chunks_per_document: int = 2,
) -> list[dict[str, Any]]:
    """Rank chunks by cosine similarity, returning diverse evidence passages.

    This is deliberately a simple exact search over the local SQLite table.
    The max-per-document cap reduces the chance that one long document crowds
    every other source out of the context window.
    """
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    if max_chunks_per_document <= 0:
        raise ValueError("max_chunks_per_document must be positive.")

    rows = [row for row in get_all_chunks() if row["embedding_model"] == EMBEDDING_MODEL]
    if not rows:
        return []

    query = np.asarray(query_embedding, dtype=np.float32)
    if query.ndim != 1:
        raise ValueError("Query embedding must be one-dimensional.")

    # Stored vectors must be produced by the same embedding model/dimension.
    vectors: list[list[float]] = []
    compatible_rows: list[dict[str, Any]] = []
    for row in rows:
        try:
            vector = json.loads(row["embedding_json"])
            if len(vector) != len(query):
                continue
            vectors.append(vector)
            compatible_rows.append(row)
        except (json.JSONDecodeError, TypeError):
            continue

    if not vectors:
        return []

    matrix = np.asarray(vectors, dtype=np.float32)
    query_norm = float(np.linalg.norm(query))
    row_norms = np.linalg.norm(matrix, axis=1)
    denominator = row_norms * query_norm
    scores = np.divide(
        matrix @ query,
        denominator,
        out=np.zeros_like(row_norms, dtype=np.float32),
        where=denominator > 0,
    )
    rank_order = np.argsort(scores)[::-1]

    results: list[dict[str, Any]] = []
    document_counts: dict[int, int] = {}
    for idx in rank_order:
        row = compatible_rows[int(idx)]
        document_id = int(row["document_id"])
        if document_counts.get(document_id, 0) >= max_chunks_per_document:
            continue

        results.append(
            {
                "chunk_id": int(row["chunk_id"]),
                "document_id": document_id,
                "filename": row["filename"],
                "chunk_index": int(row["chunk_index"]),
                "text": row["text"],
                "score": float(scores[int(idx)]),
            }
        )
        document_counts[document_id] = document_counts.get(document_id, 0) + 1
        if len(results) >= top_k:
            break

    return results

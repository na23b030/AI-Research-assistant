"""Small SQLite persistence layer for documents, chunks, and their embeddings.

This is intentionally a transparent starter implementation, not a distributed
vector database. It is suitable for a local portfolio MVP and can later be
replaced by pgvector, Qdrant, or another vector store.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from backend.config import DATABASE_PATH, EMBEDDING_MODEL


def _connect() -> sqlite3.Connection:
    Path(DATABASE_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DATABASE_PATH), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def connection() -> Iterator[sqlite3.Connection]:
    conn = _connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def initialize_database() -> None:
    """Create tables and indexes if this is the first run."""
    with connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                sha256 TEXT NOT NULL UNIQUE,
                uploaded_at TEXT NOT NULL,
                chunk_count INTEGER NOT NULL,
                char_count INTEGER NOT NULL,
                embedding_model TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                embedding_json TEXT NOT NULL,
                FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
                UNIQUE(document_id, chunk_index)
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_uploaded_at ON documents(uploaded_at)")


def find_document_by_hash(sha256: str) -> dict[str, Any] | None:
    with connection() as conn:
        row = conn.execute(
            "SELECT * FROM documents WHERE sha256 = ?", (sha256,)
        ).fetchone()
    return dict(row) if row else None


def save_document(
    filename: str,
    sha256: str,
    full_text: str,
    chunks: list[str],
    embeddings: list[list[float]],
) -> dict[str, Any]:
    if len(chunks) != len(embeddings):
        raise ValueError("Each text chunk must have exactly one embedding.")

    uploaded_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO documents
                (filename, sha256, uploaded_at, chunk_count, char_count, embedding_model)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (filename, sha256, uploaded_at, len(chunks), len(full_text), EMBEDDING_MODEL),
        )
        document_id = int(cursor.lastrowid)
        conn.executemany(
            """
            INSERT INTO chunks (document_id, chunk_index, text, embedding_json)
            VALUES (?, ?, ?, ?)
            """,
            [
                (document_id, index, chunk, json.dumps(vector, separators=(",", ":")))
                for index, (chunk, vector) in enumerate(zip(chunks, embeddings))
            ],
        )

    return {
        "id": document_id,
        "filename": filename,
        "uploaded_at": uploaded_at,
        "chunk_count": len(chunks),
        "char_count": len(full_text),
        "embedding_model": EMBEDDING_MODEL,
    }


def list_documents() -> list[dict[str, Any]]:
    with connection() as conn:
        rows = conn.execute(
            """
            SELECT id, filename, uploaded_at, chunk_count, char_count, embedding_model
            FROM documents
            ORDER BY uploaded_at DESC, id DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def get_all_chunks() -> list[dict[str, Any]]:
    with connection() as conn:
        rows = conn.execute(
            """
            SELECT
                c.id AS chunk_id,
                c.document_id,
                c.chunk_index,
                c.text,
                c.embedding_json,
                d.filename,
                d.embedding_model
            FROM chunks AS c
            INNER JOIN documents AS d ON d.id = c.document_id
            ORDER BY d.id, c.chunk_index
            """
        ).fetchall()
    return [dict(row) for row in rows]


def delete_document(document_id: int) -> bool:
    with connection() as conn:
        cursor = conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        return cursor.rowcount > 0


def document_exists(document_id: int) -> bool:
    with connection() as conn:
        row = conn.execute(
            "SELECT 1 FROM documents WHERE id = ?", (document_id,)
        ).fetchone()
    return row is not None

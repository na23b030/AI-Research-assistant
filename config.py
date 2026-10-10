"""Central configuration loaded from environment variables or a local .env file."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer.") from exc


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
# Treat the example placeholder as an unset key so the health endpoint stays honest.
if OPENAI_API_KEY.lower() in {"replace_with_your_api_key", "your_api_key_here", "replace-me"}:
    OPENAI_API_KEY = ""
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small").strip()

_data_dir = PROJECT_ROOT / "data"
_default_db = _data_dir / "research_assistant.sqlite3"
_db_env = os.getenv("DATABASE_PATH", str(_default_db)).strip()
DATABASE_PATH = Path(_db_env)
if not DATABASE_PATH.is_absolute():
    DATABASE_PATH = PROJECT_ROOT / DATABASE_PATH

MAX_FILE_MB = _int_env("MAX_FILE_MB", 10)
MAX_FILES_PER_UPLOAD = _int_env("MAX_FILES_PER_UPLOAD", 5)
CHUNK_SIZE_WORDS = _int_env("CHUNK_SIZE_WORDS", 350)
CHUNK_OVERLAP_WORDS = _int_env("CHUNK_OVERLAP_WORDS", 60)
MAX_TOP_K = _int_env("MAX_TOP_K", 10)
DEFAULT_TOP_K = min(5, MAX_TOP_K)
MAX_CONTEXT_CHARS = _int_env("MAX_CONTEXT_CHARS", 16000)

if MAX_FILE_MB <= 0:
    raise ValueError("MAX_FILE_MB must be greater than zero.")
if MAX_FILES_PER_UPLOAD <= 0:
    raise ValueError("MAX_FILES_PER_UPLOAD must be greater than zero.")
if CHUNK_SIZE_WORDS <= 0 or CHUNK_OVERLAP_WORDS < 0:
    raise ValueError("Chunk size must be positive and overlap must be zero or greater.")
if CHUNK_OVERLAP_WORDS >= CHUNK_SIZE_WORDS:
    raise ValueError("CHUNK_OVERLAP_WORDS must be smaller than CHUNK_SIZE_WORDS.")
if MAX_TOP_K <= 0:
    raise ValueError("MAX_TOP_K must be greater than zero.")

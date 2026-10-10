"""Document text extraction and simple, overlapping word-based chunking."""
from __future__ import annotations

import re
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def extract_text(filename: str, content: bytes) -> str:
    """Extract text from a PDF, TXT, or Markdown file.

    Scanned/image-only PDFs are not OCR'd in this first version. They need an
    OCR or vision-based extension, which we can add later.
    """
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError("Unsupported file type. Upload a PDF, TXT, or MD file.")
    if not content:
        raise ValueError("The uploaded file is empty.")

    if suffix == ".pdf":
        try:
            reader = PdfReader(BytesIO(content), strict=False)
            if reader.is_encrypted:
                try:
                    reader.decrypt("")
                except Exception as exc:
                    raise ValueError("This PDF is password-protected and cannot be read.") from exc
            page_texts = [page.extract_text() or "" for page in reader.pages]
            text = "\n".join(page_texts)
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("Could not read this PDF. It may be damaged or password-protected.") from exc
        if not text.strip():
            raise ValueError(
                "No selectable text was found in this PDF. Scanned PDFs need OCR, which is not included yet."
            )
    else:
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = content.decode("latin-1")

    # Normalizing whitespace simplifies chunking while keeping all the words.
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(
    text: str,
    chunk_size_words: int = 350,
    overlap_words: int = 60,
) -> list[str]:
    """Split text into overlapping chunks measured in words.

    Overlap helps avoid losing meaning when a relevant sentence falls near a
    chunk boundary. This baseline doesn't yet use semantic chunking.
    """
    if chunk_size_words <= 0:
        raise ValueError("chunk_size_words must be greater than zero.")
    if overlap_words < 0:
        raise ValueError("overlap_words cannot be negative.")
    if overlap_words >= chunk_size_words:
        raise ValueError("overlap_words must be smaller than chunk_size_words.")

    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    step = chunk_size_words - overlap_words
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + chunk_size_words]).strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_size_words >= len(words):
            break
    return chunks

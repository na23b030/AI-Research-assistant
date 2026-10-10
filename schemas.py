"""Request models for the public API."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from backend.config import DEFAULT_TOP_K, MAX_TOP_K


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=2000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=MAX_TOP_K)
    history: list[ChatTurn] = Field(default_factory=list, max_length=8)


class EvaluationExample(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    relevant_document_ids: list[int] = Field(min_length=1, max_length=20)


class RetrievalEvaluationRequest(BaseModel):
    examples: list[EvaluationExample] = Field(min_length=1, max_length=50)
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=MAX_TOP_K)

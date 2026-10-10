"""Small wrapper around OpenAI embeddings and the Responses API."""
from __future__ import annotations

from functools import lru_cache
from typing import Any

try:
    from openai import OpenAI
    from openai import OpenAIError as OpenAIError
except ImportError:  # Allows offline unit tests for the non-LLM parts of the project.
    OpenAI = None  # type: ignore[assignment,misc]

    class OpenAIError(Exception):
        """Fallback error type used only when the OpenAI SDK is not installed."""

from backend.config import EMBEDDING_MODEL, OPENAI_API_KEY, OPENAI_MODEL


class MissingAPIKeyError(RuntimeError):
    """Raised when the local environment doesn't contain an API key."""


def require_api_key() -> None:
    if not OPENAI_API_KEY:
        raise MissingAPIKeyError(
            "OPENAI_API_KEY is missing. Add your key to the project's .env file, then restart the backend."
        )


@lru_cache(maxsize=1)
def get_client():
    require_api_key()
    if OpenAI is None:
        raise RuntimeError("The OpenAI SDK is not installed. Run: pip install -r requirements.txt")
    return OpenAI(api_key=OPENAI_API_KEY, timeout=60.0, max_retries=2)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Create embeddings in batches and preserve the input order."""
    if not texts:
        return []
    require_api_key()

    client = get_client()
    all_embeddings: list[list[float]] = []
    batch_size = 64
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
            encoding_format="float",
        )
        batch_data = sorted(response.data, key=lambda item: item.index)
        all_embeddings.extend(
            [[float(value) for value in item.embedding] for item in batch_data]
        )
    return all_embeddings


def generate_text(instructions: str, prompt: str, max_output_tokens: int = 800) -> tuple[str, dict[str, Any]]:
    """Generate an answer with the OpenAI Responses API."""
    require_api_key()
    client = get_client()
    response = client.responses.create(
        model=OPENAI_MODEL,
        instructions=instructions,
        input=prompt,
        max_output_tokens=max_output_tokens,
    )

    usage = getattr(response, "usage", None)
    usage_dict = {
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
    }
    return (response.output_text or "").strip(), usage_dict

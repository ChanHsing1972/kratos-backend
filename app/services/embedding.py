from __future__ import annotations

from functools import lru_cache

from openai import OpenAI

from app.core.config import settings


EMBEDDING_DIMENSION = 1536


def embed_text(text: str) -> list[float]:
    """Return an embedding vector from the configured real embedding API."""

    value = str(text or "").strip()
    if not value:
        return [0.0] * EMBEDDING_DIMENSION

    if not settings.RAG_EMBEDDING_EFFECTIVE_API_KEY:
        raise RuntimeError("RAG_EMBEDDING_API_KEY is required for knowledge-base embeddings.")

    client = _embedding_client()
    response = client.embeddings.create(
        input=value,
        model=settings.RAG_EMBEDDING_MODEL,
        dimensions=settings.RAG_EMBEDDING_DIMENSIONS,
    )
    return [float(item) for item in response.data[0].embedding]


def embedding_model_label() -> str:
    return settings.RAG_EMBEDDING_MODEL


@lru_cache(maxsize=1)
def _embedding_client() -> OpenAI:
    if not settings.RAG_EMBEDDING_EFFECTIVE_API_KEY:
        raise RuntimeError("RAG_EMBEDDING_API_KEY is required for knowledge-base embeddings.")
    return OpenAI(
        api_key=settings.RAG_EMBEDDING_EFFECTIVE_API_KEY,
        base_url=settings.RAG_EMBEDDING_EFFECTIVE_BASE_URL,
        timeout=settings.RAG_EMBEDDING_TIMEOUT_SECONDS,
        max_retries=settings.RAG_EMBEDDING_MAX_RETRIES,
    )

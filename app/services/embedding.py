from __future__ import annotations

import hashlib
import math
from functools import lru_cache

from openai import OpenAI

from app.core.config import settings


EMBEDDING_DIMENSION = 1536


def embed_text(text: str) -> list[float]:
    """Return an embedding vector, using a deterministic local fallback when unconfigured."""

    value = str(text or "").strip()
    if not value:
        return [0.0] * EMBEDDING_DIMENSION

    if settings.RAG_EMBEDDING_EFFECTIVE_API_KEY:
        client = _embedding_client()
        response = client.embeddings.create(
            input=value,
            model=settings.RAG_EMBEDDING_MODEL,
        )
        return [float(item) for item in response.data[0].embedding]

    return _local_hash_embedding(value)


def embedding_model_label() -> str:
    if settings.RAG_EMBEDDING_EFFECTIVE_API_KEY:
        return settings.RAG_EMBEDDING_MODEL
    return "local-hash-embedding"


@lru_cache(maxsize=1)
def _embedding_client() -> OpenAI:
    return OpenAI(
        api_key=settings.RAG_EMBEDDING_EFFECTIVE_API_KEY,
        base_url=settings.RAG_EMBEDDING_EFFECTIVE_BASE_URL,
        timeout=settings.RAG_EMBEDDING_TIMEOUT_SECONDS,
        max_retries=settings.RAG_EMBEDDING_MAX_RETRIES,
    )


def _local_hash_embedding(text: str) -> list[float]:
    vector = [0.0] * EMBEDDING_DIMENSION
    tokens = _tokenize_for_embedding(text)
    if not tokens:
        tokens = [text]
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % EMBEDDING_DIMENSION
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign
    norm = math.sqrt(sum(item * item for item in vector)) or 1.0
    return [item / norm for item in vector]


def _tokenize_for_embedding(text: str) -> list[str]:
    import re

    normalized = text.lower()
    words = re.findall(r"[a-z0-9]+", normalized)
    chinese = re.findall(r"[\u4e00-\u9fff]", normalized)
    bigrams = ["".join(pair) for pair in zip(chinese, chinese[1:])]
    return words + chinese + bigrams

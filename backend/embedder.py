"""Wrapper around the OpenAI embeddings API with retry support."""
from __future__ import annotations

import logging
import time
from typing import List

import httpx
from openai import OpenAI, OpenAIError

from .config import get_settings

logger = logging.getLogger(__name__)


class EmbeddingError(RuntimeError):
    """Raised when embeddings cannot be generated."""


class Embedder:
    """Create embeddings for incoming queries."""

    def __init__(self, client: OpenAI | None = None) -> None:
        settings = get_settings()
        self._model = settings.embedding_model
        self._client = client or OpenAI(api_key=settings.openai_api_key)
        self._max_retries = 3
        self._retry_delay = 2.0

    def embed(self, text: str) -> List[float]:
        """Return the embedding vector for the supplied text."""

        if not text:
            raise EmbeddingError("Query text must not be empty")

        for attempt in range(1, self._max_retries + 1):
            try:
                logger.debug("Requesting embedding (attempt %s)", attempt)
                response = self._client.embeddings.create(
                    model=self._model,
                    input=text,
                    timeout=30,
                )
                return response.data[0].embedding
            except (OpenAIError, httpx.HTTPError) as exc:  # pragma: no cover - network call
                logger.warning("Embedding request failed on attempt %s: %s", attempt, exc)
                if attempt >= self._max_retries:
                    logger.exception("Embedding service failed after retries")
                    raise EmbeddingError("Failed to generate embedding") from exc
                time.sleep(self._retry_delay * attempt)

        raise EmbeddingError("Failed to generate embedding")
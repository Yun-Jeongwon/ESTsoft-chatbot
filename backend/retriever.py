"""Utilities for retrieving answers from Qdrant."""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from qdrant_client.http.exceptions import UnexpectedResponse

from .config import get_settings

logger = logging.getLogger(__name__)


class RetrievalError(RuntimeError):
    """Raised when retrieval from the vector store fails."""


class Retriever:
    """Perform similarity search against a Qdrant collection."""

    def __init__(self, client: QdrantClient | None = None) -> None:
        settings = get_settings()
        self._collection = settings.qdrant_collection
        self._threshold = settings.sim_threshold
        self._top_k = settings.top_k
        self._client = client or QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        try:
            if not self._client.collection_exists(self._collection):
                logger.info("Creating missing Qdrant collection '%s'", self._collection)
                self._client.create_collection(
                    collection_name=self._collection,
                    vectors_config=rest.VectorParams(
                        size=1536,
                        distance=rest.Distance.COSINE,
                    ),
                )
        except UnexpectedResponse as exc:  # pragma: no cover - network call
            logger.exception("Failed to ensure Qdrant collection existence")
            raise RetrievalError("Unable to ensure Qdrant collection") from exc

    def search(self, vector: List[float]) -> Dict[str, Any]:
        """Search for similar items and return the best match."""

        try:
            hits = self._client.search(
                collection_name=self._collection,
                query_vector=vector,
                limit=self._top_k,
                with_payload=True,
            )
        except Exception as exc:  # pragma: no cover - network call
            logger.exception("Qdrant search failed")
            raise RetrievalError("Vector search failed") from exc

        if not hits:
            logger.info("No Qdrant search results returned")
            return {
                "answer": "검색 결과가 없습니다.",
                "score": None,
                "warning": "⚠️ 신뢰도 낮음(score<threshold)",
            }

        top_hit = hits[0]
        payload = top_hit.payload or {}
        answer = payload.get("answer")
        response: Dict[str, Any] = {
            "answer": answer or "검색 결과가 없습니다.",
            "score": top_hit.score,
        }

        if "question" in payload:
            response["question"] = payload["question"]
        if "source" in payload:
            response["source"] = payload["source"]

        if top_hit.score is None or top_hit.score < self._threshold:
            response["warning"] = "⚠️ 신뢰도 낮음(score<threshold)"

        return response
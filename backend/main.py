"""FastAPI application entry point for the chatbot backend."""
from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import get_settings
from .embedder import Embedder, EmbeddingError
from .retriever import RetrievalError, Retriever

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

settings = get_settings()
embedder = Embedder()
retriever = Retriever()

app = FastAPI(title="ESTsoft Chatbot Backend", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str = Field(..., description="사용자 질문")


class QueryResponse(BaseModel):
    answer: str
    question: str | None = None
    score: float | None = None
    warning: str | None = None
    source: str | None = None


@app.middleware("http")
async def log_requests(request: Request, call_next):  # type: ignore[override]
    logger.info("Incoming request %s %s", request.method, request.url.path)
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("Unhandled error during request processing")
        raise
    logger.info("Request completed with status %s", response.status_code)
    return response


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Return basic health information."""

    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(payload: QueryRequest) -> Any:
    """Handle user query requests."""

    try:
        embedding = await run_in_threadpool(embedder.embed, payload.query)
    except EmbeddingError as exc:
        logger.error("Embedding generation failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding service unavailable",
        ) from exc

    try:
        result = await run_in_threadpool(retriever.search, embedding)
    except RetrievalError as exc:
        logger.error("Retrieval failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector search unavailable",
        ) from exc

    if not isinstance(result, dict):
        logger.error("Retriever returned invalid response type: %s", type(result))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Invalid retrieval response")

    logger.info("Query processed successfully with score=%s", result.get("score"))
    return result
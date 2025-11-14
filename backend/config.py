"""Configuration utilities for the backend service."""
from __future__ import annotations

from functools import lru_cache
from pydantic import Field
from dotenv import load_dotenv
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    openai_api_key: str = Field(..., alias="OPENAI_API_KEY")
    embedding_model: str = Field("text-embedding-3-small", alias="EMBEDDING_MODEL")
    qdrant_url: str = Field(..., alias="QDRANT_URL")
    qdrant_api_key: str | None = Field(None, alias="QDRANT_API_KEY")
    qdrant_collection: str = Field(..., alias="QDRANT_COLLECTION")
    sim_threshold: float = Field(0.82, alias="SIM_THRESHOLD")
    top_k: int = Field(5, alias="TOP_K")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "populate_by_name": True,
    }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings."""

    load_dotenv()  # ensure .env is loaded when running locally
    return Settings()
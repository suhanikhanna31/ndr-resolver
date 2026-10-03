from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    database_url: str | None
    redis_url: str | None
    embeddings_provider: str
    embeddings_model: str | None
    retrieval_k: int
    retrieval_max_distance: float


def get_settings() -> Settings:
    """Read env on every call so tests and workers can change it; empty string means unset."""
    return Settings(
        database_url=os.getenv("DATABASE_URL") or None,
        redis_url=os.getenv("REDIS_URL") or None,
        embeddings_provider=os.getenv("EMBEDDINGS_PROVIDER", "fastembed"),
        embeddings_model=os.getenv("EMBEDDINGS_MODEL") or None,
        retrieval_k=int(os.getenv("RETRIEVAL_K", "3")),
        retrieval_max_distance=float(os.getenv("RETRIEVAL_MAX_DISTANCE", "0.35")),
    )


def asyncpg_url(url: str) -> str:
    """LangChain's PGEngine speaks asyncpg; the rest of the app uses plain psycopg URLs."""
    return url.replace("postgresql://", "postgresql+asyncpg://", 1)

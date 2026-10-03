"""Retrieval feedback loop: human resolutions -> embeddings (pgvector via LangChain) -> few-shot examples."""
from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Sequence

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_postgres import PGEngine, PGVectorStore
from langchain_postgres.v2.vectorstores import DistanceStrategy

from .config import Settings, asyncpg_url
from .db import DecisionRepo
from .llm import AnthropicExtractor
from .schemas import Extraction, FeedbackRequest, Intent, NDRRequest, ResolvedExample

log = logging.getLogger("ndr.retrieval")

TABLE = "resolved_examples"
# Free-text addresses can't be redacted reliably, so they are never embedded or stored here.
INDEXABLE = set(Intent) - {Intent.ADDRESS_CORRECTION}

_PHONE = re.compile(r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}(?!\d)")
_PIN = re.compile(r"(?<!\d)[1-9]\d{5}(?!\d)")


def redact(text: str) -> str:
    """Mask phone numbers and pincodes before anything is embedded or stored."""
    return _PIN.sub("[PIN]", _PHONE.sub("[PHONE]", text))


class FastEmbedEmbeddings(Embeddings):
    """Thin LangChain adapter over fastembed (ONNX, CPU, no API key, multilingual model)."""

    def __init__(self, model_name: str):
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name, cache_dir=os.getenv("FASTEMBED_CACHE_PATH"))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def make_embeddings(s: Settings) -> Embeddings:
    p = s.embeddings_provider
    if p == "fake":  # deterministic hash vectors: plumbing tests only, no semantic meaning
        from langchain_core.embeddings import DeterministicFakeEmbedding

        return DeterministicFakeEmbedding(size=64)
    if p == "openai":
        from langchain_openai import OpenAIEmbeddings  # optional dependency

        return OpenAIEmbeddings(model=s.embeddings_model or "text-embedding-3-small")
    if p == "fastembed":
        return FastEmbedEmbeddings(s.embeddings_model or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    raise ValueError(f"unknown EMBEDDINGS_PROVIDER: {p}")


class ExampleStore:
    def __init__(self, store: PGVectorStore, engine: PGEngine, k: int, max_distance: float):
        self._store, self._engine, self.k, self.max_distance = store, engine, k, max_distance

    @classmethod
    async def create(cls, database_url: str, embeddings: Embeddings, repo: DecisionRepo, *,
                     k: int = 3, max_distance: float = 0.35, table: str = TABLE) -> "ExampleStore":
        engine = PGEngine.from_connection_string(asyncpg_url(database_url))
        if not await repo.table_exists(table):
            # Vector size is probed from the model, so it can never drift from the config.
            dim = len(await asyncio.to_thread(embeddings.embed_query, "dimension probe"))
            await engine.ainit_vectorstore_table(table, vector_size=dim)
        store = await PGVectorStore.create(
            engine, embedding_service=embeddings, table_name=table,
            distance_strategy=DistanceStrategy.COSINE_DISTANCE)
        return cls(store, engine, k, max_distance)

    async def add(self, fb: FeedbackRequest) -> bool:
        if fb.intent not in INDEXABLE:
            return False
        meta = {"awb": fb.awb, "ndr_reason": fb.ndr_reason, "intent": fb.intent.value,
                "date_expression": fb.date_expression, "time_slot": fb.time_slot}
        await self._store.aadd_documents([Document(page_content=redact(fb.customer_utterance), metadata=meta)])
        return True

    def similar(self, req: NDRRequest) -> list[ResolvedExample]:
        """Sync on purpose: called from the worker thread that runs the extractor."""
        hits = self._store.similarity_search_with_score(redact(req.customer_utterance), k=self.k * 2)
        out, seen = [], set()
        for doc, dist in hits:
            key = (doc.page_content, doc.metadata.get("intent"))
            if dist > self.max_distance or key in seen:
                continue
            seen.add(key)
            out.append(ResolvedExample(utterance=doc.page_content, distance=dist, **doc.metadata))
        return out[: self.k]

    async def close(self) -> None:
        await self._engine.close()


class RetrievalAugmentedExtractor:
    """Wraps the base extractor: retrieve similar human-resolved cases, inject as few-shot data."""

    def __init__(self, base: AnthropicExtractor, store: ExampleStore):
        self.base, self.store = base, store

    def extract(self, req: NDRRequest) -> Extraction:
        examples: Sequence[ResolvedExample] = ()
        try:
            examples = self.store.similar(req)
        except Exception as e:  # vector DB down must degrade to zero-shot, never block a decision
            log.warning("retrieval_failed awb=%s err=%s", req.awb, type(e).__name__)
        return self.base.extract(req, examples=examples)

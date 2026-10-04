from __future__ import annotations

import asyncio
import json
import logging
import time

from .config import Settings, get_settings
from .db import DecisionRepo
from .llm import Extractor, make_extractor
from .resolver import resolve
from .retrieval import ExampleStore, RetrievalAugmentedExtractor, make_embeddings
from .schemas import Decision, FeedbackRequest, NDRRequest

log = logging.getLogger("ndr")


class DecisionService:
    """One code path for both the sync endpoint and the queue worker."""

    def __init__(self, extractor: Extractor, repo: DecisionRepo | None = None, store: ExampleStore | None = None):
        self.extractor, self.repo, self.store = extractor, repo, store

    async def handle(self, req: NDRRequest) -> Decision:
        if self.repo:  # idempotency layer 2: a stored decision is returned without another LLM call
            cached = await self.repo.get(req.awb, req.attempt_number)
            if cached:
                self._log(req, cached, 0, cached=True)
                return cached
        t0 = time.perf_counter()
        decision = await asyncio.to_thread(resolve, req, self.extractor)  # sync SDK + sync retrieval
        ms = round((time.perf_counter() - t0) * 1000)
        if self.repo:
            decision = await self.repo.save(req, decision, ms)
        self._log(req, decision, ms, cached=False)
        return decision

    async def add_feedback(self, fb: FeedbackRequest) -> bool:
        assert self.store is not None
        return await self.store.add(fb)

    @staticmethod
    def _log(req: NDRRequest, d: Decision, ms: int, cached: bool) -> None:
        # Decisions only, never the raw utterance: it can contain addresses and phone numbers (PII).
        log.info(json.dumps({"awb": req.awb, "action": d.action.value, "needs_human": d.needs_human,
                             "reasons": d.reasons, "latency_ms": ms, "cached": cached}))

    async def close(self) -> None:
        if self.store:
            await self.store.close()
        if self.repo:
            await self.repo.close()


async def build_service(settings: Settings | None = None) -> DecisionService:
    s = settings or get_settings()
    base = make_extractor()
    if not s.database_url:
        return DecisionService(base)  # no DB: plain zero-shot extractor, no persistence
    repo = await DecisionRepo.open(s.database_url)
    embeddings = await asyncio.to_thread(make_embeddings, s)
    store = await ExampleStore.create(s.database_url, embeddings, repo,
                                      k=s.retrieval_k, max_distance=s.retrieval_max_distance)
    return DecisionService(RetrievalAugmentedExtractor(base, store), repo, store)

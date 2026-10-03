"""Needs real Postgres+pgvector and Redis: set DATABASE_URL and REDIS_URL (CI and docker compose do)."""
import asyncio
import os
import uuid

import httpx
import pytest

pytestmark = pytest.mark.skipif(not (os.getenv("DATABASE_URL") and os.getenv("REDIS_URL")),
                                reason="DATABASE_URL/REDIS_URL not set")

if os.getenv("DATABASE_URL") and os.getenv("REDIS_URL"):
    from arq import create_pool
    from arq.connections import RedisSettings
    from arq.worker import Worker

    from app.config import Settings
    from app.db import DecisionRepo
    from app.main import app, get_queue, get_service
    from app.resolver import decide
    from app.retrieval import ExampleStore, make_embeddings
    from app.schemas import CarrierAction, Decision, Extraction, FeedbackRequest, Intent, NDRRequest
    from app.service import DecisionService
    from app.worker import process_ndr

DB, REDIS = os.getenv("DATABASE_URL"), os.getenv("REDIS_URL")


def _req(awb, text="kal shaam ko bhej do", attempt=1):
    return NDRRequest(awb=awb, ndr_reason="customer_unavailable", customer_utterance=text,
                      attempt_number=attempt, received_at="2026-10-03T11:00:00+05:30")


class CountingStub:
    def __init__(self):
        self.calls = 0

    def extract(self, req):
        self.calls += 1
        return Extraction(intent=Intent.RESCHEDULE, confidence=0.95, date_expression="tomorrow", time_slot="evening")


def _settings():
    return Settings(DB, REDIS, "fake", None, 3, 0.35)


def test_feedback_is_retrievable_redacted_and_address_is_skipped():
    async def go():
        table = f"test_examples_{uuid.uuid4().hex[:8]}"
        repo = await DecisionRepo.open(DB)
        store = await ExampleStore.create(DB, make_embeddings(_settings()), repo, table=table)
        try:
            base = dict(awb="F1", ndr_reason="customer_unavailable")
            assert await store.add(FeedbackRequest(**base, customer_utterance="kal shaam ko bhej do",
                                                   intent=Intent.RESCHEDULE, date_expression="tomorrow", time_slot="evening"))
            assert await store.add(FeedbackRequest(**base, customer_utterance="mera number 9876543210 hai",
                                                   intent=Intent.PHONE_CORRECTION))
            assert not await store.add(FeedbackRequest(**base, customer_utterance="Flat 4, Noida 201301",
                                                       intent=Intent.ADDRESS_CORRECTION))
            hits = await asyncio.to_thread(store.similar, _req("Q1"))  # sync path, as used by the extractor
            assert len(hits) == 1 and hits[0].intent is Intent.RESCHEDULE
            assert hits[0].date_expression == "tomorrow" and hits[0].distance < 0.01
            phone = await asyncio.to_thread(store.similar, _req("Q2", "mera number 9876543210 hai"))
            assert "9876543210" not in phone[0].utterance and "[PHONE]" in phone[0].utterance
        finally:
            async with repo.pool.connection() as c:
                await c.execute(f"DROP TABLE IF EXISTS {table}")
            await store.close()
            await repo.close()

    asyncio.run(go())


def test_decision_is_idempotent_per_awb_and_attempt():
    async def go():
        repo = await DecisionRepo.open(DB)
        stub = CountingStub()
        svc = DecisionService(stub, repo)
        awb = f"IDEM{uuid.uuid4().hex[:8]}"
        try:
            first = await svc.handle(_req(awb))
            second = await svc.handle(_req(awb))
            assert stub.calls == 1 and first == second  # second call served from Postgres, no LLM call
            # a concurrent loser of the race gets the stored row, not its own decision
            other = decide(_req(awb), Extraction(intent=Intent.REFUSE_DELIVERY, confidence=0.99))
            assert (await repo.save(_req(awb), other, 5)).action is CarrierAction.REATTEMPT
            assert any(r["total"] >= 1 for r in await repo.stats(1))
        finally:
            await repo.close()

    asyncio.run(go())


def test_queue_end_to_end_with_dedup():
    async def go():
        repo = await DecisionRepo.open(DB)
        stub = CountingStub()
        svc = DecisionService(stub, repo)
        rs = RedisSettings.from_dsn(REDIS)
        qname = f"arq:test:{uuid.uuid4().hex[:8]}"  # isolated queue: a burst worker must not drain other jobs
        queue = await create_pool(rs, default_queue_name=qname)
        app.dependency_overrides[get_service] = lambda: svc
        app.dependency_overrides[get_queue] = lambda: queue
        awb = f"JOB{uuid.uuid4().hex[:8]}"
        body = _req(awb).model_dump(mode="json")
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
                r1 = await c.post("/v1/ndr/jobs", json=body)
                r2 = await c.post("/v1/ndr/jobs", json=body)  # duplicate webhook delivery
                assert r1.status_code == 202 and r1.json()["status"] == "queued"
                assert r2.status_code == 200 and r2.json()["status"] == "duplicate"
                job_id = r1.json()["job_id"]
                assert (await c.get(f"/v1/ndr/jobs/{job_id}")).json()["status"] == "queued"

                async def startup(ctx):
                    ctx["service"] = svc

                worker = Worker(functions=[process_ndr], redis_settings=rs, burst=True,
                                on_startup=startup, handle_signals=False, queue_name=qname)
                await worker.async_run()
                await worker.close()

                done = (await c.get(f"/v1/ndr/jobs/{job_id}")).json()
                assert done["status"] == "complete" and done["decision"]["action"] == "reattempt"
                assert stub.calls == 1  # two submissions, one LLM call
                stats = (await c.get("/v1/stats?days=1")).json()
                assert stats["by_intent"][0]["total"] >= 1
        finally:
            app.dependency_overrides.clear()
            await queue.aclose()
            await repo.close()

    asyncio.run(go())

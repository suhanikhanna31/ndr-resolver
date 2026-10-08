from __future__ import annotations

import logging
import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path

from arq import create_pool
from arq.connections import RedisSettings
from arq.jobs import Job, JobStatus
from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.staticfiles import StaticFiles

from .config import get_settings
from .resolver import DEFAULT_POLICY
from .schemas import Decision, FeedbackRequest, NDRRequest
from .service import DecisionService, build_service
from .whatsapp import WhatsAppState, admin_router as wa_admin_router, webhook_router as wa_webhook_router

logging.basicConfig(level=logging.INFO, format="%(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    app.state.service = await build_service(s)
    app.state.wa = WhatsAppState.from_env()
    app.state.queue = await create_pool(RedisSettings.from_dsn(s.redis_url)) if s.redis_url else None
    yield
    if app.state.queue:
        await app.state.queue.aclose()
    await app.state.service.close()


app = FastAPI(title="NDR Reply Resolver", version="0.2.0", lifespan=lifespan)


def require_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = os.getenv("API_KEY")
    if expected and not (x_api_key and secrets.compare_digest(x_api_key, expected)):
        raise HTTPException(401, "invalid or missing X-API-Key")


def get_service(request: Request) -> DecisionService:
    return request.app.state.service


def get_queue(request: Request):
    if request.app.state.queue is None:
        raise HTTPException(503, "queue not configured: set REDIS_URL")
    return request.app.state.queue


router = APIRouter(prefix="/v1", dependencies=[Depends(require_key)])


@app.get("/healthz")
async def healthz():
    return {"ok": True}


@app.get("/meta")
async def meta(request: Request):
    """Public, non-sensitive capability flags so the UI knows what to enable and whether to ask for a key."""
    return {"auth_required": bool(os.getenv("API_KEY")),
            "persistence": request.app.state.service.repo is not None,
            "queue": request.app.state.queue is not None,
            "provider": os.getenv("NDR_PROVIDER", "anthropic").lower(),
            "model": getattr(request.app.state.service.extractor, "model", None)
                     or getattr(getattr(request.app.state.service.extractor, "base", None), "model", None),
            "version": app.version}


@router.get("/policy")
async def policy():
    """The live business-rule thresholds, read-only, for the UI spec sheet."""
    return DEFAULT_POLICY.model_dump()


@router.get("/decisions")
async def decisions(limit: int = Query(50, ge=1, le=200), svc: DecisionService = Depends(get_service)):
    """Recent decisions (no utterances, no payloads) for the ledger view."""
    if svc.repo is None:
        raise HTTPException(503, "ledger not configured: set DATABASE_URL")
    return {"decisions": await svc.repo.recent(limit)}


@router.post("/ndr/resolve", response_model=Decision)
async def resolve_ndr(req: NDRRequest, svc: DecisionService = Depends(get_service)) -> Decision:
    """Synchronous path for latency-sensitive callers."""
    return await svc.handle(req)


@router.post("/ndr/jobs", status_code=202)
async def submit_job(req: NDRRequest, response: Response, queue=Depends(get_queue)):
    """Async path for webhooks. Job id = awb:attempt, so duplicate deliveries are no-ops."""
    job_id = f"{req.awb}:{req.attempt_number}"
    job = await queue.enqueue_job("process_ndr", req.model_dump(mode="json"), _job_id=job_id)
    if job is None:
        response.status_code = 200
        return {"job_id": job_id, "status": "duplicate"}
    return {"job_id": job_id, "status": "queued"}


@router.get("/ndr/jobs/{job_id}")
async def job_status(job_id: str, queue=Depends(get_queue)):
    job = Job(job_id, redis=queue, _queue_name=queue.default_queue_name)
    status = await job.status()
    if status != JobStatus.complete:
        return {"job_id": job_id, "status": status.value}
    info = await job.result_info()
    if not info.success:
        return {"job_id": job_id, "status": "failed", "error": str(info.result)}
    return {"job_id": job_id, "status": "complete", "decision": info.result}


@router.post("/ndr/feedback")
async def feedback(fb: FeedbackRequest, svc: DecisionService = Depends(get_service)):
    """A human resolved (or overrode) a reply. Embed it so similar future replies get it as a few-shot example."""
    if svc.store is None:
        raise HTTPException(503, "feedback loop not configured: set DATABASE_URL")
    indexed = await svc.add_feedback(fb)
    return {"indexed": indexed, "note": None if indexed else "address text is PII and is never indexed"}


@router.get("/stats")
async def stats(days: int = 7, svc: DecisionService = Depends(get_service)):
    """Escalation rate, confidence and p95 latency per intent, straight from SQL."""
    if svc.repo is None:
        raise HTTPException(503, "stats not configured: set DATABASE_URL")
    return {"days": days, "by_intent": await svc.repo.stats(days)}


app.include_router(router)
app.include_router(wa_webhook_router)  # authenticated by Meta's HMAC signature, not X-API-Key
app.include_router(wa_admin_router, dependencies=[Depends(require_key)])

# Frontend: plain static files, no build step. Mounted last so API routes always win.
app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="ui")

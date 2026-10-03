"""arq worker. Run:  arq app.worker.WorkerSettings"""
from __future__ import annotations

import psycopg
from arq.connections import RedisSettings
from arq.worker import Retry

from .config import get_settings
from .schemas import NDRRequest
from .service import build_service


async def startup(ctx: dict) -> None:
    ctx["service"] = await build_service()


async def shutdown(ctx: dict) -> None:
    await ctx["service"].close()


async def process_ndr(ctx: dict, payload: dict) -> dict:
    req = NDRRequest.model_validate(payload)
    try:
        decision = await ctx["service"].handle(req)
    except psycopg.OperationalError:  # DB blip or pool timeout: retry with backoff (resolve() itself never raises)
        raise Retry(defer=ctx["job_try"] * 5)
    return decision.model_dump(mode="json")


class WorkerSettings:
    functions = [process_ndr]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = RedisSettings.from_dsn(get_settings().redis_url or "redis://localhost:6379")
    max_jobs = 10        # concurrent LLM calls per worker
    job_timeout = 30
    max_tries = 3
    keep_result = 3600   # arq's job-id dedup window; the DB primary key covers everything older

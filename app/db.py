from __future__ import annotations

from typing import Any

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from .schemas import Decision, NDRRequest

# (awb, attempt_number) is the idempotency key: a retried webhook can never double-act.
SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS decisions (
    awb            TEXT        NOT NULL,
    attempt_number INT         NOT NULL,
    action         TEXT        NOT NULL,
    needs_human    BOOLEAN     NOT NULL,
    intent         TEXT,
    confidence     REAL,
    latency_ms     INT,
    decision       JSONB       NOT NULL,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (awb, attempt_number)
);
CREATE INDEX IF NOT EXISTS decisions_created_at_idx ON decisions (created_at);
"""

STATS_SQL = """
SELECT COALESCE(intent, 'extraction_failed')                                  AS intent,
       count(*)                                                               AS total,
       count(*) FILTER (WHERE action = 'escalate_human')                      AS escalated,
       count(*) FILTER (WHERE needs_human AND action <> 'escalate_human')     AS human_review,
       round(avg(confidence)::numeric, 3)::float                              AS avg_confidence,
       round(percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms)::numeric)::int AS p95_latency_ms
FROM decisions
WHERE created_at >= now() - make_interval(days => %s)
GROUP BY 1
ORDER BY total DESC
"""


RECENT_SQL = """
SELECT awb, attempt_number, action, needs_human, intent, confidence, latency_ms, created_at,
       COALESCE(decision->'reasons', '[]'::jsonb) AS reasons,
       decision->>'reattempt_date' AS reattempt_date
FROM decisions
ORDER BY created_at DESC
LIMIT %s
"""


class DecisionRepo:
    def __init__(self, pool: AsyncConnectionPool):
        self.pool = pool

    @classmethod
    async def open(cls, url: str) -> "DecisionRepo":
        pool = AsyncConnectionPool(url, min_size=1, max_size=5, open=False, kwargs={"autocommit": True})
        await pool.open(wait=True, timeout=10)
        async with pool.connection() as conn:
            await conn.execute(SCHEMA)
        return cls(pool)

    async def close(self) -> None:
        await self.pool.close()

    async def table_exists(self, name: str) -> bool:
        async with self.pool.connection() as conn:
            cur = await conn.execute("SELECT to_regclass(%s) IS NOT NULL", (f"public.{name}",))
            return (await cur.fetchone())[0]

    async def get(self, awb: str, attempt: int) -> Decision | None:
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                "SELECT decision FROM decisions WHERE awb = %s AND attempt_number = %s", (awb, attempt))
            row = await cur.fetchone()
        return Decision.model_validate(row[0]) if row else None

    async def save(self, req: NDRRequest, d: Decision, latency_ms: int) -> Decision:
        """Insert once. If a concurrent call already stored this (awb, attempt), return the stored one."""
        ext = d.extraction
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                """INSERT INTO decisions
                       (awb, attempt_number, action, needs_human, intent, confidence, latency_ms, decision)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT (awb, attempt_number) DO NOTHING RETURNING 1""",
                (req.awb, req.attempt_number, d.action.value, d.needs_human,
                 ext.intent.value if ext else None, ext.confidence if ext else None,
                 latency_ms, Jsonb(d.model_dump(mode="json"))))
            inserted = await cur.fetchone()
        if inserted:
            return d
        return await self.get(req.awb, req.attempt_number) or d

    async def stats(self, days: int = 7) -> list[dict[str, Any]]:
        async with self.pool.connection() as conn:
            cur = conn.cursor(row_factory=dict_row)
            await cur.execute(STATS_SQL, (days,))
            return await cur.fetchall()

    async def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        """Newest decisions for the ledger view. Deliberately omits payload/extraction (can hold PII)."""
        async with self.pool.connection() as conn:
            cur = conn.cursor(row_factory=dict_row)
            await cur.execute(RECENT_SQL, (limit,))
            rows = await cur.fetchall()
        for r in rows:
            r["created_at"] = r["created_at"].isoformat()
        return rows

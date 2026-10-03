import json
import logging
import time

from fastapi import Depends, FastAPI

from .llm import AnthropicExtractor, Extractor
from .resolver import resolve
from .schemas import Decision, NDRRequest

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("ndr")
app = FastAPI(title="NDR Reply Resolver", version="0.1.0")

_extractor: Extractor | None = None


def get_extractor() -> Extractor:
    global _extractor
    if _extractor is None:
        _extractor = AnthropicExtractor()
    return _extractor


@app.get("/healthz")
def healthz():
    return {"ok": True}


# Sync endpoint on purpose: FastAPI runs it in a threadpool, which suits the sync SDK client.
@app.post("/v1/ndr/resolve", response_model=Decision)
def resolve_ndr(req: NDRRequest, extractor: Extractor = Depends(get_extractor)) -> Decision:
    t0 = time.perf_counter()
    decision = resolve(req, extractor)
    # Log decisions, never the raw utterance: it can contain addresses and phone numbers (PII).
    log.info(json.dumps({
        "awb": req.awb, "action": decision.action.value, "needs_human": decision.needs_human,
        "reasons": decision.reasons, "latency_ms": round((time.perf_counter() - t0) * 1000),
    }))
    return decision

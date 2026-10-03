from __future__ import annotations

import logging
import re

from pydantic import BaseModel

from .dates import resolve_date
from .llm import Extractor
from .schemas import IST, CarrierAction, Decision, Extraction, Intent, NDRRequest

log = logging.getLogger("ndr")

PINCODE = re.compile(r"\b[1-9]\d{5}\b")
IN_MOBILE = re.compile(r"^[6-9]\d{9}$")


class Policy(BaseModel):
    min_confidence: float = 0.75              # reversible actions (reattempt, contact fixes)
    min_confidence_irreversible: float = 0.90  # RTO cannot be undone
    max_reattempt_days: int = 3                # carriers rarely honour dates further out
    max_attempts: int = 3
    auto_push_address: bool = False            # address changes are a fraud vector: human verifies


DEFAULT_POLICY = Policy()


def _escalate(req: NDRRequest, ext: Extraction | None, *reasons: str) -> Decision:
    return Decision(awb=req.awb, action=CarrierAction.ESCALATE_HUMAN, needs_human=True,
                    reasons=list(reasons), extraction=ext)


def resolve(req: NDRRequest, extractor: Extractor, policy: Policy = DEFAULT_POLICY) -> Decision:
    try:
        ext = extractor.extract(req)
    except Exception as e:  # API down, schema violation, no tool call: always fail safe
        log.warning("extraction_failed awb=%s err=%s", req.awb, type(e).__name__)
        return _escalate(req, None, f"extraction_failed:{type(e).__name__}")
    return decide(req, ext, policy)


def decide(req: NDRRequest, ext: Extraction, policy: Policy = DEFAULT_POLICY) -> Decision:
    """Pure, deterministic policy layer. Every business rule lives here, not in the prompt."""
    today = req.received_at.astimezone(IST).date()
    i = ext.intent

    if i is Intent.OPT_OUT_OR_ABUSE:
        return _escalate(req, ext, "customer_opt_out_or_abuse")
    if i is Intent.UNCLEAR:
        return _escalate(req, ext, "unclear_intent")
    if i is Intent.ALREADY_RECEIVED:
        return _escalate(req, ext, "possible_fake_delivery_check_pod")

    floor = policy.min_confidence_irreversible if i is Intent.REFUSE_DELIVERY else policy.min_confidence
    if ext.confidence < floor:
        return _escalate(req, ext, f"low_confidence:{ext.confidence:.2f}<{floor}")

    if i is Intent.REFUSE_DELIVERY:
        return Decision(awb=req.awb, action=CarrierAction.RTO, needs_human=False, extraction=ext)

    if i is Intent.RESCHEDULE:
        if req.attempt_number >= policy.max_attempts:
            return _escalate(req, ext, "max_attempts_reached")
        d = resolve_date(ext.date_expression, today)
        if d is None:
            return _escalate(req, ext, "reschedule_without_date")
        if (d - today).days > policy.max_reattempt_days:
            return _escalate(req, ext, f"date_outside_window:{d.isoformat()}")
        return Decision(awb=req.awb, action=CarrierAction.REATTEMPT, reattempt_date=d,
                        time_slot=ext.time_slot, needs_human=False, extraction=ext)

    if i is Intent.ADDRESS_CORRECTION:
        addr = (ext.new_address or "").strip()
        if len(addr) < 15 or not PINCODE.search(addr):
            return _escalate(req, ext, "address_incomplete_or_missing_pincode")
        return Decision(awb=req.awb, action=CarrierAction.UPDATE_ADDRESS, payload={"address": addr},
                        needs_human=not policy.auto_push_address, extraction=ext)

    if i is Intent.PHONE_CORRECTION:
        digits = re.sub(r"\D", "", ext.new_phone or "")[-10:]
        if not IN_MOBILE.match(digits):
            return _escalate(req, ext, "invalid_phone_number")
        return Decision(awb=req.awb, action=CarrierAction.UPDATE_PHONE, payload={"phone": digits},
                        needs_human=False, extraction=ext)

    return _escalate(req, ext, "unhandled_intent")

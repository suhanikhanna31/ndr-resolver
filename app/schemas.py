from __future__ import annotations

import re
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

# India has no DST, so a fixed offset avoids a tzdata dependency in slim containers.
IST = timezone(timedelta(hours=5, minutes=30))

# The LLM may only emit dates in this tiny grammar. Calendar math happens in code.
DATE_GRAMMAR = re.compile(
    r"^(today|tomorrow|day_after_tomorrow"
    r"|weekday:(monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
    r"|in_days:\d{1,2}"
    r"|day_of_month:([1-9]|[12]\d|3[01]))$"
)


class Intent(str, Enum):
    RESCHEDULE = "reschedule"
    ADDRESS_CORRECTION = "address_correction"
    PHONE_CORRECTION = "phone_correction"
    REFUSE_DELIVERY = "refuse_delivery"
    ALREADY_RECEIVED = "already_received"
    UNCLEAR = "unclear"
    OPT_OUT_OR_ABUSE = "opt_out_or_abuse"


class CarrierAction(str, Enum):
    REATTEMPT = "reattempt"
    UPDATE_ADDRESS = "update_address"
    UPDATE_PHONE = "update_phone"
    RTO = "rto"
    ESCALATE_HUMAN = "escalate_human"


class Extraction(BaseModel):
    """What the LLM is allowed to produce: understanding only, never decisions."""

    intent: Intent
    confidence: float = Field(ge=0.0, le=1.0)
    date_expression: Optional[str] = None
    time_slot: Optional[Literal["morning", "afternoon", "evening"]] = None
    new_address: Optional[str] = None
    new_phone: Optional[str] = None
    reasoning: str = ""

    @field_validator("date_expression")
    @classmethod
    def _grammar(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not DATE_GRAMMAR.match(v):
            raise ValueError(f"date_expression outside grammar: {v!r}")
        return v


class NDRRequest(BaseModel):
    awb: str
    ndr_reason: str = Field(description="Normalised carrier NDR reason, e.g. customer_unavailable")
    customer_utterance: str = Field(max_length=2000)
    attempt_number: int = Field(default=1, ge=1)
    received_at: datetime = Field(default_factory=lambda: datetime.now(IST))

    @field_validator("received_at")
    @classmethod
    def _tz(cls, v: datetime) -> datetime:
        return v if v.tzinfo else v.replace(tzinfo=IST)


class Decision(BaseModel):
    awb: str
    action: CarrierAction
    reattempt_date: Optional[date] = None
    time_slot: Optional[str] = None
    payload: dict = Field(default_factory=dict)
    needs_human: bool
    reasons: list[str] = Field(default_factory=list)
    extraction: Optional[Extraction] = None

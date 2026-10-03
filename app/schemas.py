from __future__ import annotations

import re
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

# India has no DST, so a fixed offset avoids a tzdata dependency in slim containers.
IST = timezone(timedelta(hours=5, minutes=30))

# The LLM may only emit dates in this tiny grammar. Calendar math happens in code.
DATE_GRAMMAR = re.compile(
    r"^(today|tomorrow|day_after_tomorrow"
    r"|weekday:(monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
    r"|in_days:\d{1,2}"
    r"|day_of_month:([1-9]|[12]\d|3[01]))$"
)


AWB_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"  # also used in queue job ids and URLs


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
    awb: str = Field(pattern=AWB_PATTERN)
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


class ResolvedExample(BaseModel):
    """A past reply a human operator resolved, retrieved as a few-shot example."""

    utterance: str  # already redacted
    ndr_reason: str
    intent: Intent
    date_expression: Optional[str] = None
    time_slot: Optional[Literal["morning", "afternoon", "evening"]] = None
    distance: float = 0.0


class FeedbackRequest(BaseModel):
    """A human operator's resolution of a reply. This is what feeds the retrieval loop."""

    awb: str = Field(pattern=AWB_PATTERN)
    ndr_reason: str
    customer_utterance: str = Field(max_length=2000)
    intent: Intent
    date_expression: Optional[str] = None
    time_slot: Optional[Literal["morning", "afternoon", "evening"]] = None

    @field_validator("date_expression")
    @classmethod
    def _grammar(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not DATE_GRAMMAR.match(v):
            raise ValueError(f"date_expression outside grammar: {v!r}")
        return v

    @model_validator(mode="after")
    def _reschedule_needs_date(self):
        if self.intent is Intent.RESCHEDULE and not self.date_expression:
            raise ValueError("a reschedule resolution must include date_expression")
        return self

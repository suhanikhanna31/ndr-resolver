import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.main import app, get_extractor
from app.resolver import decide, resolve
from app.schemas import CarrierAction, Extraction, Intent, NDRRequest

REQ = dict(awb="T1", ndr_reason="customer_unavailable", customer_utterance="x",
           received_at="2026-10-03T11:00:00+05:30")


def ext(intent, conf=0.95, **kw):
    return Extraction(intent=intent, confidence=conf, **kw)


def test_reschedule_happy_path():
    d = decide(NDRRequest(**REQ), ext(Intent.RESCHEDULE, date_expression="tomorrow", time_slot="evening"))
    assert d.action is CarrierAction.REATTEMPT and str(d.reattempt_date) == "2026-10-04" and not d.needs_human


def test_low_confidence_escalates():
    d = decide(NDRRequest(**REQ), ext(Intent.RESCHEDULE, 0.5, date_expression="tomorrow"))
    assert d.action is CarrierAction.ESCALATE_HUMAN


def test_rto_needs_higher_confidence_than_reattempt():
    assert decide(NDRRequest(**REQ), ext(Intent.REFUSE_DELIVERY, 0.85)).action is CarrierAction.ESCALATE_HUMAN
    assert decide(NDRRequest(**REQ), ext(Intent.REFUSE_DELIVERY, 0.95)).action is CarrierAction.RTO


def test_date_outside_window_and_max_attempts():
    far = ext(Intent.RESCHEDULE, date_expression="day_of_month:15")
    assert decide(NDRRequest(**REQ), far).action is CarrierAction.ESCALATE_HUMAN
    third = NDRRequest(**{**REQ, "attempt_number": 3})
    assert decide(third, ext(Intent.RESCHEDULE, date_expression="tomorrow")).action is CarrierAction.ESCALATE_HUMAN


def test_address_requires_pincode_and_human_review():
    bad = ext(Intent.ADDRESS_CORRECTION, new_address="near the temple, main road")
    assert decide(NDRRequest(**REQ), bad).action is CarrierAction.ESCALATE_HUMAN
    good = ext(Intent.ADDRESS_CORRECTION, new_address="Flat 402, Green Park, Sector 21, Noida 201301")
    d = decide(NDRRequest(**REQ), good)
    assert d.action is CarrierAction.UPDATE_ADDRESS and d.needs_human


def test_phone_validated_in_code():
    assert decide(NDRRequest(**REQ), ext(Intent.PHONE_CORRECTION, new_phone="98765")).needs_human
    d = decide(NDRRequest(**REQ), ext(Intent.PHONE_CORRECTION, new_phone="+91 98765 43210"))
    assert d.payload == {"phone": "9876543210"}


def test_llm_date_outside_grammar_is_rejected():
    with pytest.raises(ValidationError):
        Extraction(intent=Intent.RESCHEDULE, confidence=0.9, date_expression="2026-10-04")


def test_extractor_failure_fails_safe():
    class Boom:
        def extract(self, req):
            raise RuntimeError("api down")

    d = resolve(NDRRequest(**REQ), Boom())
    assert d.action is CarrierAction.ESCALATE_HUMAN and "extraction_failed" in d.reasons[0]


def test_api_endpoint():
    class Stub:
        def extract(self, req):
            return ext(Intent.RESCHEDULE, date_expression="tomorrow")

    app.dependency_overrides[get_extractor] = lambda: Stub()
    r = TestClient(app).post("/v1/ndr/resolve", json=REQ)
    app.dependency_overrides.clear()
    assert r.status_code == 200 and r.json()["action"] == "reattempt"

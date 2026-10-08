import hashlib
import hmac
import json
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from app import whatsapp
from app.main import app
from app.schemas import IST, Extraction, Intent
from app.service import DecisionService
from app.whatsapp import InMemoryShipments, Shipment, WhatsAppState, parse_inbound, verify_signature

SECRET = "app-secret"
PHONE = "919876543210"


class FakeSender:
    def __init__(self):
        self.sent = []

    async def send_text(self, to, body):
        self.sent.append((to, body))


class StubExtractor:
    def __init__(self, ext=None, boom=False):
        self.ext, self.boom = ext, boom

    def extract(self, req):
        if self.boom:
            raise RuntimeError("llm down")
        return self.ext


TOMORROW_EVENING = Extraction(intent=Intent.RESCHEDULE, confidence=0.95, date_expression="tomorrow", time_slot="evening")


def payload(text="kal shaam bhej do", msg_id="wamid.1", kind="text", context_id=None, wa_id=PHONE):
    msg = {"id": msg_id, "from": wa_id, "type": kind}
    if kind == "text":
        msg["text"] = {"body": text}
    if context_id:
        msg["context"] = {"id": context_id}
    return {"entry": [{"changes": [{"value": {"messages": [msg]}}]}]}


def post(client, body: dict, secret=SECRET, sig=None):
    raw = json.dumps(body).encode()
    sig = sig if sig is not None else "sha256=" + hmac.new(secret.encode(), raw, hashlib.sha256).hexdigest()
    return client.post("/webhooks/whatsapp", content=raw,
                       headers={"x-hub-signature-256": sig, "content-type": "application/json"})


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("WHATSAPP_APP_SECRET", SECRET)
    monkeypatch.setenv("WHATSAPP_VERIFY_TOKEN", "verify-me")
    sender, ships = FakeSender(), InMemoryShipments()
    ships.register(Shipment("AWB1", PHONE, "customer_unavailable"), message_id="wamid.out1")
    wa = WhatsAppState(ships, sender)

    def build(extractor):
        app.dependency_overrides[whatsapp.get_wa] = lambda: wa
        app.dependency_overrides[whatsapp.get_service] = lambda: DecisionService(extractor)
        return TestClient(app)

    yield build, sender, wa
    app.dependency_overrides.clear()


# ---- pure functions ----

def test_signature_accepts_correct_and_rejects_everything_else():
    body = b'{"a":1}'
    good = "sha256=" + hmac.new(b"s", body, hashlib.sha256).hexdigest()
    assert verify_signature(body, good, "s")
    assert not verify_signature(body, good, "other-secret")
    assert not verify_signature(b'{"a":2}', good, "s")
    assert not verify_signature(body, None, "s")
    assert not verify_signature(body, good.removeprefix("sha256="), "s")


def test_parse_ignores_status_callbacks_and_reads_context():
    status = {"entry": [{"changes": [{"value": {"statuses": [{"id": "x", "status": "read"}]}}]}]}
    assert parse_inbound(status) == []
    m = parse_inbound(payload(context_id="wamid.out1"))[0]
    assert (m.message_id, m.wa_id, m.kind, m.text, m.context_id) == ("wamid.1", PHONE, "text", "kal shaam bhej do", "wamid.out1")


# ---- handshake ----

def test_verify_handshake(env):
    build, *_ = env
    c = build(StubExtractor())
    ok = c.get("/webhooks/whatsapp", params={"hub.mode": "subscribe", "hub.verify_token": "verify-me", "hub.challenge": "123"})
    assert ok.status_code == 200 and ok.text == "123"
    bad = c.get("/webhooks/whatsapp", params={"hub.mode": "subscribe", "hub.verify_token": "nope", "hub.challenge": "123"})
    assert bad.status_code == 403


def test_verify_fails_closed_when_token_unset(env, monkeypatch):
    build, *_ = env
    monkeypatch.delenv("WHATSAPP_VERIFY_TOKEN")
    r = build(StubExtractor()).get("/webhooks/whatsapp", params={"hub.mode": "subscribe", "hub.verify_token": "", "hub.challenge": "1"})
    assert r.status_code == 503


# ---- signature gate ----

def test_unsigned_or_badly_signed_post_is_rejected_and_nothing_is_sent(env):
    build, sender, _ = env
    c = build(StubExtractor(TOMORROW_EVENING))
    assert post(c, payload(), sig="sha256=deadbeef").status_code == 401
    assert post(c, payload(), secret="wrong").status_code == 401
    assert sender.sent == []


def test_post_fails_closed_when_secret_unset(env, monkeypatch):
    build, sender, _ = env
    monkeypatch.delenv("WHATSAPP_APP_SECRET")
    assert post(build(StubExtractor(TOMORROW_EVENING)), payload()).status_code == 503
    assert sender.sent == []


# ---- behaviour ----

def test_reschedule_reply_goes_through_the_real_policy(env):
    build, sender, _ = env
    r = post(build(StubExtractor(TOMORROW_EVENING)), payload(context_id="wamid.out1"))
    assert r.status_code == 200 and r.json() == {"received": 1}
    expected = (datetime.now(IST).date() + timedelta(days=1)).strftime("%d %b %Y")
    assert sender.sent == [(PHONE, f"Noted. We will try to deliver your order on {expected} (evening).")]


def test_duplicate_delivery_is_processed_once(env):
    build, sender, _ = env
    c = build(StubExtractor(TOMORROW_EVENING))
    assert post(c, payload()).json() == {"received": 1}
    assert post(c, payload()).json() == {"received": 0}
    assert len(sender.sent) == 1


def test_low_confidence_never_promises_an_action(env):
    build, sender, _ = env
    ext = Extraction(intent=Intent.REFUSE_DELIVERY, confidence=0.80)  # below the 0.90 floor for RTO
    post(build(StubExtractor(ext)), payload(text="maybe cancel it"))
    assert "returned" not in sender.sent[0][1] and "review" in sender.sent[0][1]


def test_llm_failure_escalates_instead_of_going_silent(env):
    build, sender, _ = env
    post(build(StubExtractor(boom=True)), payload())
    assert len(sender.sent) == 1 and "review" in sender.sent[0][1]


def test_unknown_sender_gets_no_reply(env):
    build, sender, _ = env
    post(build(StubExtractor(TOMORROW_EVENING)), payload(wa_id="911111111111"))
    assert sender.sent == []


def test_reply_context_from_a_different_number_is_not_matched(env):
    build, sender, _ = env
    post(build(StubExtractor(TOMORROW_EVENING)), payload(wa_id="911111111111", context_id="wamid.out1"))
    assert sender.sent == []


def test_voice_note_gets_a_text_only_prompt(env):
    build, sender, _ = env
    post(build(StubExtractor(TOMORROW_EVENING)), payload(kind="audio"))
    assert sender.sent == [(PHONE, whatsapp.UNSUPPORTED_REPLY)]


def test_status_callbacks_are_acknowledged_without_work(env):
    build, sender, _ = env
    status = {"entry": [{"changes": [{"value": {"statuses": [{"id": "x", "status": "delivered"}]}}]}]}
    assert post(build(StubExtractor()), status).json() == {"received": 0}
    assert sender.sent == []


# ---- shipment registration ----

def test_registration_requires_api_key_when_set(env, monkeypatch):
    build, _, wa = env
    monkeypatch.setenv("API_KEY", "k")
    c = build(StubExtractor())
    body = {"awb": "AWB9", "wa_id": "919000000001", "ndr_reason": "customer_unavailable", "message_id": "wamid.out9"}
    assert c.post("/v1/whatsapp/shipments", json=body).status_code == 401
    assert c.post("/v1/whatsapp/shipments", json=body, headers={"x-api-key": "k"}).status_code == 201
    assert wa.shipments.find("919000000001", "wamid.out9").awb == "AWB9"


def test_registration_rejects_malformed_phone(env):
    build, *_ = env
    r = build(StubExtractor()).post("/v1/whatsapp/shipments",
                                    json={"awb": "A", "wa_id": "+91 98765", "ndr_reason": "x"})
    assert r.status_code == 422

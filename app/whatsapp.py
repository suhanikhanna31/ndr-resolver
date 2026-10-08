"""WhatsApp Cloud API adapter: a customer replies to an NDR message on WhatsApp, the existing resolver decides,
and the customer gets a confirmation. No decision logic lives here; this file only translates channels.

    Meta  --POST /webhooks/whatsapp-->  verify signature -> find shipment -> DecisionService.handle -> reply

A WhatsApp message carries no AWB, so the outbound NDR sender registers (awb, phone) through
POST /v1/whatsapp/shipments when it sends the template message. Replies are matched by the id of the message
being replied to (context.id) first, then by phone number.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import secrets
from collections import OrderedDict
from dataclasses import dataclass
from typing import Protocol

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from .schemas import AWB_PATTERN, CarrierAction, Decision, NDRRequest
from .service import DecisionService

log = logging.getLogger("ndr")

GRAPH_URL = "https://graph.facebook.com/v21.0/{phone_number_id}/messages"
MAX_SEEN = 10_000  # bounded memory for webhook de-duplication


# ---------- signature + parsing (pure functions) ----------

def verify_signature(raw_body: bytes, header: str | None, app_secret: str) -> bool:
    """Meta signs the raw body: X-Hub-Signature-256 = 'sha256=' + HMAC_SHA256(app_secret, body)."""
    if not header or not header.startswith("sha256="):
        return False
    expected = hmac.new(app_secret.encode(), raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, header.removeprefix("sha256="))


@dataclass(frozen=True)
class Inbound:
    message_id: str
    wa_id: str            # sender phone in international format, no '+'
    kind: str             # 'text', 'audio', 'image', ...
    text: str | None
    context_id: str | None  # id of the outbound message the customer replied to, if any


def parse_inbound(payload: dict) -> list[Inbound]:
    """Extract customer messages. Status callbacks (sent/delivered/read) have no 'messages' and are ignored."""
    out: list[Inbound] = []
    for entry in payload.get("entry") or []:
        for change in entry.get("changes") or []:
            for m in (change.get("value") or {}).get("messages") or []:
                if not m.get("id") or not m.get("from"):
                    continue
                kind = m.get("type", "unknown")
                text = (m.get("text") or {}).get("body") if kind == "text" else None
                out.append(Inbound(m["id"], m["from"], kind, text, (m.get("context") or {}).get("id")))
    return out


def reply_text(d: Decision) -> str:
    """What the customer is told. Only states what was actually done; anything uncertain goes to a human."""
    if d.action is CarrierAction.UPDATE_ADDRESS:
        return "Thanks. Our team will verify the new address and confirm with you shortly."
    if d.needs_human or d.action is CarrierAction.ESCALATE_HUMAN:
        return "Thanks for your reply. Our delivery team will review it and contact you shortly."
    if d.action is CarrierAction.REATTEMPT and d.reattempt_date:
        slot = f" ({d.time_slot})" if d.time_slot else ""
        return f"Noted. We will try to deliver your order on {d.reattempt_date:%d %b %Y}{slot}."
    if d.action is CarrierAction.RTO:
        return "Understood. This order will be returned to the sender."
    if d.action is CarrierAction.UPDATE_PHONE:
        return "Thanks. We have updated your contact number for this delivery."
    return "Thanks for your reply. Our delivery team will contact you shortly."


UNSUPPORTED_REPLY = "Sorry, I can only read text messages right now. Please type your reply."


# ---------- shipment lookup + sender (swappable) ----------

@dataclass(frozen=True)
class Shipment:
    awb: str
    wa_id: str
    ndr_reason: str
    attempt_number: int = 1


class ShipmentLookup(Protocol):
    def register(self, s: Shipment, message_id: str | None = None) -> None: ...
    def find(self, wa_id: str, context_id: str | None) -> Shipment | None: ...


class InMemoryShipments:
    """Default lookup. Lost on restart: swap for a Postgres-backed one in production (same two methods)."""

    def __init__(self) -> None:
        self._by_message: dict[str, Shipment] = {}
        self._by_phone: dict[str, Shipment] = {}

    def register(self, s: Shipment, message_id: str | None = None) -> None:
        if message_id:
            self._by_message[message_id] = s
        self._by_phone[s.wa_id] = s  # newest NDR for this phone wins

    def find(self, wa_id: str, context_id: str | None) -> Shipment | None:
        if context_id and context_id in self._by_message:
            s = self._by_message[context_id]
            return s if s.wa_id == wa_id else None  # a reply must come from the number we messaged
        return self._by_phone.get(wa_id)


class Sender(Protocol):
    async def send_text(self, to: str, body: str) -> None: ...


class CloudApiSender:
    def __init__(self, token: str, phone_number_id: str, client=None):
        import httpx

        self._token, self._pnid = token, phone_number_id
        self._client = client or httpx.AsyncClient(timeout=10.0)

    async def send_text(self, to: str, body: str) -> None:
        # Free-form text is allowed because the customer messaged first (24-hour service window).
        r = await self._client.post(
            GRAPH_URL.format(phone_number_id=self._pnid),
            headers={"Authorization": f"Bearer {self._token}"},
            json={"messaging_product": "whatsapp", "to": to, "type": "text", "text": {"body": body}},
        )
        r.raise_for_status()


class LogOnlySender:
    """Used when no access token is configured: the decision still runs, nothing is sent."""

    async def send_text(self, to: str, body: str) -> None:
        log.info(json.dumps({"whatsapp": "send_skipped", "reason": "WHATSAPP_ACCESS_TOKEN not set"}))


class WhatsAppState:
    def __init__(self, shipments: ShipmentLookup, sender: Sender):
        self.shipments, self.sender = shipments, sender
        self._seen: OrderedDict[str, None] = OrderedDict()

    @classmethod
    def from_env(cls) -> "WhatsAppState":
        token, pnid = os.getenv("WHATSAPP_ACCESS_TOKEN"), os.getenv("WHATSAPP_PHONE_NUMBER_ID")
        sender: Sender = CloudApiSender(token, pnid) if token and pnid else LogOnlySender()
        return cls(InMemoryShipments(), sender)

    def first_time(self, message_id: str) -> bool:
        """Meta retries deliveries; a message id is processed once."""
        if message_id in self._seen:
            return False
        self._seen[message_id] = None
        if len(self._seen) > MAX_SEEN:
            self._seen.popitem(last=False)
        return True


# ---------- routes ----------

webhook_router = APIRouter()  # public: Meta cannot send our API key, so it is authenticated by signature
admin_router = APIRouter(prefix="/v1/whatsapp")  # mounted behind X-API-Key in main.py


def get_wa(request: Request) -> WhatsAppState:
    return request.app.state.wa


def get_service(request: Request) -> DecisionService:
    return request.app.state.service


@webhook_router.get("/webhooks/whatsapp")
async def verify(mode: str | None = Query(None, alias="hub.mode"),
                 token: str | None = Query(None, alias="hub.verify_token"),
                 challenge: str | None = Query(None, alias="hub.challenge")):
    """Meta's one-time subscription handshake."""
    expected = os.getenv("WHATSAPP_VERIFY_TOKEN")
    if not expected:
        raise HTTPException(503, "WHATSAPP_VERIFY_TOKEN not configured")
    if mode == "subscribe" and token and challenge and secrets.compare_digest(token, expected):
        return PlainTextResponse(challenge)
    raise HTTPException(403, "verification failed")


@webhook_router.post("/webhooks/whatsapp")
async def receive(request: Request, background: BackgroundTasks,
                  wa: WhatsAppState = Depends(get_wa), svc: DecisionService = Depends(get_service)):
    secret = os.getenv("WHATSAPP_APP_SECRET")
    if not secret:  # fail closed: never accept unsigned traffic because of a missing setting
        raise HTTPException(503, "WHATSAPP_APP_SECRET not configured")
    raw = await request.body()
    if not verify_signature(raw, request.headers.get("x-hub-signature-256"), secret):
        raise HTTPException(401, "invalid signature")
    try:
        payload = json.loads(raw)
    except ValueError:
        raise HTTPException(400, "body is not JSON")

    queued = 0
    for msg in parse_inbound(payload):
        if not wa.first_time(msg.message_id):
            continue
        background.add_task(_process, msg, wa, svc)  # Meta needs a fast 200; the LLM call happens after
        queued += 1
    return {"received": queued}


async def _process(msg: Inbound, wa: WhatsAppState, svc: DecisionService) -> None:
    try:
        if msg.kind != "text" or not msg.text:
            await wa.sender.send_text(msg.wa_id, UNSUPPORTED_REPLY)
            return
        ship = wa.shipments.find(msg.wa_id, msg.context_id)
        if ship is None:  # unknown sender: do not guess an AWB and do not message strangers
            log.info(json.dumps({"whatsapp": "no_shipment_for_sender"}))
            return
        req = NDRRequest(awb=ship.awb, ndr_reason=ship.ndr_reason, attempt_number=ship.attempt_number,
                         customer_utterance=msg.text[:2000])
        decision = await svc.handle(req)
        await wa.sender.send_text(msg.wa_id, reply_text(decision))
    except Exception as e:  # background task: never raise into the server, never log the utterance
        log.warning(json.dumps({"whatsapp": "processing_failed", "err": type(e).__name__}))


class ShipmentIn(BaseModel):
    awb: str = Field(pattern=AWB_PATTERN)
    wa_id: str = Field(pattern=r"^\d{8,15}$", description="International format, digits only (919876543210)")
    ndr_reason: str
    attempt_number: int = Field(default=1, ge=1)
    message_id: str | None = Field(default=None, description="wamid of the outbound NDR message, for reply matching")


@admin_router.post("/shipments", status_code=201)
async def register_shipment(body: ShipmentIn, wa: WhatsAppState = Depends(get_wa)):
    """Called by whatever sends the outbound NDR template, so replies can be tied back to an AWB."""
    wa.shipments.register(Shipment(body.awb, body.wa_id, body.ndr_reason, body.attempt_number), body.message_id)
    return {"registered": body.awb}

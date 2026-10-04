import json

import httpx
import pytest

from app.llm import GEMINI_RESPONSE_SCHEMA, GeminiExtractor, make_extractor
from app.schemas import Intent, NDRRequest

REQ = NDRRequest(awb="G1", ndr_reason="customer_unavailable", customer_utterance="kal shaam ko bhej do",
                 received_at="2026-10-03T11:00:00+05:30")
GOOD = {"intent": "reschedule", "confidence": 0.92, "date_expression": "tomorrow", "time_slot": "evening",
        "new_address": None, "new_phone": None, "reasoning": "kal shaam"}


def reply(obj):
    text = obj if isinstance(obj, str) else json.dumps(obj)
    return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": text}]}}]})


def extractor(*responses, **kw):
    """GeminiExtractor over a scripted fake HTTP transport. Returns (extractor, recorded requests)."""
    seen, queue = [], list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return queue.pop(0)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    return GeminiExtractor(api_key="k", client=client, sleep=lambda s: None, **kw), seen


def test_request_shape_and_key_in_header_not_url():
    ex, seen = extractor(reply(GOOD))
    out = ex.extract(REQ)
    assert out.intent is Intent.RESCHEDULE and out.date_expression == "tomorrow"
    req = seen[0]
    assert req.headers["x-goog-api-key"] == "k" and "key=" not in str(req.url)
    assert str(req.url).endswith(":generateContent")
    body = json.loads(req.content)
    assert {"systemInstruction", "contents", "generationConfig"} <= set(body)
    assert body["generationConfig"]["responseMimeType"] == "application/json"
    assert "kal shaam ko bhej do" in body["contents"][0]["parts"][0]["text"]


def test_schema_enum_tracks_the_intent_enum():
    assert GEMINI_RESPONSE_SCHEMA["properties"]["intent"]["enum"] == [i.value for i in Intent]


def test_malformed_then_good_retries_once():
    ex, seen = extractor(reply("not json"), reply(GOOD))
    assert ex.extract(REQ).intent is Intent.RESCHEDULE and len(seen) == 2


def test_date_outside_grammar_is_rejected_then_gives_up():
    bad = {**GOOD, "date_expression": "2026-10-04"}
    ex, seen = extractor(reply(bad), reply(bad))
    with pytest.raises(Exception):
        ex.extract(REQ)
    assert len(seen) == 2


def test_empty_candidates_is_an_error_not_a_guess():
    blocked = httpx.Response(200, json={"promptFeedback": {"blockReason": "SAFETY"}})
    ex, _ = extractor(blocked, blocked)
    with pytest.raises(ValueError):
        ex.extract(REQ)


def test_rate_limit_backs_off_then_succeeds():
    ex, seen = extractor(httpx.Response(429), httpx.Response(503), reply(GOOD))
    assert ex.extract(REQ).intent is Intent.RESCHEDULE and len(seen) == 3


def test_persistent_rate_limit_raises_so_resolver_escalates():
    ex, seen = extractor(*[httpx.Response(429)] * 3)
    with pytest.raises(httpx.HTTPStatusError):
        ex.extract(REQ)
    assert len(seen) == 3


def test_missing_key_fails_loudly_at_startup(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        GeminiExtractor()


def test_make_extractor_selects_provider(monkeypatch):
    monkeypatch.setenv("NDR_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    assert isinstance(make_extractor(), GeminiExtractor)
    monkeypatch.setenv("NDR_PROVIDER", "nope")
    with pytest.raises(RuntimeError):
        make_extractor()

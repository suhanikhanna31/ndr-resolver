from __future__ import annotations

import json
import os
import time
from typing import Protocol, Sequence

from pydantic import ValidationError

from .prompts import SYSTEM_PROMPT, build_user_message
from .schemas import Extraction, Intent, NDRRequest, ResolvedExample

TOOL_NAME = "record_extraction"


class Extractor(Protocol):
    def extract(self, req: NDRRequest) -> Extraction: ...


class AnthropicExtractor:
    """Forces a tool call so the model returns schema-shaped JSON, then validates with Pydantic."""

    def __init__(self, model: str | None = None, client=None):
        import anthropic

        self.model = model or os.getenv("NDR_MODEL", "claude-haiku-4-5-20251001")
        # SDK handles transient retries (429/5xx). Hard timeout keeps the voice loop responsive.
        self.client = client or anthropic.Anthropic(max_retries=2, timeout=10.0)
        self._tool = {
            "name": TOOL_NAME,
            "description": "Record the structured interpretation of the customer's reply.",
            "input_schema": Extraction.model_json_schema(),
        }

    def extract(self, req: NDRRequest, examples: Sequence[ResolvedExample] = ()) -> Extraction:
        # Current SDKs no longer accept sampling params like temperature, so reliability comes from the
        # forced tool call + schema validation, not from the sampler. One bounded retry covers rare
        # malformed output; after that the exception propagates and the resolver fails safe.
        last: Exception | None = None
        for _ in range(2):
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=400,
                system=SYSTEM_PROMPT,
                tools=[self._tool],
                tool_choice={"type": "tool", "name": TOOL_NAME},
                messages=[{"role": "user", "content": build_user_message(req, examples)}],
            )
            block = next((b for b in resp.content if b.type == "tool_use"), None)
            try:
                if block is None:
                    raise ValueError("model returned no tool_use block")
                return Extraction.model_validate(block.input)
            except (ValidationError, ValueError) as e:
                last = e
        raise last  # type: ignore[misc]


GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

# Hand-written (not derived from Pydantic) because Gemini accepts only an OpenAPI-style subset of JSON Schema.
# It steers the model; Extraction.model_validate remains the actual guarantee.
GEMINI_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "intent": {"type": "STRING", "enum": [i.value for i in Intent]},
        "confidence": {"type": "NUMBER"},
        "date_expression": {"type": "STRING", "nullable": True},
        "time_slot": {"type": "STRING", "nullable": True, "description": "morning, afternoon, evening or null"},
        "new_address": {"type": "STRING", "nullable": True},
        "new_phone": {"type": "STRING", "nullable": True},
        "reasoning": {"type": "STRING"},
    },
    "required": ["intent", "confidence"],
}


class GeminiExtractor:
    """Same contract as AnthropicExtractor, on the Gemini REST API (Google AI Studio key, free tier works).

    Structured output via responseSchema, then the same Pydantic validation, one bounded retry on malformed
    output, and a small backoff on 429/5xx. Any final failure raises, and the resolver fails safe to a human.
    """

    def __init__(self, model: str | None = None, api_key: str | None = None, client=None,
                 sleep=time.sleep, max_http_retries: int | None = None):
        import httpx

        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
        self._key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if client is None and not self._key:
            # Fail loudly at startup. Silently escalating every request is the worst failure mode.
            raise RuntimeError("GEMINI_API_KEY is not set (get one at https://aistudio.google.com)")
        self.client = client or httpx.Client(timeout=15.0)
        self._sleep = sleep
        self._http_retries = int(os.getenv("GEMINI_MAX_RETRIES", "2")) if max_http_retries is None else max_http_retries

    def _post(self, body: dict) -> dict:
        headers = {"x-goog-api-key": self._key or "", "content-type": "application/json"}  # key in header, not URL
        for attempt in range(self._http_retries + 1):
            resp = self.client.post(GEMINI_URL.format(model=self.model), json=body, headers=headers)
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < self._http_retries:
                self._sleep(2 ** (attempt + 1))  # free tier is rate limited: back off 2s, 4s
                continue
            resp.raise_for_status()
            return resp.json()
        raise RuntimeError("unreachable")

    def extract(self, req: NDRRequest, examples: Sequence[ResolvedExample] = ()) -> Extraction:
        body = {
            "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": [{"text": build_user_message(req, examples)}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": GEMINI_RESPONSE_SCHEMA,
                "maxOutputTokens": 1024,  # headroom: some Gemini models count internal thinking tokens here
            },
        }
        last: Exception | None = None
        for _ in range(2):
            data = self._post(body)
            try:
                cands = data.get("candidates") or []
                if not cands:
                    raise ValueError(f"no candidates (blocked? {data.get('promptFeedback')})")
                parts = (cands[0].get("content") or {}).get("parts") or []
                text = "".join(p.get("text", "") for p in parts).strip()
                if not text:
                    raise ValueError(f"empty response (finishReason={cands[0].get('finishReason')})")
                return Extraction.model_validate(json.loads(text))
            except (ValidationError, ValueError) as e:  # json.JSONDecodeError is a ValueError
                last = e
        raise last  # type: ignore[misc]


def make_extractor():
    """Choose the model provider from NDR_PROVIDER: anthropic (default) or gemini."""
    provider = os.getenv("NDR_PROVIDER", "anthropic").lower()
    if provider == "gemini":
        return GeminiExtractor()
    if provider == "anthropic":
        return AnthropicExtractor()
    raise RuntimeError(f"unknown NDR_PROVIDER={provider!r} (use 'anthropic' or 'gemini')")

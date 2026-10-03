from __future__ import annotations

import os
from typing import Protocol, Sequence

from pydantic import ValidationError

from .prompts import SYSTEM_PROMPT, build_user_message
from .schemas import Extraction, NDRRequest, ResolvedExample

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

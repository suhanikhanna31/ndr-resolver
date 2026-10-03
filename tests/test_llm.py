import inspect
from types import SimpleNamespace

import anthropic
import pytest

from app.llm import AnthropicExtractor
from app.schemas import Intent, NDRRequest

REQ = NDRRequest(awb="L1", ndr_reason="customer_unavailable", customer_utterance="kal bhej do",
                 received_at="2026-10-03T11:00:00+05:30")
GOOD = {"intent": "reschedule", "confidence": 0.9, "date_expression": "tomorrow"}


class FakeClient:
    """Records create() kwargs and replays scripted tool inputs (None = no tool_use block)."""

    def __init__(self, *outputs):
        self.outputs, self.calls = list(outputs), []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kw):
        self.calls.append(kw)
        out = self.outputs.pop(0)
        block = SimpleNamespace(type="tool_use", input=out) if out is not None else SimpleNamespace(type="text")
        return SimpleNamespace(content=[block])


def test_call_kwargs_exist_in_installed_sdk():
    """Regression: the SDK rejects unknown kwargs (e.g. temperature) with a TypeError at call time."""
    client = FakeClient(GOOD)
    AnthropicExtractor(client=client).extract(REQ)
    accepted = set(inspect.signature(anthropic.Anthropic(api_key="x").messages.create).parameters)
    assert set(client.calls[0]) <= accepted, set(client.calls[0]) - accepted


def test_retries_once_on_malformed_output_then_succeeds():
    client = FakeClient({"intent": "reschedule", "confidence": 0.9, "date_expression": "2026-10-04"}, GOOD)
    assert AnthropicExtractor(client=client).extract(REQ).intent is Intent.RESCHEDULE
    assert len(client.calls) == 2


def test_gives_up_after_two_bad_outputs():
    client = FakeClient(None, {"intent": "nonsense", "confidence": 2})
    with pytest.raises(Exception):
        AnthropicExtractor(client=client).extract(REQ)
    assert len(client.calls) == 2

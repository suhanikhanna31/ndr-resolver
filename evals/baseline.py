"""Regex baseline. It exists to prove the LLM earns its cost, and to let CI exercise the eval harness for free."""
import re

from app.schemas import Extraction, Intent, NDRRequest

_DAYS = "monday tuesday wednesday thursday friday saturday sunday".split()


class BaselineExtractor:
    def extract(self, req: NDRRequest) -> Extraction:
        t = req.customer_utterance.lower()
        slot = "morning" if "morning" in t or "subah" in t else "evening" if re.search(r"evening|shaam", t) else None

        def out(intent, conf=0.9, **kw):
            return Extraction(intent=intent, confidence=conf, time_slot=slot, **kw)

        if re.search(r"stop calling|court|complain|report you", t):
            return out(Intent.OPT_OUT_OR_ABUSE)
        if re.search(r"already received|mil gaya", t):
            return out(Intent.ALREADY_RECEIVED)
        if re.search(r"don't want|do not want|cancel|nahi chahiye", t):
            return out(Intent.REFUSE_DELIVERY)
        m = re.search(r"\b[6-9]\d{9}\b", t)
        if m and re.search(r"number|phone|call", t):
            return out(Intent.PHONE_CORRECTION, new_phone=m.group())
        if "address" in t:
            addr = req.customer_utterance.split(":", 1)[-1].strip()
            return out(Intent.ADDRESS_CORRECTION, new_address=addr)
        for pat, expr in [(r"day after tomorrow|parso", "day_after_tomorrow"), (r"tomorrow|\bkal\b", "tomorrow"),
                          (r"today|\baaj\b", "today")]:
            if re.search(pat, t):
                return out(Intent.RESCHEDULE, date_expression=expr)
        for d in _DAYS:
            if d in t:
                return out(Intent.RESCHEDULE, date_expression=f"weekday:{d}")
        m = re.search(r"in (\d+) days", t)
        if m:
            return out(Intent.RESCHEDULE, date_expression=f"in_days:{m.group(1)}")
        return out(Intent.UNCLEAR, 0.3)

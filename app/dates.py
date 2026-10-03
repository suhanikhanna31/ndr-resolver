from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def resolve_date(expr: Optional[str], today: date) -> Optional[date]:
    """Deterministically turn a grammar token into a future-or-today date. None if impossible."""
    if not expr:
        return None
    if expr == "today":
        return today
    if expr == "tomorrow":
        return today + timedelta(days=1)
    if expr == "day_after_tomorrow":
        return today + timedelta(days=2)
    kind, _, arg = expr.partition(":")
    if kind == "in_days":
        return today + timedelta(days=int(arg))
    if kind == "weekday":
        # Strictly after today: "Saturday" said on a Saturday means next Saturday.
        delta = (_WEEKDAYS.index(arg) - today.weekday()) % 7 or 7
        return today + timedelta(days=delta)
    if kind == "day_of_month":
        day = int(arg)
        for offset in range(13):  # skip months where the day does not exist (e.g. 31 Nov)
            m = today.month - 1 + offset
            y, mo = today.year + m // 12, m % 12 + 1
            try:
                cand = date(y, mo, day)
            except ValueError:
                continue
            if cand >= today:
                return cand
    return None

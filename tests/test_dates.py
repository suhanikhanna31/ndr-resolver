from datetime import date

from app.dates import resolve_date

SAT = date(2026, 10, 3)  # a Saturday


def test_relative_tokens():
    assert resolve_date("today", SAT) == SAT
    assert resolve_date("tomorrow", SAT) == date(2026, 10, 4)
    assert resolve_date("day_after_tomorrow", SAT) == date(2026, 10, 5)
    assert resolve_date("in_days:2", SAT) == date(2026, 10, 5)


def test_weekday_is_strictly_future():
    assert resolve_date("weekday:saturday", SAT) == date(2026, 10, 10)
    assert resolve_date("weekday:monday", SAT) == date(2026, 10, 5)


def test_day_of_month_rolls_forward_and_skips_invalid():
    assert resolve_date("day_of_month:15", SAT) == date(2026, 10, 15)
    assert resolve_date("day_of_month:2", SAT) == date(2026, 11, 2)
    assert resolve_date("day_of_month:31", date(2026, 11, 5)) == date(2026, 12, 31)


def test_none():
    assert resolve_date(None, SAT) is None

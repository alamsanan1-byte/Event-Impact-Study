from datetime import date

from eventstudy.build_events import align_day0


TRADING_DAYS = [date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7)]


def test_before_close_filing_uses_same_trading_day():
    assert align_day0("2026-01-05T15:30:00-05:00", TRADING_DAYS) == date(2026, 1, 5)


def test_after_close_filing_uses_next_trading_day():
    assert align_day0("2026-01-05T16:30:00-05:00", TRADING_DAYS) == date(2026, 1, 6)

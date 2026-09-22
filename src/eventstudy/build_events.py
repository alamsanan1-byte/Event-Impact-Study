"""Align SEC filing timestamps to trading-day event dates and deduplicate events."""

from __future__ import annotations

import hashlib
from datetime import date, time
from zoneinfo import ZoneInfo

import pandas as pd

from .common import connect_db, load_config

EASTERN = ZoneInfo("America/New_York")


def align_day0(acceptance_datetime: object, trading_days: list[date]) -> date | None:
    """Map an acceptance timestamp to day 0 using the 16:00 US/Eastern rule."""
    ts = pd.Timestamp(acceptance_datetime)
    if ts.tzinfo is None:
        ts = ts.tz_localize(EASTERN)
    else:
        ts = ts.tz_convert(EASTERN)

    candidate = ts.date()
    if ts.time() > time(16, 0):
        candidate = (ts + pd.Timedelta(days=1)).date()

    for d in trading_days:
        if d >= candidate:
            return d
    return None


def _event_id(ticker: str, accession: str) -> str:
    return hashlib.sha1(f"{ticker}|{accession}".encode("utf-8")).hexdigest()[:16]


def main() -> None:
    cfg = load_config()
    con = connect_db()
    try:
        filings = con.execute(
            """
            SELECT ticker, accession_number, acceptance_datetime, source_url
            FROM raw_edgar_filings
            ORDER BY ticker, acceptance_datetime
            """
        ).fetchdf()
        market_days = con.execute(
            "SELECT date FROM prices WHERE ticker = ? ORDER BY date",
            [cfg["benchmark"]],
        ).fetchdf()["date"].tolist()
        if not market_days:
            raise RuntimeError("Benchmark prices are missing. Run make data in order.")

        built: list[dict] = []
        dropped: list[dict] = []
        for row in filings.itertuples(index=False):
            event_id = _event_id(row.ticker, row.accession_number)
            day0 = align_day0(row.acceptance_datetime, market_days)
            if day0 is None:
                dropped.append(
                    dict(event_id=event_id, ticker=row.ticker, accession_number=row.accession_number,
                         reason="no_trading_day_after_filing", detail=None)
                )
                continue
            built.append(
                dict(
                    event_id=event_id,
                    ticker=row.ticker,
                    accession_number=row.accession_number,
                    announce_ts=row.acceptance_datetime,
                    day0=day0,
                    source="SEC EDGAR 8-K Item 2.02",
                    source_url=row.source_url,
                )
            )

        events = pd.DataFrame(built)
        if events.empty:
            raise RuntimeError("No events could be aligned to trading days.")

        dup_mask = events.duplicated(["ticker", "day0"], keep="first")
        for row in events.loc[dup_mask].itertuples(index=False):
            dropped.append(
                dict(event_id=row.event_id, ticker=row.ticker, accession_number=row.accession_number,
                     reason="duplicate_ticker_day0", detail=f"day0={row.day0}")
            )
        events = events.loc[~dup_mask].copy()

        con.execute("DELETE FROM events")
        con.execute("DELETE FROM dropped_events")
        con.register("events_df", events)
        con.execute(
            """
            INSERT INTO events
            SELECT event_id, ticker, accession_number, announce_ts, day0, source, source_url
            FROM events_df
            """
        )
        con.unregister("events_df")

        if dropped:
            dropped_df = pd.DataFrame(dropped)
            con.register("dropped_df", dropped_df)
            con.execute("INSERT INTO dropped_events SELECT event_id, ticker, accession_number, reason, detail FROM dropped_df")
            con.unregister("dropped_df")

        bad_day0 = con.execute(
            """
            SELECT e.event_id
            FROM events e
            LEFT JOIN prices p ON p.ticker = e.ticker AND p.date = e.day0
            WHERE p.date IS NULL
            """
        ).fetchall()
        if bad_day0:
            raise ValueError(f"Day 0 is not a trading day for {len(bad_day0)} events.")
    finally:
        con.close()

    print(f"Built {len(events):,} unique earnings events; dropped {len(dropped):,}.")


if __name__ == "__main__":
    main()

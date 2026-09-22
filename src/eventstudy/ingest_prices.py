"""Download adjusted daily prices with yfinance, cache them, and load DuckDB."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import yfinance as yf

from .common import connect_db, ensure_dirs, load_config, ROOT


def download_one(ticker: str, start: str, end_exclusive: str, cache_dir: Path) -> pd.DataFrame:
    cache_path = cache_dir / f"{ticker.replace('^', 'index_')}.parquet"
    if cache_path.exists():
        frame = pd.read_parquet(cache_path)
    else:
        raw = yf.download(
            ticker,
            start=start,
            end=end_exclusive,
            auto_adjust=True,
            progress=False,
            actions=False,
            threads=False,
        )
        if raw.empty:
            raise RuntimeError(f"No price data returned for {ticker}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        frame = raw.reset_index()[["Date", "Close"]].rename(
            columns={"Date": "date", "Close": "adj_close"}
        )
        frame["date"] = pd.to_datetime(frame["date"]).dt.date
        frame["adj_close"] = pd.to_numeric(frame["adj_close"], errors="coerce")
        frame = frame.dropna().sort_values("date")
        frame.to_parquet(cache_path, index=False)
        time.sleep(0.15)

    frame = frame.copy()
    frame["ticker"] = ticker
    frame["ret"] = frame["adj_close"].pct_change(fill_method=None)
    return frame[["ticker", "date", "adj_close", "ret"]]


def validate_prices(frame: pd.DataFrame, vix: str) -> None:
    dupes = frame.duplicated(["ticker", "date"], keep=False)
    if dupes.any():
        examples = frame.loc[dupes, ["ticker", "date"]].head().to_dict("records")
        raise ValueError(f"Duplicate (ticker, date) prices found: {examples}")

    bounded = frame[(frame["ticker"] != vix) & frame["ret"].notna()]
    bad = bounded[~bounded["ret"].between(-0.50, 0.50)]
    if not bad.empty:
        examples = bad[["ticker", "date", "ret"]].head().to_dict("records")
        raise ValueError(f"Returns outside [-50%, +50%]: {examples}")


def main() -> None:
    cfg = load_config()
    ensure_dirs()
    cache_dir = ROOT / cfg["paths"]["raw"] / "prices"
    cache_dir.mkdir(parents=True, exist_ok=True)

    end_exclusive = (pd.Timestamp(cfg["period"]["end"]) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    tickers = [c["ticker"] for c in cfg["companies"]] + [cfg["benchmark"], cfg["vix"]]
    frames = [download_one(t, cfg["period"]["start"], end_exclusive, cache_dir) for t in tickers]
    prices = pd.concat(frames, ignore_index=True)
    validate_prices(prices, cfg["vix"])

    curated = ROOT / cfg["paths"]["curated"] / "prices.parquet"
    prices.to_parquet(curated, index=False)

    con = connect_db()
    try:
        companies = pd.DataFrame(cfg["companies"])
        con.register("companies_df", companies)
        con.execute("DELETE FROM companies")
        con.execute("INSERT INTO companies SELECT ticker, cik, name FROM companies_df")
        con.unregister("companies_df")

        con.register("prices_df", prices)
        con.execute("DELETE FROM prices")
        con.execute("INSERT INTO prices SELECT ticker, date, adj_close, ret FROM prices_df")
        con.unregister("prices_df")
    finally:
        con.close()

    print(f"Loaded {len(prices):,} daily price rows for {len(tickers)} series.")


if __name__ == "__main__":
    main()

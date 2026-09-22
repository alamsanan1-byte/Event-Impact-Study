"""Fetch SEC submissions metadata and retain earnings-related 8-K Item 2.02 filings."""

from __future__ import annotations

import os
import re
import time
from typing import Any, Iterable

import pandas as pd
import requests
from dotenv import load_dotenv

from .common import ROOT, connect_db, ensure_dirs, load_config

SEC_BASE = "https://data.sec.gov/submissions"
ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"


def _records_from_columns(obj: dict[str, Any]) -> Iterable[dict[str, Any]]:
    keys = [
        "accessionNumber",
        "filingDate",
        "acceptanceDateTime",
        "form",
        "items",
        "primaryDocument",
    ]
    columns = {k: obj.get(k) if isinstance(obj.get(k), list) else [] for k in keys}
    n = max((len(v) for v in columns.values()), default=0)
    for i in range(n):
        yield {k: (values[i] if i < len(values) else None) for k, values in columns.items()}


def _is_item_202(items: Any) -> bool:
    if items is None:
        return False
    return bool(re.search(r"(?:^|[,;\s])2\.02(?:$|[,;\s])", str(items)))


def parse_sec_acceptance(value: str) -> pd.Timestamp:
    """Return an SEC acceptance timestamp as UTC, treating naive SEC times as US/Eastern."""
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("America/New_York")
    return ts.tz_convert("UTC")


def _filing_url(cik: str, accession: str, primary_document: str | None) -> str:
    cik_int = str(int(cik))
    accession_compact = accession.replace("-", "")
    document = primary_document or f"{accession}-index.html"
    return f"{ARCHIVES_BASE}/{cik_int}/{accession_compact}/{document}"


def _request_json(session: requests.Session, url: str, min_interval: float) -> dict[str, Any]:
    response = session.get(url, timeout=30)
    response.raise_for_status()
    time.sleep(min_interval)
    return response.json()


def fetch_company(session: requests.Session, ticker: str, cik: str, min_interval: float) -> list[dict[str, Any]]:
    main_url = f"{SEC_BASE}/CIK{cik.zfill(10)}.json"
    payload = _request_json(session, main_url, min_interval)

    batches: list[dict[str, Any]] = [payload.get("filings", {}).get("recent", {})]
    for older in payload.get("filings", {}).get("files", []):
        name = older.get("name")
        if name:
            batches.append(_request_json(session, f"{SEC_BASE}/{name}", min_interval))

    rows: list[dict[str, Any]] = []
    for batch in batches:
        for rec in _records_from_columns(batch):
            if rec.get("form") != "8-K" or not _is_item_202(rec.get("items")):
                continue
            accession = rec.get("accessionNumber")
            acceptance = rec.get("acceptanceDateTime")
            if not accession or not acceptance:
                continue
            rows.append(
                {
                    "ticker": ticker,
                    "cik": cik.zfill(10),
                    "accession_number": accession,
                    "form": "8-K",
                    "filing_date": rec.get("filingDate"),
                    "acceptance_datetime": parse_sec_acceptance(acceptance),
                    "items": rec.get("items"),
                    "primary_document": rec.get("primaryDocument"),
                    "source_url": _filing_url(cik, accession, rec.get("primaryDocument")),
                }
            )
    return rows


def main() -> None:
    cfg = load_config()
    ensure_dirs()
    load_dotenv(ROOT / ".env")
    name = os.getenv("SEC_NAME", "").strip()
    email = os.getenv("SEC_EMAIL", "").strip()
    if not name or not email or "@" not in email:
        raise RuntimeError("Set SEC_NAME and SEC_EMAIL in .env before calling the SEC API.")

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": f"{name} {email}",
            "Accept-Encoding": "gzip, deflate",
            "Host": "data.sec.gov",
        }
    )
    rps = min(float(cfg["sec"]["max_requests_per_second"]), 9.0)
    min_interval = 1.0 / rps

    rows: list[dict[str, Any]] = []
    for company in cfg["companies"]:
        rows.extend(fetch_company(session, company["ticker"], company["cik"], min_interval))

    filings = pd.DataFrame(rows)
    if filings.empty:
        raise RuntimeError("No Item 2.02 8-K filings were retrieved.")
    filings = filings.drop_duplicates(["ticker", "accession_number"]).sort_values(
        ["ticker", "acceptance_datetime"]
    )
    start = pd.Timestamp(cfg["period"]["start"], tz="UTC")
    end = pd.Timestamp(cfg["period"]["end"], tz="UTC") + pd.Timedelta(days=1)
    filings = filings[filings["acceptance_datetime"].between(start, end, inclusive="left")]

    cache = ROOT / cfg["paths"]["raw"] / "edgar_item_202.parquet"
    filings.to_parquet(cache, index=False)

    con = connect_db()
    try:
        con.register("filings_df", filings)
        con.execute("DELETE FROM raw_edgar_filings")
        con.execute(
            """
            INSERT INTO raw_edgar_filings
            SELECT ticker, cik, accession_number, form, filing_date,
                   acceptance_datetime, items, primary_document, source_url
            FROM filings_df
            """
        )
        con.unregister("filings_df")
    finally:
        con.close()

    print(f"Loaded {len(filings):,} SEC Item 2.02 8-K filings.")


if __name__ == "__main__":
    main()

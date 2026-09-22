"""Estimate event-specific market models and compute abnormal returns and CARs."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .common import ROOT, connect_db, load_config


def fit_market_model(est: pd.DataFrame) -> dict[str, float]:
    x = sm.add_constant(est["market_ret"].to_numpy(dtype=float))
    y = est["stock_ret"].to_numpy(dtype=float)
    fit = sm.OLS(y, x).fit()
    residuals = fit.resid
    return {
        "alpha": float(fit.params[0]),
        "beta": float(fit.params[1]),
        "sigma_resid": float(np.std(residuals, ddof=2)),
        "n_est_days": int(len(est)),
        "market_mean": float(est["market_ret"].mean()),
        "market_ssx": float(((est["market_ret"] - est["market_ret"].mean()) ** 2).sum()),
    }


def main() -> None:
    cfg = load_config()
    est_start, est_end = cfg["estimation_window"]
    path_start, path_end = cfg["event_path"]
    min_days = int(cfg["min_estimation_days"])

    con = connect_db()
    try:
        events = con.execute("SELECT event_id, ticker, day0, accession_number FROM events ORDER BY day0").fetchdf()
        prices = con.execute("SELECT ticker, date, ret FROM prices WHERE ret IS NOT NULL ORDER BY ticker, date").fetchdf()
        market = prices[prices["ticker"] == cfg["benchmark"]][["date", "ret"]].rename(columns={"ret": "market_ret"})
        market = market.sort_values("date").reset_index(drop=True)
        market_index = {d: i for i, d in enumerate(market["date"])}

        model_rows: list[dict] = []
        ar_rows: list[dict] = []
        new_drops: list[dict] = []

        for event in events.itertuples(index=False):
            if event.day0 not in market_index:
                new_drops.append(dict(event_id=event.event_id, ticker=event.ticker, accession_number=event.accession_number,
                                      reason="day0_missing_from_benchmark", detail=None))
                continue
            d0_idx = market_index[event.day0]
            lo, hi = d0_idx + est_start, d0_idx + est_end
            plo, phi = d0_idx + path_start, d0_idx + path_end
            if lo < 0 or plo < 0 or phi >= len(market):
                new_drops.append(dict(event_id=event.event_id, ticker=event.ticker, accession_number=event.accession_number,
                                      reason="insufficient_calendar_history_or_forward_window", detail=None))
                continue

            stock = prices[prices["ticker"] == event.ticker][["date", "ret"]].rename(columns={"ret": "stock_ret"})
            merged = market.merge(stock, on="date", how="left")
            est = merged.iloc[lo:hi + 1].dropna()
            if len(est) < min_days:
                new_drops.append(dict(event_id=event.event_id, ticker=event.ticker, accession_number=event.accession_number,
                                      reason="insufficient_estimation_days", detail=f"n={len(est)}"))
                continue

            event_slice = merged.iloc[plo:phi + 1].copy()
            if event_slice["stock_ret"].isna().any() or len(event_slice) != path_end - path_start + 1:
                new_drops.append(dict(event_id=event.event_id, ticker=event.ticker, accession_number=event.accession_number,
                                      reason="missing_event_window_price", detail=None))
                continue

            params = fit_market_model(est)
            if not np.isfinite(params["sigma_resid"]) or params["sigma_resid"] <= 0 or params["market_ssx"] <= 0:
                new_drops.append(dict(event_id=event.event_id, ticker=event.ticker, accession_number=event.accession_number,
                                      reason="invalid_market_model", detail=None))
                continue

            model_rows.append({"event_id": event.event_id, **params})
            for rel_day, row in zip(range(path_start, path_end + 1), event_slice.itertuples(index=False)):
                expected = params["alpha"] + params["beta"] * row.market_ret
                ar = row.stock_ret - expected
                pred_var = 1.0 + 1.0 / params["n_est_days"] + (
                    (row.market_ret - params["market_mean"]) ** 2 / params["market_ssx"]
                )
                sar = ar / (params["sigma_resid"] * math.sqrt(pred_var))
                ar_rows.append(
                    dict(event_id=event.event_id, rel_day=rel_day, date=row.date,
                         actual_ret=row.stock_ret, market_ret=row.market_ret,
                         expected_ret=expected, ar=ar, sar=sar)
                )

        con.execute("DELETE FROM market_model")
        con.execute("DELETE FROM abnormal_returns")
        con.execute("DELETE FROM car")
        con.execute("DELETE FROM windows")

        model_df = pd.DataFrame(model_rows)
        ar_df = pd.DataFrame(ar_rows)
        if model_df.empty or ar_df.empty:
            raise RuntimeError("No events survived market-model estimation.")
        con.register("model_df", model_df)
        con.execute("INSERT INTO market_model SELECT * FROM model_df")
        con.unregister("model_df")
        con.register("ar_df", ar_df)
        con.execute("INSERT INTO abnormal_returns SELECT * FROM ar_df")
        con.unregister("ar_df")

        window_rows = [(name, int(bounds[0]), int(bounds[1])) for name, bounds in cfg["windows"].items()]
        con.executemany("INSERT INTO windows VALUES (?, ?, ?)", window_rows)
        con.execute((ROOT / "sql" / "car_windows.sql").read_text(encoding="utf-8"))

        if new_drops:
            drops_df = pd.DataFrame(new_drops)
            con.register("drops_df", drops_df)
            con.execute("INSERT INTO dropped_events SELECT event_id, ticker, accession_number, reason, detail FROM drops_df")
            con.unregister("drops_df")
            ids = [r["event_id"] for r in new_drops]
            con.executemany("DELETE FROM events WHERE event_id = ?", [(x,) for x in ids])

        print(f"Estimated {len(model_df):,} market models and {len(ar_df):,} abnormal-return rows.")
    finally:
        con.close()


if __name__ == "__main__":
    main()

"""Build leakage-safe predictive features known by the close of event day 0."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .common import connect_db, load_config

# Inclusive relative-day ranges. Every predictive feature ends at day 0 or earlier.
FEATURE_RELATIVE_RANGES = {
    "day0_ar": (0, 0),
    "car_pre": (-5, -1),
    "momentum_20": (-25, -6),
    "residual_vol": (-250, -30),
    "beta": (-250, -30),
    "vix_close": (0, 0),
    "sp500_return_20": (-20, -1),
}


def compounded_return(values: pd.Series) -> float:
    return float(np.prod(1.0 + values.to_numpy(dtype=float)) - 1.0)


def assert_no_feature_leakage() -> None:
    leaking = {name: bounds for name, bounds in FEATURE_RELATIVE_RANGES.items() if bounds[1] > 0}
    if leaking:
        raise ValueError(f"Features use information after day 0: {leaking}")


def main() -> None:
    assert_no_feature_leakage()
    cfg = load_config()
    con = connect_db()
    try:
        events = con.execute("SELECT event_id, ticker, day0 FROM events ORDER BY day0").fetchdf()
        models = con.execute("SELECT event_id, beta, sigma_resid FROM market_model").fetchdf().set_index("event_id")
        ar = con.execute("SELECT event_id, rel_day, ar FROM abnormal_returns").fetchdf()
        cars = con.execute("SELECT event_id, window, car FROM car").fetchdf()
        prices = con.execute("SELECT ticker, date, adj_close, ret FROM prices ORDER BY ticker, date").fetchdf()

        market = prices[prices.ticker == cfg["benchmark"]].sort_values("date").reset_index(drop=True)
        market_idx = {d: i for i, d in enumerate(market.date)}
        vix = prices[prices.ticker == cfg["vix"]].set_index("date")
        stock_groups = {t: g.sort_values("date").set_index("date") for t, g in prices.groupby("ticker")}

        car_lookup = cars.pivot(index="event_id", columns="window", values="car")
        day0_ar = ar[ar.rel_day == 0].set_index("event_id")["ar"]
        rows = []
        for e in events.itertuples(index=False):
            if e.event_id not in models.index or e.event_id not in car_lookup.index or e.event_id not in day0_ar.index:
                continue
            if e.day0 not in market_idx or e.day0 not in vix.index:
                continue
            i = market_idx[e.day0]
            if i < 25:
                continue
            stock = stock_groups[e.ticker]
            stock_dates = market.iloc[i - 25:i + 1]["date"].tolist()
            if any(d not in stock.index for d in stock_dates):
                continue

            mom_dates = market.iloc[i - 25:i - 5]["date"]  # -25..-6 = 20 trading days
            mkt_dates = market.iloc[i - 20:i]["date"]      # -20..-1 = 20 trading days
            momentum = compounded_return(stock.loc[mom_dates, "ret"])
            sp_ret = compounded_return(market.set_index("date").loc[mkt_dates, "ret"])
            target = int(car_lookup.loc[e.event_id, "post_5"] > 0)
            rows.append(
                dict(
                    event_id=e.event_id,
                    ticker=e.ticker,
                    day0=e.day0,
                    target=target,
                    day0_ar=float(day0_ar.loc[e.event_id]),
                    car_pre=float(car_lookup.loc[e.event_id, "pre"]),
                    momentum_20=momentum,
                    residual_vol=float(models.loc[e.event_id, "sigma_resid"]),
                    beta=float(models.loc[e.event_id, "beta"]),
                    vix_close=float(vix.loc[e.day0, "adj_close"]),
                    sp500_return_20=sp_ret,
                )
            )

        features = pd.DataFrame(rows).dropna()
        if features.empty:
            raise RuntimeError("No complete model feature rows were built.")
        con.execute("DELETE FROM event_features")
        con.register("features_df", features)
        con.execute("INSERT INTO event_features SELECT * FROM features_df")
        con.unregister("features_df")
    finally:
        con.close()

    print(f"Built {len(features):,} leakage-safe feature rows.")


if __name__ == "__main__":
    main()

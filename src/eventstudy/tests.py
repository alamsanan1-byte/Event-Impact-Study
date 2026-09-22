"""Event-study inference: cross-sectional t, BMP standardized test, sign test, and H2."""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .common import ROOT, connect_db, ensure_dirs, load_config


def one_sample_stats(values: np.ndarray, standardized: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    standardized = np.asarray(standardized, dtype=float)
    n = len(values)
    if n < 2:
        return dict(n=n, mean_car=float(np.mean(values)) if n else np.nan,
                    ci_low=np.nan, ci_high=np.nan, t_stat=np.nan, t_pvalue=np.nan,
                    bmp_stat=np.nan, bmp_pvalue=np.nan, sign_pvalue=np.nan)

    mean = float(values.mean())
    sem = float(stats.sem(values))
    crit = float(stats.t.ppf(0.975, df=n - 1))
    t_res = stats.ttest_1samp(values, popmean=0.0)

    bmp_sd = float(np.std(standardized, ddof=1))
    bmp = float(np.mean(standardized) / (bmp_sd / math.sqrt(n))) if bmp_sd > 0 else np.nan
    bmp_p = float(2 * stats.t.sf(abs(bmp), df=n - 1)) if np.isfinite(bmp) else np.nan

    positive = int((values > 0).sum())
    nonzero = int((values != 0).sum())
    sign_p = float(stats.binomtest(positive, nonzero, p=0.5).pvalue) if nonzero else np.nan
    return {
        "n": n,
        "mean_car": mean,
        "ci_low": mean - crit * sem,
        "ci_high": mean + crit * sem,
        "t_stat": float(t_res.statistic),
        "t_pvalue": float(t_res.pvalue),
        "bmp_stat": bmp,
        "bmp_pvalue": bmp_p,
        "sign_pvalue": sign_p,
    }


def create_car_figure(ar: pd.DataFrame, output_path) -> None:
    wide = ar.pivot(index="event_id", columns="rel_day", values="ar").sort_index(axis=1)
    wide = wide.loc[:, [c for c in wide.columns if -5 <= c <= 20]].dropna()
    cumulative = wide.cumsum(axis=1)
    n = len(cumulative)
    mean = cumulative.mean(axis=0)
    sem = cumulative.sem(axis=0)
    crit = stats.t.ppf(0.975, df=n - 1) if n > 1 else np.nan

    fig, ax = plt.subplots(figsize=(9, 5))
    x = mean.index.to_numpy(dtype=int)
    y = mean.to_numpy(dtype=float)
    ax.plot(x, y, linewidth=2, label="Average CAR")
    ax.fill_between(x, (mean - crit * sem).to_numpy(), (mean + crit * sem).to_numpy(), alpha=0.2, label="95% CI")
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.axhline(0, linewidth=0.8)
    ax.set(title="Average cumulative abnormal return around earnings 8-K filings",
           xlabel="Trading days relative to day 0", ylabel="Average CAR")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    cfg = load_config()
    ensure_dirs()
    con = connect_db()
    try:
        cars = con.execute("SELECT event_id, window, car, scar FROM car ORDER BY window, event_id").fetchdf()
        results = []
        for window, group in cars.groupby("window"):
            results.append({"window": window, **one_sample_stats(group["car"].to_numpy(), group["scar"].to_numpy())})
        result_df = pd.DataFrame(results)
        con.execute("DELETE FROM study_results")
        con.register("study_df", result_df)
        con.execute("INSERT INTO study_results SELECT window, n, mean_car, ci_low, ci_high, t_stat, t_pvalue, bmp_stat, bmp_pvalue, sign_pvalue FROM study_df")
        con.unregister("study_df")

        h2 = con.execute(
            """
            SELECT d0.event_id, d0.ar AS day0_ar, c.car AS post20
            FROM abnormal_returns d0
            JOIN car c ON c.event_id = d0.event_id AND c.window = 'post_20'
            WHERE d0.rel_day = 0 AND d0.ar <> 0
            """
        ).fetchdf()
        pos = h2.loc[h2["day0_ar"] > 0, "post20"].to_numpy()
        neg = h2.loc[h2["day0_ar"] < 0, "post20"].to_numpy()
        welch = stats.ttest_ind(pos, neg, equal_var=False) if len(pos) > 1 and len(neg) > 1 else None
        h2_row = pd.DataFrame([
            dict(
                positive_n=len(pos), negative_n=len(neg),
                positive_mean=float(np.mean(pos)) if len(pos) else np.nan,
                negative_mean=float(np.mean(neg)) if len(neg) else np.nan,
                difference=float(np.mean(pos) - np.mean(neg)) if len(pos) and len(neg) else np.nan,
                welch_t=float(welch.statistic) if welch is not None else np.nan,
                pvalue=float(welch.pvalue) if welch is not None else np.nan,
            )
        ])
        con.execute("DELETE FROM h2_results")
        con.register("h2_df", h2_row)
        con.execute("INSERT INTO h2_results SELECT * FROM h2_df")
        con.unregister("h2_df")

        ar = con.execute("SELECT event_id, rel_day, ar FROM abnormal_returns WHERE rel_day BETWEEN -5 AND 20").fetchdf()
        figure_path = ROOT / cfg["paths"]["outputs"] / "average_car_path.png"
        create_car_figure(ar, figure_path)
        result_df.to_csv(ROOT / cfg["paths"]["outputs"] / "study_results.csv", index=False)
        h2_row.to_csv(ROOT / cfg["paths"]["outputs"] / "h2_results.csv", index=False)
    finally:
        con.close()

    print(result_df.to_string(index=False))
    print("\nH2 comparison:\n", h2_row.to_string(index=False))


if __name__ == "__main__":
    main()

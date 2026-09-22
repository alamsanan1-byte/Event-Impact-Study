"""Fatal data-quality checks plus a concise quality report."""

from __future__ import annotations

from .common import ROOT, connect_db, ensure_dirs, load_config


def main() -> None:
    cfg = load_config()
    ensure_dirs()
    con = connect_db()
    try:
        duplicate_prices = con.execute(
            "SELECT COUNT(*) FROM (SELECT ticker, date, COUNT(*) n FROM prices GROUP BY 1,2 HAVING n > 1)"
        ).fetchone()[0]
        duplicate_events = con.execute(
            "SELECT COUNT(*) FROM (SELECT ticker, day0, COUNT(*) n FROM events GROUP BY 1,2 HAVING n > 1)"
        ).fetchone()[0]
        invalid_day0 = con.execute(
            """
            SELECT COUNT(*) FROM events e
            LEFT JOIN prices p ON p.ticker=e.ticker AND p.date=e.day0
            WHERE p.date IS NULL
            """
        ).fetchone()[0]
        outlier_returns = con.execute(
            "SELECT COUNT(*) FROM prices WHERE ticker <> ? AND ret IS NOT NULL AND (ret < -0.50 OR ret > 0.50)",
            [cfg["vix"]],
        ).fetchone()[0]

        failures = {
            "duplicate_prices": duplicate_prices,
            "duplicate_events": duplicate_events,
            "day0_not_trading_day": invalid_day0,
            "returns_outside_50pct": outlier_returns,
        }
        bad = {k: v for k, v in failures.items() if v}
        if bad:
            raise ValueError(f"Fatal quality checks failed: {bad}")

        missing = con.execute(
            """
            WITH calendar AS (
                SELECT date FROM prices WHERE ticker = ?
            ), stocks AS (
                SELECT ticker FROM companies
            )
            SELECT s.ticker, COUNT(*) FILTER (WHERE p.date IS NULL) AS missing_trading_days
            FROM stocks s
            CROSS JOIN calendar c
            LEFT JOIN prices p ON p.ticker=s.ticker AND p.date=c.date
            GROUP BY s.ticker
            ORDER BY missing_trading_days DESC, s.ticker
            """,
            [cfg["benchmark"]],
        ).fetchdf()
        drops = con.execute(
            "SELECT reason, COUNT(*) AS events_dropped FROM dropped_events GROUP BY reason ORDER BY events_dropped DESC"
        ).fetchdf()

        report_path = ROOT / cfg["paths"]["outputs"] / "quality_report.md"
        lines = [
            "# Data quality report",
            "",
            "All fatal pipeline checks passed: no duplicate `(ticker, date)` prices, no duplicate `(ticker, day0)` events, every day 0 is a trading day, and all non-VIX daily returns are within -50% to +50%.",
            "",
            "## Missing benchmark trading days per ticker",
            "",
            missing.to_markdown(index=False),
            "",
            "## Events dropped",
            "",
            drops.to_markdown(index=False) if not drops.empty else "No events were dropped.",
            "",
        ]
        report_path.write_text("\n".join(lines), encoding="utf-8")
    finally:
        con.close()

    print(f"Quality checks passed. Report: {report_path}")


if __name__ == "__main__":
    main()

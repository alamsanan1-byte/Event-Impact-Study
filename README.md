# Earnings Announcement Event Study

This personal Data & AI portfolio project studies whether 30 current US large-cap stocks earn abnormal returns around SEC-reported earnings announcements, whether the initial reaction is followed by post-earnings drift, and whether the sign of the next five trading days' abnormal return can be predicted out of sample. Prices come from Yahoo Finance, event timestamps come from SEC EDGAR Item 2.02 8-K filings, inference is based on a market-model event study, and prediction uses a fixed chronological train/test split. Null results are valid findings and are reported rather than hidden.

## Hypotheses — fixed before results

- **H1:** The mean abnormal return in window **[0,+1]** around earnings differs from zero.
- **H2:** **Post-earnings drift:** stocks with a positive day-0 abnormal return have a higher **CAR[+1,+20]** than stocks with a negative day-0 reaction.
- **H3:** The sign of **CAR[+1,+5]** can be predicted from information known at the close of day 0 better than an **"always up"** baseline.

These hypotheses are written down before looking at the final results. Null results are acceptable and will be reported honestly.

## Headline results

**Status: pipeline implemented; empirical results not yet populated in this repository.** I have intentionally not invented numbers. After running `make all` with the required SEC contact details, the generated files in `outputs/` provide the values to report here:

| Question | Result to report |
|---|---|
| H1: mean CAR[0,+1] | `outputs/study_results.csv` → `announcement` row: mean, 95% CI, N, t/BMP/sign p-values |
| H2: positive vs negative day-0 reaction | `outputs/h2_results.csv` → group means, difference, Welch t-test p-value |
| H3: out-of-sample prediction | `outputs/model_metrics.csv` + `outputs/model_diagnostics.csv` → test metrics, accuracy difference CI, shuffled-label AUC |

The event-study figure is generated at `outputs/average_car_path.png`.

![Average CAR path](outputs/average_car_path.png)

## Method

### 1. Data ingest and storage

The sample contains 30 current US large-cap stocks from 2012-01-01 through 2025-12-31. Adjusted daily closes are downloaded with `yfinance` using `auto_adjust=True` and cached as Parquet. The S&P 500 price index (`^GSPC`) is the benchmark and `^VIX` supplies the day-0 volatility feature. Curated data are loaded into DuckDB using `sql/schema.sql`.

Earnings events come from the SEC submissions API. For each company the code reads the main `CIK##########.json` file and the older submission files listed under `filings.files`, retaining 8-K filings whose `items` field contains Item 2.02. Each event stores its source and the EDGAR filing URL. Requests identify the user using `SEC_NAME` and `SEC_EMAIL` from `.env` and are throttled below 10 requests per second.

**Day 0 rule:** the filing acceptance timestamp is converted to US Eastern time. A filing accepted after 16:00 rolls to the next trading day; a filing accepted at or before 16:00 uses that trading day when it is a trading day.

### 2. Event study

For each event, an OLS market model is estimated over trading days **[-250,-30]**:

`stock_return = alpha + beta × market_return + residual`

At least 120 matched estimation days are required. Abnormal return is:

`AR = actual_return - (alpha + beta × market_return)`

The four CAR windows are **[-5,-1]**, **[0,+1]**, **[+1,+5]**, and **[+1,+20]**. CAR aggregation is performed in SQL (`sql/car_windows.sql`). Each window is reported with mean CAR, a 95% confidence interval, N, a cross-sectional one-sample t-test, a BMP standardized test, and a two-sided sign test.

H2 divides events by the sign of day-0 AR and compares **CAR[+1,+20]** with a Welch unequal-variance t-test. The code also generates an average CAR path from day -5 through +20 with a 95% confidence band.

### 3. Predictive model

The target is 1 when **CAR[+1,+5] > 0** and 0 otherwise. Every feature is observable at the close of day 0 or earlier:

- day-0 abnormal return;
- CAR[-5,-1];
- 20-trading-day stock momentum ending on day -6;
- pre-event market-model residual volatility;
- market-model beta;
- VIX close on day 0;
- compounded S&P 500 return over [-20,-1].

The split is chronological: **2012-2019 train, 2020-2025 test**. There is no random train/test split. Models are an "always up" baseline, a training-majority baseline, and logistic regression with standardized features. Test metrics are accuracy, balanced accuracy, class-specific precision/recall, and ROC-AUC. A paired bootstrap gives a 95% CI for the logistic model's accuracy difference versus "always up". As a leakage sanity check, logistic regression is refit after shuffling the training labels; its test AUC should be around 0.50 subject to sampling variation.

## Data quality

The pipeline stops if it finds duplicate `(ticker, date)` price rows, duplicate `(ticker, day0)` events, a day 0 that is not a trading day for the stock, or a non-VIX daily return outside -50% to +50%. `make quality` writes `outputs/quality_report.md`, including missing benchmark trading days by ticker and counts of dropped events by reason.

Three pytest checks cover the most important implementation invariants:

1. CAR equals the sum of ARs in its window.
2. A before-close filing aligns to the same trading day while an after-close filing aligns to the next trading day.
3. No predictive feature uses information after day 0.

## Data sources

- **SEC EDGAR submissions API:** 8-K Item 2.02 filing metadata and acceptance timestamps.
- **Yahoo Finance via `yfinance`:** adjusted daily prices for the 30 equities, `^GSPC`, and `^VIX`.

## Limitations

- **Survivorship bias:** the universe is selected from today's large-cap stocks rather than historical index membership, so failed/delisted former large caps are absent.
- **Price-index benchmark:** `^GSPC` is a price index, not a total-return benchmark, so dividends are treated differently from the auto-adjusted stock series.
- **Day-0 timing:** 16:00 Eastern is a practical rule. It does not perfectly capture when information became tradable, pre-announcements, conference calls, or intraday reactions.
- **SEC Item 2.02 proxy:** the filing identifies earnings-result disclosure filings, but it is not a vendor-curated earnings calendar and may miss or differently time some announcements.
- **Daily data:** daily closes cannot isolate intraday announcement effects.

## Reproduce

Use Python 3.11+.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your real SEC_NAME and SEC_EMAIL.

make data
make study
make model
make quality
make test
```

Or run the full pipeline with:

```bash
make all
```

Important generated outputs remain reproducible from source and can be committed as portfolio evidence after the run: `data/event_study.duckdb`, cached Parquet files, `outputs/study_results.csv`, `outputs/h2_results.csv`, `outputs/model_metrics.csv`, `outputs/model_diagnostics.csv`, `outputs/average_car_path.png`, and `outputs/quality_report.md`.

## Interview-ready explanation

The central design choice is to separate **explanation** from **prediction**. The event study asks whether returns differ from what a market model would predict, while the classifier asks whether information available by day 0 has any out-of-sample predictive value. The fixed time split prevents future events from influencing model training, and the shuffled-label test is a direct sanity check for leakage. The project is credible even if H1-H3 are null because the hypotheses, data rules, and evaluation protocol are fixed before the results.

## Author

**Sanan Alam** — personal project, not investment advice.

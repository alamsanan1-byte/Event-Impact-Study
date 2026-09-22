# Earnings Announcement Event Study — Specification

## Research question

Do US large-cap stocks earn abnormal returns around earnings announcements, does the effect continue after the announcement, and can the post-earnings direction be predicted better than a naive baseline?

## Pre-specified hypotheses

- H1: Mean abnormal return in [0,+1] differs from zero.
- H2: Events with positive day-0 abnormal return have higher CAR[+1,+20] than events with negative day-0 abnormal return.
- H3: The sign of CAR[+1,+5] can be predicted from information known at the close of day 0 better than an always-up baseline.

Null results are acceptable and must be reported.

## Data

- 30 current US large-cap stocks, 2012-01-01 through 2025-12-31.
- Adjusted daily stock prices from yfinance (`auto_adjust=True`), cached to Parquet.
- S&P 500 price index (`^GSPC`) as benchmark and `^VIX` as a predictive feature.
- Earnings events from SEC EDGAR submissions metadata: 8-K filings with Item 2.02, including older files listed in `filings.files`.
- SEC requests use a descriptive User-Agent from `.env` and remain below 10 requests/second.
- If acceptance is after 16:00 US Eastern, day 0 is the next trading day.
- Every event stores `source` and `source_url`.

## Event study

- OLS market model on [-250,-30], minimum 120 observations.
- AR = actual return - (alpha + beta * market return).
- SQL CAR windows: [-5,-1], [0,+1], [+1,+5], [+1,+20].
- Report cross-sectional t-test, BMP standardized test, sign test, mean CAR, 95% CI and N.
- H2 uses a Welch t-test on CAR[+1,+20] after splitting by the sign of day-0 AR.
- Figure: average CAR path from -5 to +20 with a 95% CI band.

## Prediction

Target: 1 if CAR[+1,+5] > 0, otherwise 0.

Features known by the close of day 0:

- day-0 AR;
- CAR[-5,-1];
- 20-day stock momentum ending at day -6;
- residual volatility from the market model;
- beta;
- VIX close on day 0;
- S&P 500 return over [-20,-1].

Train on 2012-2019; test on 2020-2025. No random split. Compare always-up, training-majority and standardized logistic regression. Report accuracy, balanced accuracy, class-specific precision/recall and ROC-AUC. Bootstrap the accuracy difference between logistic regression and always-up. Refit with shuffled training labels as a leakage diagnostic.

## Quality gates

The pipeline fails on duplicate `(ticker,date)` prices, duplicate `(ticker,day0)` events, non-trading day event dates, or non-VIX returns outside [-50%,+50%]. A Markdown quality report records missing benchmark trading days by ticker and dropped-event reasons.

## Required tests

- CAR equals the sum of ARs.
- before-close and after-close day-0 alignment.
- no predictive feature uses data after day 0.

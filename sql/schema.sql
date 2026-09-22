CREATE TABLE IF NOT EXISTS companies (
    ticker VARCHAR PRIMARY KEY,
    cik VARCHAR NOT NULL,
    name VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS prices (
    ticker VARCHAR NOT NULL,
    date DATE NOT NULL,
    adj_close DOUBLE NOT NULL,
    ret DOUBLE,
    PRIMARY KEY (ticker, date)
);

CREATE TABLE IF NOT EXISTS raw_edgar_filings (
    ticker VARCHAR NOT NULL,
    cik VARCHAR NOT NULL,
    accession_number VARCHAR NOT NULL,
    form VARCHAR NOT NULL,
    filing_date DATE,
    acceptance_datetime TIMESTAMPTZ,
    items VARCHAR,
    primary_document VARCHAR,
    source_url VARCHAR NOT NULL,
    PRIMARY KEY (ticker, accession_number)
);

CREATE TABLE IF NOT EXISTS events (
    event_id VARCHAR PRIMARY KEY,
    ticker VARCHAR NOT NULL,
    accession_number VARCHAR NOT NULL,
    announce_ts TIMESTAMPTZ NOT NULL,
    day0 DATE NOT NULL,
    source VARCHAR NOT NULL,
    source_url VARCHAR NOT NULL,
    UNIQUE (ticker, accession_number),
    UNIQUE (ticker, day0)
);

CREATE TABLE IF NOT EXISTS dropped_events (
    event_id VARCHAR,
    ticker VARCHAR,
    accession_number VARCHAR,
    reason VARCHAR NOT NULL,
    detail VARCHAR
);

CREATE TABLE IF NOT EXISTS market_model (
    event_id VARCHAR PRIMARY KEY,
    alpha DOUBLE NOT NULL,
    beta DOUBLE NOT NULL,
    sigma_resid DOUBLE NOT NULL,
    n_est_days INTEGER NOT NULL,
    market_mean DOUBLE NOT NULL,
    market_ssx DOUBLE NOT NULL
);

CREATE TABLE IF NOT EXISTS abnormal_returns (
    event_id VARCHAR NOT NULL,
    rel_day INTEGER NOT NULL,
    date DATE NOT NULL,
    actual_ret DOUBLE NOT NULL,
    market_ret DOUBLE NOT NULL,
    expected_ret DOUBLE NOT NULL,
    ar DOUBLE NOT NULL,
    sar DOUBLE NOT NULL,
    PRIMARY KEY (event_id, rel_day)
);

CREATE TABLE IF NOT EXISTS windows (
    window VARCHAR PRIMARY KEY,
    start_day INTEGER NOT NULL,
    end_day INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS car (
    event_id VARCHAR NOT NULL,
    window VARCHAR NOT NULL,
    car DOUBLE NOT NULL,
    scar DOUBLE NOT NULL,
    n_days INTEGER NOT NULL,
    PRIMARY KEY (event_id, window)
);

CREATE TABLE IF NOT EXISTS study_results (
    window VARCHAR PRIMARY KEY,
    n INTEGER NOT NULL,
    mean_car DOUBLE NOT NULL,
    ci_low DOUBLE,
    ci_high DOUBLE,
    t_stat DOUBLE,
    t_pvalue DOUBLE,
    bmp_stat DOUBLE,
    bmp_pvalue DOUBLE,
    sign_pvalue DOUBLE
);

CREATE TABLE IF NOT EXISTS h2_results (
    positive_n INTEGER,
    negative_n INTEGER,
    positive_mean DOUBLE,
    negative_mean DOUBLE,
    difference DOUBLE,
    welch_t DOUBLE,
    pvalue DOUBLE
);

CREATE TABLE IF NOT EXISTS event_features (
    event_id VARCHAR PRIMARY KEY,
    ticker VARCHAR NOT NULL,
    day0 DATE NOT NULL,
    target INTEGER NOT NULL,
    day0_ar DOUBLE NOT NULL,
    car_pre DOUBLE NOT NULL,
    momentum_20 DOUBLE NOT NULL,
    residual_vol DOUBLE NOT NULL,
    beta DOUBLE NOT NULL,
    vix_close DOUBLE NOT NULL,
    sp500_return_20 DOUBLE NOT NULL
);

CREATE TABLE IF NOT EXISTS model_predictions (
    event_id VARCHAR NOT NULL,
    split VARCHAR NOT NULL,
    model VARCHAR NOT NULL,
    y_true INTEGER NOT NULL,
    y_pred INTEGER NOT NULL,
    p_up DOUBLE NOT NULL,
    PRIMARY KEY (event_id, model)
);

CREATE TABLE IF NOT EXISTS model_metrics (
    model VARCHAR PRIMARY KEY,
    n_test INTEGER NOT NULL,
    accuracy DOUBLE NOT NULL,
    balanced_accuracy DOUBLE NOT NULL,
    precision_down DOUBLE,
    recall_down DOUBLE,
    precision_up DOUBLE,
    recall_up DOUBLE,
    roc_auc DOUBLE
);

CREATE TABLE IF NOT EXISTS model_diagnostics (
    metric VARCHAR PRIMARY KEY,
    value DOUBLE,
    ci_low DOUBLE,
    ci_high DOUBLE,
    note VARCHAR
);

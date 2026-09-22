"""Time-split baselines and standardized logistic regression for H3."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .common import ROOT, connect_db, ensure_dirs, load_config

FEATURES = [
    "day0_ar",
    "car_pre",
    "momentum_20",
    "residual_vol",
    "beta",
    "vix_close",
    "sp500_return_20",
]


def metrics(y_true: np.ndarray, y_pred: np.ndarray, p_up: np.ndarray) -> dict[str, float]:
    auc = roc_auc_score(y_true, p_up) if len(np.unique(y_true)) == 2 else np.nan
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_down": float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "recall_down": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "precision_up": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_up": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "roc_auc": float(auc),
    }


def bootstrap_accuracy_difference(
    y_true: np.ndarray,
    pred_model: np.ndarray,
    pred_baseline: np.ndarray,
    draws: int,
    seed: int,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    diffs = np.empty(draws)
    n = len(y_true)
    for b in range(draws):
        idx = rng.integers(0, n, n)
        diffs[b] = (
            (pred_model[idx] == y_true[idx]).mean()
            - (pred_baseline[idx] == y_true[idx]).mean()
        )
    observed = float((pred_model == y_true).mean() - (pred_baseline == y_true).mean())
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return observed, float(lo), float(hi)


def main() -> None:
    cfg = load_config()
    ensure_dirs()
    con = connect_db()
    try:
        df = con.execute("SELECT * FROM event_features ORDER BY day0, event_id").fetchdf()
        train = df[pd.to_datetime(df.day0) <= pd.Timestamp(cfg["period"]["train_end"])].copy()
        test = df[pd.to_datetime(df.day0) >= pd.Timestamp(cfg["period"]["test_start"])].copy()
        if train.empty or test.empty:
            raise RuntimeError("Train or test split is empty.")
        if train.target.nunique() < 2:
            raise RuntimeError("Training labels contain only one class; logistic regression cannot be fit.")

        X_train, y_train = train[FEATURES].to_numpy(), train.target.to_numpy(dtype=int)
        X_test, y_test = test[FEATURES].to_numpy(), test.target.to_numpy(dtype=int)

        model = Pipeline([
            ("scale", StandardScaler()),
            ("logit", LogisticRegression(max_iter=2000, random_state=int(cfg["model"]["seed"]))),
        ])
        model.fit(X_train, y_train)
        p_logit = model.predict_proba(X_test)[:, 1]
        pred_logit = (p_logit >= 0.5).astype(int)

        pred_up = np.ones_like(y_test)
        p_up = np.ones_like(y_test, dtype=float)
        majority = int(y_train.mean() >= 0.5)
        pred_majority = np.full_like(y_test, majority)
        p_majority = np.full_like(y_test, float(majority), dtype=float)

        predictions = []
        for name, pred, prob in [
            ("always_up", pred_up, p_up),
            ("training_majority", pred_majority, p_majority),
            ("logistic_regression", pred_logit, p_logit),
        ]:
            for event_id, yt, yp, pp in zip(test.event_id, y_test, pred, prob):
                predictions.append(dict(event_id=event_id, split="test_2020_2025", model=name,
                                        y_true=int(yt), y_pred=int(yp), p_up=float(pp)))

        metric_rows = []
        for name, pred, prob in [
            ("always_up", pred_up, p_up),
            ("training_majority", pred_majority, p_majority),
            ("logistic_regression", pred_logit, p_logit),
        ]:
            metric_rows.append(dict(model=name, n_test=len(y_test), **metrics(y_test, pred, prob)))

        diff, lo, hi = bootstrap_accuracy_difference(
            y_test, pred_logit, pred_up,
            draws=int(cfg["model"]["bootstrap_draws"]), seed=int(cfg["model"]["seed"]),
        )

        rng = np.random.default_rng(int(cfg["model"]["seed"]))
        shuffled = rng.permutation(y_train)
        shuffled_model = Pipeline([
            ("scale", StandardScaler()),
            ("logit", LogisticRegression(max_iter=2000, random_state=int(cfg["model"]["seed"]))),
        ])
        shuffled_model.fit(X_train, shuffled)
        shuffled_auc = roc_auc_score(y_test, shuffled_model.predict_proba(X_test)[:, 1]) if len(np.unique(y_test)) == 2 else np.nan

        pred_df = pd.DataFrame(predictions)
        metric_df = pd.DataFrame(metric_rows)
        diag_df = pd.DataFrame([
            dict(metric="logit_accuracy_minus_always_up", value=diff, ci_low=lo, ci_high=hi,
                 note="Paired bootstrap 95% CI on the 2020-2025 test events."),
            dict(metric="shuffled_label_auc", value=float(shuffled_auc), ci_low=np.nan, ci_high=np.nan,
                 note="Leakage diagnostic; should be near 0.50, with sampling variation."),
        ])

        con.execute("DELETE FROM model_predictions")
        con.execute("DELETE FROM model_metrics")
        con.execute("DELETE FROM model_diagnostics")
        con.register("pred_df", pred_df)
        con.execute("INSERT INTO model_predictions SELECT * FROM pred_df")
        con.unregister("pred_df")
        con.register("metric_df", metric_df)
        con.execute("INSERT INTO model_metrics SELECT * FROM metric_df")
        con.unregister("metric_df")
        con.register("diag_df", diag_df)
        con.execute("INSERT INTO model_diagnostics SELECT * FROM diag_df")
        con.unregister("diag_df")

        metric_df.to_csv(ROOT / cfg["paths"]["outputs"] / "model_metrics.csv", index=False)
        diag_df.to_csv(ROOT / cfg["paths"]["outputs"] / "model_diagnostics.csv", index=False)
    finally:
        con.close()

    print(metric_df.to_string(index=False))
    print("\nDiagnostics:\n", diag_df.to_string(index=False))


if __name__ == "__main__":
    main()

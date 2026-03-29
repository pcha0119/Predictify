"""
Evaluation metrics for the forecasting pipeline.

Metric choices
--------------
MAE    : Interpretable in the original scale (sales currency units).
RMSE   : Penalises large errors more; useful to detect outlier days.
MAPE   : Percentage error — but uses epsilon=1.0 to avoid explosion near zero.
sMAPE  : Symmetric MAPE — more robust when actuals are near zero (zero-sale days).
Bias   : Signed mean error — positive = over-forecast, negative = under-forecast.
WAPE   : Weighted Absolute Percentage Error = sum(|error|) / sum(|actual|).
         Equivalent to MAE / mean(actual); aggregates well across groups.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import REPORT_DIR

logger = logging.getLogger(__name__)


# ── atomic metric functions ───────────────────────────────────────────────────

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    epsilon: float = 1.0,
) -> float:
    """
    Mean Absolute Percentage Error.

    *epsilon* is added to the denominator to prevent explosion when actuals
    are near zero (common on zero-sales days for sparse items).
    """
    denom = np.maximum(np.abs(y_true), epsilon)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Symmetric Mean Absolute Percentage Error.

    Uses (|actual| + |forecast|) / 2 in the denominator, which handles
    zero actuals gracefully (returns 200% only when one of them is zero).
    """
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    denom = np.where(denom < 1e-9, 1e-9, denom)  # avoid divide-by-zero
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100)


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean signed error = mean(forecast - actual).
    Positive → model over-forecasts on average.
    Negative → model under-forecasts.
    """
    return float(np.mean(y_pred - y_true))


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Weighted Absolute Percentage Error = sum(|error|) / sum(|actual|).
    More stable than MAPE for heterogeneous group aggregation.
    """
    total_actual = np.sum(np.abs(y_true))
    if total_actual < 1e-9:
        return 0.0
    return float(np.sum(np.abs(y_true - y_pred)) / total_actual * 100)


# ── composite metric dict ─────────────────────────────────────────────────────

def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    grain: str,
    group_key: str,
    horizon: int,
    n_folds: int = 0,
) -> dict:
    """
    Compute all metrics and return as a flat dict ready for JSON serialisation.
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    # Guard: skip if no valid pairs
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if valid.sum() == 0:
        return {
            "model_name": model_name, "grain": grain,
            "group_key": group_key, "horizon": horizon,
            "n_folds": n_folds, "n_obs": 0,
            "mae": None, "rmse": None, "mape": None,
            "smape": None, "bias": None, "wape": None,
        }

    y_t = y_true[valid]
    y_p = y_pred[valid]

    return {
        "model_name": model_name,
        "grain": grain,
        "group_key": group_key,
        "horizon": horizon,
        "n_folds": n_folds,
        "n_obs": int(valid.sum()),
        "mae":   round(mae(y_t, y_p), 4),
        "rmse":  round(rmse(y_t, y_p), 4),
        "mape":  round(mape(y_t, y_p), 2),
        "smape": round(smape(y_t, y_p), 2),
        "bias":  round(bias(y_t, y_p), 4),
        "wape":  round(wape(y_t, y_p), 2),
    }


# ── comparison table ──────────────────────────────────────────────────────────

def compare_models(metric_records: list[dict]) -> pd.DataFrame:
    """
    Produce a model comparison table from a list of metric dicts.

    Rows  = model × grain × group_key × horizon
    Cols  = MAE, RMSE, sMAPE, WAPE, Bias
    Sorted by MAE ascending (best model first).
    """
    if not metric_records:
        return pd.DataFrame()

    df = pd.DataFrame(metric_records)

    # Sort: best first by MAE
    if "mae" in df.columns:
        df = df.sort_values("mae", ascending=True, na_position="last")

    df.reset_index(drop=True, inplace=True)
    return df


# ── persistence ───────────────────────────────────────────────────────────────

def save_evaluation_report(
    metric_records: list[dict],
    grain: str,
    suffix: str = "",
) -> Path:
    """
    Save metric records as JSON and CSV to REPORT_DIR.

    Returns the path of the JSON file.
    """
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{grain}{('_' + suffix) if suffix else ''}"

    json_path = REPORT_DIR / f"metrics_{tag}.json"
    csv_path  = REPORT_DIR / f"metrics_{tag}.csv"

    with open(json_path, "w") as f:
        json.dump(metric_records, f, indent=2, default=str)

    df = compare_models(metric_records)
    df.to_csv(csv_path, index=False)

    logger.info("Evaluation report saved: %s  (%d records)", json_path, len(metric_records))
    return json_path

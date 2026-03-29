"""
Core abstractions for the forecasting pipeline.

BaseForecaster     — ABC that all models implement.
ForecastingPipeline — orchestrates walk-forward validation + recursive
                      multi-step forecasting for any grain.
ForecastResult     — typed output container.
WalkForwardResult  — typed validation result.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    TRAIN_FRACTION,
    VAL_FRACTION,
    WALK_FORWARD_STEP,
    FORECAST_HORIZONS,
    MIN_ITEM_DAYS,
    SPARSE_FALLBACK_ORDER,
)
from feature_engineering.features import build_features_for_grain, get_feature_columns

logger = logging.getLogger(__name__)


# ── data classes ──────────────────────────────────────────────────────────────

@dataclass
class WalkForwardResult:
    model_name: str
    grain: str
    group_key: str
    horizon: int
    per_fold_mae: list[float]
    per_fold_rmse: list[float]
    n_folds: int
    mean_mae: float
    mean_rmse: float

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "grain": self.grain,
            "group_key": self.group_key,
            "horizon": self.horizon,
            "n_folds": self.n_folds,
            "mean_mae": round(self.mean_mae, 4),
            "mean_rmse": round(self.mean_rmse, 4),
            "per_fold_mae": [round(v, 4) for v in self.per_fold_mae],
            "per_fold_rmse": [round(v, 4) for v in self.per_fold_rmse],
        }


@dataclass
class ForecastResult:
    grain: str
    group_key: str
    horizon: int
    model_name: str
    forecast_df: pd.DataFrame          # columns: date, forecast, ci_lower, ci_upper
    actuals_df: pd.DataFrame           # columns: date, actual
    validation: WalkForwardResult | None = None
    fallback_level: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "grain": self.grain,
            "group_key": self.group_key,
            "horizon": self.horizon,
            "model_name": self.model_name,
            "fallback_level": self.fallback_level,
            "warnings": self.warnings,
            "validation": self.validation.to_dict() if self.validation else None,
            "forecast": self.forecast_df.to_dict(orient="records"),
            "actuals": self.actuals_df.to_dict(orient="records"),
        }


# ── abstract base ─────────────────────────────────────────────────────────────

class BaseForecaster(ABC):
    """Interface every forecasting model must satisfy."""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseForecaster":
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        ...

    def get_model_name(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        return f"{self.get_model_name()}()"


# ── walk-forward helpers ──────────────────────────────────────────────────────

def _time_split(
    df: pd.DataFrame,
    date_col: str = "date",
    train_frac: float = TRAIN_FRACTION,
    val_frac: float = VAL_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split sorted DataFrame into train / val / test by time fraction."""
    n = len(df)
    n_train = max(int(n * train_frac), 2)
    n_val = max(int(n * val_frac), 1)

    train = df.iloc[:n_train].copy()
    val   = df.iloc[n_train: n_train + n_val].copy()
    test  = df.iloc[n_train + n_val:].copy()
    return train, val, test


def _walk_forward_folds(
    df: pd.DataFrame,
    horizon: int,
    step: int = WALK_FORWARD_STEP,
    min_train_rows: int = 15,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Generate (train_df, val_df) pairs for expanding-window walk-forward CV.

    Each fold: train on rows 0..split_point-1, validate on rows
    split_point..split_point+horizon-1.  The split_point advances by *step*
    on each iteration.
    """
    n = len(df)
    folds: list[tuple[pd.DataFrame, pd.DataFrame]] = []

    # First split: use 70% of available data as initial training window
    split = max(int(n * TRAIN_FRACTION), min_train_rows)

    while split + horizon <= n:
        train_fold = df.iloc[:split].copy()
        val_fold   = df.iloc[split: split + horizon].copy()
        if len(train_fold) >= min_train_rows and len(val_fold) > 0:
            folds.append((train_fold, val_fold))
        split += step

    if not folds:
        logger.warning(
            "walk_forward_folds: not enough data for even one fold "
            "(n=%d, horizon=%d, min_train=%d). Returning single 70/30 split.",
            n, horizon, min_train_rows,
        )
        split_fallback = max(int(n * 0.7), min_train_rows)
        if split_fallback < n:
            folds.append((df.iloc[:split_fallback].copy(), df.iloc[split_fallback:].copy()))

    return folds


# ── forecasting pipeline ──────────────────────────────────────────────────────

class ForecastingPipeline:
    """
    Orchestrates feature engineering, walk-forward validation, final model
    training, and recursive multi-step future forecasting for a single grain.

    Usage
    -----
    pipeline = ForecastingPipeline(
        model=RidgeForecaster(),
        grain="store",
        group_cols=["store_id"],
    )
    result = pipeline.run(store_daily_df, group_value="KPHB", horizon=7)
    """

    def __init__(
        self,
        model: BaseForecaster,
        grain: str,
        group_cols: list[str],
        target_col: str = "sales_value",
    ) -> None:
        self.model = model
        self.grain = grain
        self.group_cols = group_cols
        self.target_col = target_col

    # ── public ────────────────────────────────────────────────────────────────

    def run(
        self,
        daily_df: pd.DataFrame,
        horizon: int,
        group_value: str | None = None,
        add_uncertainty: bool = True,
        n_bootstrap: int = 30,
    ) -> ForecastResult:
        """
        Full pipeline for one group × one horizon.

        Parameters
        ----------
        daily_df      : Daily aggregated DataFrame for the relevant grain.
        horizon       : Forecast horizon in days (7, 14, or 30).
        group_value   : Filter the DataFrame to this group (e.g. store_id).
                        Pass None for total-level.
        add_uncertainty: Whether to compute bootstrap CI.
        """
        warnings: list[str] = []

        # Filter to requested group
        if group_value and self.group_cols:
            mask = daily_df[self.group_cols[0]] == group_value
            df = daily_df[mask].copy()
        else:
            df = daily_df.copy()

        group_key = group_value or "total"

        if len(df) < 5:
            warnings.append(
                f"Insufficient data ({len(df)} rows) for group '{group_key}'. "
                "Using NaN forecast."
            )
            empty_fc = pd.DataFrame({
                "date": pd.date_range(
                    df["date"].max() + pd.Timedelta(days=1)
                    if len(df) > 0 else pd.Timestamp.today(),
                    periods=horizon
                ),
                "forecast": np.nan,
                "ci_lower": np.nan,
                "ci_upper": np.nan,
            })
            return ForecastResult(
                grain=self.grain, group_key=group_key, horizon=horizon,
                model_name=self.model.get_model_name(),
                forecast_df=empty_fc,
                actuals_df=df[["date", self.target_col]].rename(
                    columns={self.target_col: "actual"}
                ),
                warnings=warnings,
            )

        # Build features
        featured = build_features_for_grain(
            df, group_cols=self.group_cols, target_col=self.target_col
        )
        feature_cols = get_feature_columns(featured, self.target_col)

        if not feature_cols:
            warnings.append("No usable feature columns after feature engineering.")
            feature_cols = []

        # Walk-forward validation
        wf_result = self._walk_forward_validate(featured, feature_cols, horizon)

        # Train final model on all available featured data
        X_all = featured[feature_cols].fillna(0.0)
        y_all = featured[self.target_col]
        valid_mask = y_all.notna()
        self.model.fit(X_all[valid_mask], y_all[valid_mask])

        # Recursive forecast into the future
        forecast_df = self._recursive_forecast(featured, feature_cols, horizon)

        # Residual-bootstrap uncertainty intervals
        # We compute in-sample residuals then propagate uncertainty to the forecast.
        if add_uncertainty and len(X_all[valid_mask]) >= 10:
            try:
                y_hat_train = self.model.predict(X_all[valid_mask])
                residuals = y_all[valid_mask].values - y_hat_train

                from config import CI_LOWER, CI_UPPER, N_BOOTSTRAP, RANDOM_SEED
                rng = np.random.default_rng(RANDOM_SEED)
                boot_forecasts = np.zeros((n_bootstrap, horizon))

                point_vals = forecast_df["forecast"].values.copy()

                for b in range(n_bootstrap):
                    # Sample residuals for each future step
                    sampled = rng.choice(residuals, size=horizon, replace=True)
                    boot_forecasts[b] = np.maximum(point_vals + sampled, 0.0)

                forecast_df["ci_lower"] = np.percentile(boot_forecasts, CI_LOWER, axis=0)
                forecast_df["ci_upper"] = np.percentile(boot_forecasts, CI_UPPER, axis=0)
            except Exception as exc:
                warnings.append(f"Bootstrap CI failed: {exc}")
                logger.debug("Bootstrap error: %s", exc, exc_info=True)

        if "ci_lower" not in forecast_df.columns:
            # Fallback: simple ±20% band
            forecast_df["ci_lower"] = (forecast_df["forecast"] * 0.80).clip(lower=0)
            forecast_df["ci_upper"] = forecast_df["forecast"] * 1.20

        actuals_df = df[["date", self.target_col]].rename(
            columns={self.target_col: "actual"}
        )

        if len(wf_result.per_fold_mae) == 0:
            warnings.append(
                "Walk-forward CV could not produce any folds — history too short."
            )

        logger.info(
            "ForecastingPipeline.run: grain=%s group=%s model=%s horizon=%d "
            "n_folds=%d mean_mae=%.2f",
            self.grain, group_key, self.model.get_model_name(),
            horizon, wf_result.n_folds, wf_result.mean_mae,
        )

        return ForecastResult(
            grain=self.grain,
            group_key=group_key,
            horizon=horizon,
            model_name=self.model.get_model_name(),
            forecast_df=forecast_df,
            actuals_df=actuals_df,
            validation=wf_result,
            warnings=warnings,
        )

    # ── internal ──────────────────────────────────────────────────────────────

    def _walk_forward_validate(
        self,
        featured: pd.DataFrame,
        feature_cols: list[str],
        horizon: int,
    ) -> WalkForwardResult:
        """Expanding-window walk-forward cross-validation."""
        from evaluation.metrics import mae as _mae, rmse as _rmse

        folds = _walk_forward_folds(featured, horizon)
        per_fold_mae: list[float] = []
        per_fold_rmse: list[float] = []

        for train_fold, val_fold in folds:
            X_tr = train_fold[feature_cols].fillna(0.0)
            y_tr = train_fold[self.target_col].fillna(0.0)
            X_vl = val_fold[feature_cols].fillna(0.0)
            y_vl = val_fold[self.target_col].fillna(0.0).values

            if len(X_tr) < 5 or len(X_vl) == 0:
                continue

            try:
                model_copy = type(self.model)()  # fresh instance, same hyperparams
                model_copy.fit(X_tr, y_tr)
                y_pred = model_copy.predict(X_vl)
                per_fold_mae.append(float(_mae(y_vl, y_pred)))
                per_fold_rmse.append(float(_rmse(y_vl, y_pred)))
            except Exception as exc:
                logger.debug("Walk-forward fold failed: %s", exc)

        n_folds = len(per_fold_mae)
        mean_mae  = float(np.mean(per_fold_mae))  if per_fold_mae  else np.nan
        mean_rmse = float(np.mean(per_fold_rmse)) if per_fold_rmse else np.nan

        return WalkForwardResult(
            model_name=self.model.get_model_name(),
            grain=self.grain,
            group_key="",
            horizon=horizon,
            per_fold_mae=per_fold_mae,
            per_fold_rmse=per_fold_rmse,
            n_folds=n_folds,
            mean_mae=mean_mae,
            mean_rmse=mean_rmse,
        )

    def _recursive_forecast(
        self,
        featured: pd.DataFrame,
        feature_cols: list[str],
        horizon: int,
    ) -> pd.DataFrame:
        """
        Recursive multi-step forecast.

        For each step h = 1..horizon:
          1. Build a one-row feature vector by updating lag/rolling columns
             using the previous prediction.
          2. Predict one step.
          3. Append to forecast history so next step can use it as a lag.

        Returns DataFrame with columns: [date, forecast].
        """
        # Start from the last date in the featured set
        last_date = featured["date"].max()
        history = featured[self.target_col].values.copy().astype(float)

        forecast_dates: list[pd.Timestamp] = []
        forecast_values: list[float] = []

        for h in range(1, horizon + 1):
            forecast_date = last_date + pd.Timedelta(days=h)

            # Build feature row for this forecast step
            row = _build_forecast_feature_row(
                history=history,
                forecast_date=forecast_date,
                featured_template=featured,
                feature_cols=feature_cols,
            )

            X_pred = pd.DataFrame([row], columns=feature_cols).fillna(0.0)
            y_hat = float(self.model.predict(X_pred)[0])
            y_hat = max(0.0, y_hat)   # sales can't be negative

            forecast_dates.append(forecast_date)
            forecast_values.append(y_hat)
            history = np.append(history, y_hat)

        return pd.DataFrame({
            "date": forecast_dates,
            "forecast": forecast_values,
        })


def _build_forecast_feature_row(
    history: np.ndarray,
    forecast_date: pd.Timestamp,
    featured_template: pd.DataFrame,
    feature_cols: list[str],
) -> dict[str, float]:
    """
    Build a single feature vector for one recursive forecast step.

    Uses *history* (which grows with each predicted step) to fill lag features.
    Calendar features are computed from *forecast_date*.
    Rolling/EWM features are approximated from the tail of *history*.
    """
    row: dict[str, float] = {}
    n = len(history)

    for col in feature_cols:
        if col.startswith("lag_"):
            try:
                k = int(col.split("_")[1])
                row[col] = float(history[-k]) if n >= k else 0.0
            except (ValueError, IndexError):
                row[col] = 0.0

        elif col.startswith("rolling_mean_"):
            try:
                w = int(col.split("_")[2])
                tail = history[-w:] if n >= w else history
                row[col] = float(np.mean(tail)) if len(tail) > 0 else 0.0
            except (ValueError, IndexError):
                row[col] = 0.0

        elif col.startswith("rolling_std_"):
            try:
                w = int(col.split("_")[2])
                tail = history[-w:] if n >= w else history
                row[col] = float(np.std(tail)) if len(tail) > 1 else 0.0
            except (ValueError, IndexError):
                row[col] = 0.0

        elif col == "ewm_mean_7":
            tail = history[-min(14, n):]
            weights = np.array([0.75 ** i for i in range(len(tail) - 1, -1, -1)])
            row[col] = float(np.average(tail, weights=weights)) if len(tail) > 0 else 0.0

        elif col == "ewm_mean_14":
            tail = history[-min(28, n):]
            weights = np.array([0.88 ** i for i in range(len(tail) - 1, -1, -1)])
            row[col] = float(np.average(tail, weights=weights)) if len(tail) > 0 else 0.0

        elif col == "mom_1_2":
            row[col] = (float(history[-1]) - float(history[-2])) if n >= 2 else 0.0

        elif col == "mom_1_7":
            row[col] = (float(history[-1]) - float(history[-7])) if n >= 7 else 0.0

        elif col == "day_of_week":
            row[col] = float(forecast_date.dayofweek)
        elif col == "day_of_month":
            row[col] = float(forecast_date.day)
        elif col == "week_of_year":
            row[col] = float(forecast_date.isocalendar()[1])
        elif col == "month":
            row[col] = float(forecast_date.month)
        elif col == "quarter":
            row[col] = float(forecast_date.quarter)
        elif col == "is_weekend":
            row[col] = float(forecast_date.dayofweek >= 5)
        elif col == "is_month_start":
            row[col] = float(forecast_date.is_month_start)
        elif col == "is_month_end":
            row[col] = float(forecast_date.is_month_end)
        elif col == "week_of_month":
            row[col] = float((forecast_date.day - 1) // 7 + 1)
        elif col == "pre_holi_flag":
            row[col] = float(
                pd.Timestamp("2026-03-07") <= forecast_date <= pd.Timestamp("2026-03-14")
            )
        elif col == "holi_flag":
            row[col] = float(forecast_date == pd.Timestamp("2026-03-14"))
        elif col == "eid_flag":
            row[col] = float(
                pd.Timestamp("2026-03-29") <= forecast_date <= pd.Timestamp("2026-04-02")
            )
        elif col == "diwali_flag":
            row[col] = float(
                pd.Timestamp("2026-10-28") <= forecast_date <= pd.Timestamp("2026-11-05")
            )
        elif col == "festival_window":
            row[col] = float(
                (pd.Timestamp("2026-03-07") <= forecast_date <= pd.Timestamp("2026-03-14"))
                or (pd.Timestamp("2026-03-29") <= forecast_date <= pd.Timestamp("2026-04-02"))
                or (pd.Timestamp("2026-10-28") <= forecast_date <= pd.Timestamp("2026-11-05"))
            )
        else:
            # For any feature not explicitly handled, use the last known value
            if col in featured_template.columns:
                last_val = featured_template[col].iloc[-1]
                row[col] = float(last_val) if pd.notna(last_val) else 0.0
            else:
                row[col] = 0.0

    return row

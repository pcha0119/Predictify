"""
Naive baseline forecasters.

Every ML model must beat these before being considered useful.
All three baselines ignore X (feature matrix) entirely and use only y (target history).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from models.forecaster import BaseForecaster


class NaiveLastValue(BaseForecaster):
    """
    Forecast = last observed value, repeated for every future step.
    Simple but often hard to beat for stable series.
    """

    def __init__(self) -> None:
        self._last_value: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "NaiveLastValue":
        self._last_value = float(y.iloc[-1]) if len(y) > 0 else 0.0
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self._last_value)

    def get_model_name(self) -> str:
        return "NaiveLastValue"


class NaiveRollingMean(BaseForecaster):
    """
    Forecast = mean of the last 7 observed values.
    Smooths out single-day noise.
    """

    def __init__(self, window: int = 7) -> None:
        self._window = window
        self._mean: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "NaiveRollingMean":
        tail = y.dropna().tail(self._window)
        self._mean = float(tail.mean()) if len(tail) > 0 else 0.0
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self._mean)

    def get_model_name(self) -> str:
        return f"NaiveRollingMean(w={self._window})"


class NaiveSeasonalWeekly(BaseForecaster):
    """
    Forecast = same weekday from 1 week ago.
    Captures the dominant weekly pattern in retail data.

    If a matching weekday is not found in history (very short series),
    falls back to the 7-day rolling mean.
    """

    def __init__(self) -> None:
        self._weekday_means: dict[int, float] = {}
        self._fallback: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "NaiveSeasonalWeekly":
        # Need day_of_week in X
        if "day_of_week" in X.columns:
            combined = pd.DataFrame({"y": y.values, "dow": X["day_of_week"].values})
            self._weekday_means = (
                combined.groupby("dow")["y"].mean().to_dict()
            )
        self._fallback = float(y.dropna().tail(7).mean()) if len(y) > 0 else 0.0
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if "day_of_week" not in X.columns or not self._weekday_means:
            return np.full(len(X), self._fallback)

        preds = X["day_of_week"].map(
            lambda d: self._weekday_means.get(d, self._fallback)
        ).values.astype(float)
        return np.maximum(preds, 0.0)

    def get_model_name(self) -> str:
        return "NaiveSeasonalWeekly"


def get_all_baselines() -> dict[str, BaseForecaster]:
    """Return all three baseline instances keyed by display name."""
    return {
        "NaiveLastValue": NaiveLastValue(),
        "NaiveRollingMean7": NaiveRollingMean(window=7),
        "NaiveSeasonalWeekly": NaiveSeasonalWeekly(),
    }

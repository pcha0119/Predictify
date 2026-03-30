"""
Ridge Regression forecaster.

Uses sklearn Pipeline(StandardScaler → RidgeCV) so feature scale differences
don't bias coefficient magnitudes.  RidgeCV selects the best regularisation
strength via leave-one-out CV (fast, exact for linear models).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import RIDGE_ALPHAS, RANDOM_SEED
from models.forecaster import BaseForecaster

logger = logging.getLogger(__name__)


class RidgeForecaster(BaseForecaster):
    """
    Ridge regression with cross-validated alpha selection.

    Why Ridge for this dataset:
    - Lag features are highly correlated (lag_1 and lag_2 share most variance).
    - Regularisation prevents large, unstable coefficients on these correlated inputs.
    - LinearRegression without regularisation overfits badly on 25–30 training rows.
    """

    def __init__(
        self,
        alphas: list[float] | None = None,
        fit_intercept: bool = True,
    ) -> None:
        self.alphas = alphas or RIDGE_ALPHAS
        self.fit_intercept = fit_intercept
        self._pipeline: Pipeline | None = None
        self._feature_names: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RidgeForecaster":
        """
        Fit on non-NaN rows only.  Stores feature names for coefficient lookup.
        """
        self._feature_names = list(X.columns)

        # Drop rows where any feature or target is NaN
        mask = X.notna().all(axis=1) & y.notna()
        X_fit = X[mask].values
        y_fit = y[mask].values

        if len(X_fit) < 3:
            raise ValueError(
                f"RidgeForecaster.fit: only {len(X_fit)} clean rows — "
                "cannot train a meaningful model."
            )

        # RidgeCV uses generalised cross-validation (GCV) — leave-one-out exact
        ridge_cv = RidgeCV(
            alphas=self.alphas,
            fit_intercept=self.fit_intercept,
            scoring=None,     # uses default GCV
        )

        self._pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge",  ridge_cv),
        ])
        self._pipeline.fit(X_fit, y_fit)

        chosen_alpha = self._pipeline.named_steps["ridge"].alpha_
        logger.info(
            "RidgeForecaster fitted: n=%d, alpha*=%.4f",
            len(X_fit), chosen_alpha,
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._pipeline is None:
            raise RuntimeError("Call fit() before predict().")

        X_vals = X[self._feature_names].fillna(0.0).values
        preds = self._pipeline.predict(X_vals)
        return np.maximum(preds, 0.0)   # sales cannot be negative

    def get_coefficients(self) -> pd.Series:
        """
        Return feature importances as a Series indexed by feature name.
        Values are the Ridge coefficients in the *original* (unscaled) space,
        scaled by the standard deviation of each feature for comparability.
        """
        if self._pipeline is None:
            raise RuntimeError("Model has not been fitted yet.")

        scaler = self._pipeline.named_steps["scaler"]
        ridge  = self._pipeline.named_steps["ridge"]
        # Scale coefficients by feature std to get units of "sales per std-dev"
        scaled_coefs = ridge.coef_ * scaler.scale_
        return pd.Series(scaled_coefs, index=self._feature_names).sort_values(
            key=abs, ascending=False
        )

    def get_model_name(self) -> str:
        return "RidgeForecaster"

    @property
    def chosen_alpha(self) -> float | None:
        if self._pipeline is None:
            return None
        return float(self._pipeline.named_steps["ridge"].alpha_)

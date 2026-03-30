"""
Lasso and Elastic Net forecasters.

Why these after Ridge:
- Lasso can zero-out weak features entirely (feature selection built-in).
- Elastic Net combines L1 and L2 penalties: handles correlated features
  better than pure Lasso while still doing feature selection.
- Particularly useful when the feature space grows (item-level pooling, many
  category indicators, etc.).

Short-series safety: use cv=2 when n_train < 40 to avoid degenerate folds.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV, LassoCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import ELASTICNET_L1_RATIOS, LASSO_ALPHAS, RANDOM_SEED
from models.forecaster import BaseForecaster

logger = logging.getLogger(__name__)
# Avoid flooding terminal output for tiny/ill-conditioned slices.
warnings.filterwarnings(
    "ignore",
    category=ConvergenceWarning,
    module=r"sklearn\.linear_model\._coordinate_descent",
)

_MIN_ROWS_FOR_CV3 = 40   # below this, use cv=2 instead of cv=3


def _choose_cv(n: int) -> int:
    return 2 if n < _MIN_ROWS_FOR_CV3 else 3


class LassoForecaster(BaseForecaster):
    """
    Lasso regression with cross-validated alpha selection using TimeSeriesSplit.

    TimeSeriesSplit is used instead of standard KFold to respect the time
    ordering of the data — no future leakage in cross-validation.
    """

    def __init__(self, alphas: list[float] | None = None) -> None:
        self.alphas = alphas or LASSO_ALPHAS
        self._pipeline: Pipeline | None = None
        self._feature_names: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LassoForecaster":
        self._feature_names = list(X.columns)
        mask = X.notna().all(axis=1) & y.notna()
        X_fit = X[mask].values
        y_fit = y[mask].values

        if len(X_fit) < 4:
            raise ValueError(
                f"LassoForecaster.fit: only {len(X_fit)} clean rows."
            )

        cv = TimeSeriesSplit(n_splits=_choose_cv(len(X_fit)))
        lasso_cv = LassoCV(
            alphas=self.alphas,
            cv=cv,
            max_iter=20000,
            tol=1e-3,
            selection="random",
            random_state=RANDOM_SEED,
            fit_intercept=True,
        )
        self._pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("lasso",  lasso_cv),
        ])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self._pipeline.fit(X_fit, y_fit)

        chosen = self._pipeline.named_steps["lasso"].alpha_
        n_nonzero = int(np.sum(self._pipeline.named_steps["lasso"].coef_ != 0))
        logger.info(
            "LassoForecaster fitted: n=%d, alpha*=%.4f, non-zero features=%d/%d",
            len(X_fit), chosen, n_nonzero, len(self._feature_names),
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._pipeline is None:
            raise RuntimeError("Call fit() before predict().")
        X_vals = X[self._feature_names].fillna(0.0).values
        preds = self._pipeline.predict(X_vals)
        return np.maximum(preds, 0.0)

    def get_model_name(self) -> str:
        return "LassoForecaster"

    def get_selected_features(self) -> list[str]:
        """Return feature names with non-zero Lasso coefficients."""
        if self._pipeline is None:
            return []
        coefs = self._pipeline.named_steps["lasso"].coef_
        return [f for f, c in zip(self._feature_names, coefs) if c != 0]


class ElasticNetForecaster(BaseForecaster):
    """
    Elastic Net regression with cross-validated alpha and l1_ratio selection.

    The l1_ratio grid controls the L1 vs L2 mix:
    - l1_ratio = 1.0 → pure Lasso
    - l1_ratio = 0.0 → pure Ridge
    - Intermediate values balance feature selection and stability.
    """

    def __init__(
        self,
        alphas: list[float] | None = None,
        l1_ratios: list[float] | None = None,
    ) -> None:
        self.alphas = alphas or LASSO_ALPHAS
        self.l1_ratios = l1_ratios or ELASTICNET_L1_RATIOS
        self._pipeline: Pipeline | None = None
        self._feature_names: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ElasticNetForecaster":
        self._feature_names = list(X.columns)
        mask = X.notna().all(axis=1) & y.notna()
        X_fit = X[mask].values
        y_fit = y[mask].values

        if len(X_fit) < 4:
            raise ValueError(
                f"ElasticNetForecaster.fit: only {len(X_fit)} clean rows."
            )

        cv = TimeSeriesSplit(n_splits=_choose_cv(len(X_fit)))
        enet_cv = ElasticNetCV(
            alphas=self.alphas,
            l1_ratio=self.l1_ratios,
            cv=cv,
            max_iter=20000,
            tol=1e-3,
            selection="random",
            random_state=RANDOM_SEED,
            fit_intercept=True,
        )
        self._pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("enet",   enet_cv),
        ])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self._pipeline.fit(X_fit, y_fit)

        model = self._pipeline.named_steps["enet"]
        logger.info(
            "ElasticNetForecaster fitted: n=%d, alpha*=%.4f, l1_ratio*=%.2f",
            len(X_fit), model.alpha_, model.l1_ratio_,
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._pipeline is None:
            raise RuntimeError("Call fit() before predict().")
        X_vals = X[self._feature_names].fillna(0.0).values
        preds = self._pipeline.predict(X_vals)
        return np.maximum(preds, 0.0)

    def get_model_name(self) -> str:
        return "ElasticNetForecaster"

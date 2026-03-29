"""
Bootstrap prediction intervals for linear forecasters.

Approach: residual bootstrap.
  1. Fit the model on the full training set.
  2. Compute in-sample residuals: r_i = y_i - ŷ_i.
  3. For n_bootstrap iterations:
     a. Sample residuals with replacement.
     b. Create perturbed targets: y_tilde = ŷ_train + sampled_residuals.
     c. Refit model on (X_train, y_tilde).
     d. Predict on X_future → one bootstrap trajectory.
  4. Aggregate trajectories into percentile bands.

Why residual bootstrap (not block bootstrap):
- Linear models are fast to refit (< 1ms), so 50 iterations is trivial.
- Residual bootstrap is appropriate when residuals are roughly i.i.d.
- Block bootstrap would be more correct for strongly autocorrelated residuals
  but requires tuning block size — overkill for this prototype.

Notes:
- Only works with BaseForecaster subclasses that support fresh instantiation
  via type(model)() with no required constructor arguments.
- Predictions are clipped to >= 0 before percentile computation.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CI_LOWER, CI_UPPER, N_BOOTSTRAP, RANDOM_SEED
from models.forecaster import BaseForecaster

logger = logging.getLogger(__name__)


def bootstrap_prediction_intervals(
    model: BaseForecaster,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_future: pd.DataFrame,
    n_bootstrap: int = N_BOOTSTRAP,
    ci_lower: int = CI_LOWER,
    ci_upper: int = CI_UPPER,
    random_state: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute bootstrap prediction intervals.

    Parameters
    ----------
    model        : Already-fitted BaseForecaster instance (used to get residuals).
    X_train      : Training features (no NaN).
    y_train      : Training target (no NaN).
    X_future     : Feature matrix for the forecast horizon.
    n_bootstrap  : Number of bootstrap replicates.
    ci_lower     : Lower percentile for the interval (e.g. 10).
    ci_upper     : Upper percentile for the interval (e.g. 90).
    random_state : Seed for reproducibility.

    Returns
    -------
    (point_forecast, lower_bound, upper_bound)
      Each is a numpy array of length len(X_future).
    """
    rng = np.random.default_rng(random_state)

    # Point forecast from the already-fitted model
    point_forecast = np.maximum(model.predict(X_future), 0.0)

    # In-sample residuals
    y_hat_train = model.predict(X_train)
    residuals = y_train.values - y_hat_train

    bootstrap_trajectories: list[np.ndarray] = []

    for i in range(n_bootstrap):
        # Sample residuals with replacement (same length as training set)
        sampled_residuals = rng.choice(residuals, size=len(y_train), replace=True)
        y_perturbed = pd.Series(
            np.maximum(y_hat_train + sampled_residuals, 0.0),
            index=y_train.index,
        )

        try:
            # Fresh model instance — same class, same default hyperparams
            boot_model = type(model)()
            boot_model.fit(X_train, y_perturbed)
            boot_pred = np.maximum(boot_model.predict(X_future), 0.0)
            bootstrap_trajectories.append(boot_pred)
        except Exception as exc:
            logger.debug("Bootstrap replicate %d failed: %s", i, exc)
            # If a replicate fails, use the point forecast as a neutral fill
            bootstrap_trajectories.append(point_forecast.copy())

    if not bootstrap_trajectories:
        logger.warning("All bootstrap replicates failed. Returning ±20%% band.")
        return (
            point_forecast,
            np.maximum(point_forecast * 0.80, 0.0),
            point_forecast * 1.20,
        )

    trajectories = np.array(bootstrap_trajectories)   # shape: (n_bootstrap, horizon)

    lower_bound = np.percentile(trajectories, ci_lower, axis=0)
    upper_bound = np.percentile(trajectories, ci_upper, axis=0)

    logger.info(
        "Bootstrap intervals: %d replicates, CI [%d%%, %d%%], horizon=%d.",
        len(bootstrap_trajectories), ci_lower, ci_upper, len(X_future),
    )

    return (
        point_forecast,
        np.maximum(lower_bound, 0.0),
        np.maximum(upper_bound, 0.0),
    )

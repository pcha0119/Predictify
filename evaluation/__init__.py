"""Evaluation layer — metrics, model comparison, and report persistence."""
from .metrics import (
    mae, rmse, mape, smape, bias,
    compute_all_metrics,
    compare_models,
    save_evaluation_report,
)

__all__ = [
    "mae", "rmse", "mape", "smape", "bias",
    "compute_all_metrics", "compare_models", "save_evaluation_report",
]

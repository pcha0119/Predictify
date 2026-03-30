"""
Feature engineering pipeline.

Design rules (enforced throughout):
  - ZERO future leakage: every lag and rolling feature references strictly
    past observations (minimum shift = 1).
  - Group-aware: features are computed independently per group (store, category,
    item) so that store A's lag_1 never bleeds into store B.
  - Missing-day fill: gaps in a series are zero-filled before feature computation
    (no sales on a day = 0 sales, not NaN).
  - Short-series safety: lag values that would exceed available history are
    dropped so at least MIN_KEEP_ROWS rows survive after feature trimming.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import LAG_DAYS, ROLLING_WINDOWS

logger = logging.getLogger(__name__)

MIN_KEEP_ROWS = 10   # minimum training rows to retain after lag trimming


# ── calendar features ─────────────────────────────────────────────────────────

def add_calendar_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Add standard calendar features derived from *date_col*."""
    dt = df[date_col]
    df["day_of_week"]   = dt.dt.dayofweek          # 0=Mon, 6=Sun
    df["day_of_month"]  = dt.dt.day
    df["week_of_year"]  = dt.dt.isocalendar().week.astype(int)
    df["month"]         = dt.dt.month
    df["quarter"]       = dt.dt.quarter
    df["is_weekend"]    = (dt.dt.dayofweek >= 5).astype(int)
    df["is_month_start"] = dt.dt.is_month_start.astype(int)
    df["is_month_end"]   = dt.dt.is_month_end.astype(int)
    # Week-of-month (1-based)
    df["week_of_month"] = ((dt.dt.day - 1) // 7 + 1)
    return df


def add_indian_festival_flags(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """
    Add binary flags for major Indian festivals relevant to sweet / namkeen
    retail sales.  Dates are hardcoded for 2026.

    Holi (Mar 14, 2026) is the only festival in the observed 36-day window;
    the pre-Holi shopping spike is clearly visible on 2026-03-07 (Saturday).
    Future months include Eid (~Mar-Apr 2026) and Diwali (Nov 2026).
    """
    dt = df[date_col]

    # Pre-Holi shopping window: Mar 7–14, 2026
    pre_holi_start = pd.Timestamp("2026-03-07")
    pre_holi_end   = pd.Timestamp("2026-03-14")
    df["pre_holi_flag"] = (
        (dt >= pre_holi_start) & (dt <= pre_holi_end)
    ).astype(int)

    # Holi day itself
    df["holi_flag"] = (dt == pd.Timestamp("2026-03-14")).astype(int)

    # Eid al-Fitr 2026 (approximate: Mar 31 ± 1 day depending on moon)
    eid_start = pd.Timestamp("2026-03-29")
    eid_end   = pd.Timestamp("2026-04-02")
    df["eid_flag"] = (
        (dt >= eid_start) & (dt <= eid_end)
    ).astype(int)

    # Diwali 2026 (approximate: Oct 30 – Nov 3)
    diwali_start = pd.Timestamp("2026-10-28")
    diwali_end   = pd.Timestamp("2026-11-05")
    df["diwali_flag"] = (
        (dt >= diwali_start) & (dt <= diwali_end)
    ).astype(int)

    # Union festival window
    df["festival_window"] = (
        (df["pre_holi_flag"] | df["holi_flag"] |
         df["eid_flag"] | df["diwali_flag"])
    ).astype(int)

    return df


# ── lag / rolling helpers ─────────────────────────────────────────────────────

def _clip_lags_to_series_length(
    lags: list[int],
    n_rows: int,
    min_keep: int = MIN_KEEP_ROWS,
) -> list[int]:
    """
    Remove lags so large that fewer than *min_keep* rows would survive.
    E.g. with 36 rows and min_keep=10: max_lag = 36 - 10 = 26 → drop lag_28.
    """
    max_lag = max(n_rows - min_keep, 1)
    clipped = [l for l in lags if l <= max_lag]
    dropped = set(lags) - set(clipped)
    if dropped:
        logger.debug(
            "Clipped lags %s (series length %d, min_keep %d).",
            sorted(dropped), n_rows, min_keep,
        )
    return clipped


def _add_lag_rolling_for_group(
    group: pd.DataFrame,
    target_col: str,
    lag_days: list[int],
    rolling_windows: list[int],
) -> pd.DataFrame:
    """
    Compute lag and rolling features for a SINGLE group's sorted DataFrame.
    The group must already be sorted by date and zero-filled for gaps.
    """
    n = len(group)
    clipped_lags = _clip_lags_to_series_length(lag_days, n)

    y = group[target_col]

    # Lag features — shift by k means lag_k(t) = y(t-k), strictly past
    for k in clipped_lags:
        group[f"lag_{k}"] = y.shift(k)

    # Rolling mean/std — rolling over the *already-shifted* series (shift(1))
    y_lagged = y.shift(1)
    for w in rolling_windows:
        group[f"rolling_mean_{w}"] = y_lagged.rolling(w, min_periods=max(1, w // 2)).mean()
        group[f"rolling_std_{w}"]  = y_lagged.rolling(w, min_periods=max(2, w // 2)).std().fillna(0)

    # Exponentially weighted mean (trend-sensitive, handles sparse spikes better)
    group["ewm_mean_7"]  = y_lagged.ewm(span=7,  min_periods=2).mean()
    group["ewm_mean_14"] = y_lagged.ewm(span=14, min_periods=2).mean()

    # Momentum features: difference between recent lags (direction signal)
    if "lag_1" in group.columns and "lag_2" in group.columns:
        group["mom_1_2"] = group["lag_1"] - group["lag_2"]
    if "lag_1" in group.columns and "lag_7" in group.columns:
        group["mom_1_7"] = group["lag_1"] - group["lag_7"]

    # Lagged price (if available)
    if "avg_unit_price" in group.columns:
        group["lag_price_1"]       = group["avg_unit_price"].shift(1)
        group["rolling_avg_price"] = group["avg_unit_price"].shift(1).rolling(7, min_periods=1).mean()

    # Lagged receipt count (proxy for foot traffic)
    if "receipt_count" in group.columns:
        group["lag_receipts_1"] = group["receipt_count"].shift(1)
        group["rolling_receipts_7"] = (
            group["receipt_count"].shift(1).rolling(7, min_periods=1).mean()
        )

    return group


# ── public API ────────────────────────────────────────────────────────────────

def build_features_for_grain(
    daily_df: pd.DataFrame,
    group_cols: list[str],
    target_col: str = "sales_value",
    lag_days: list[int] | None = None,
    rolling_windows: list[int] | None = None,
    date_col: str = "date",
    drop_leading_nans: bool = True,
) -> pd.DataFrame:
    """
    Build the full supervised feature set for a given aggregation grain.

    Parameters
    ----------
    daily_df        : Daily aggregated DataFrame (one row per date per group).
    group_cols      : Columns that define the group, e.g. ['store_id'] or
                      ['category'] or [] for total-level.
    target_col      : Column to forecast; used as source for lag/rolling features.
    lag_days        : Override config.LAG_DAYS.
    rolling_windows : Override config.ROLLING_WINDOWS.
    date_col        : Name of the date column.
    drop_leading_nans: If True, drop rows where ALL lag features are NaN
                       (the warm-up period at the start of each group).

    Returns
    -------
    Featured DataFrame with no future leakage.
    """
    if lag_days is None:
        lag_days = LAG_DAYS
    if rolling_windows is None:
        rolling_windows = ROLLING_WINDOWS

    df = daily_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    results: list[pd.DataFrame] = []

    if group_cols:
        groups = df.groupby(group_cols, sort=False)
    else:
        # Total-level: treat as a single group
        df["_total_group"] = "total"
        groups = df.groupby("_total_group", sort=False)

    for grp_key, grp in groups:
        grp = grp.sort_values(date_col).copy()

        # Zero-fill missing dates within the group's span
        full_idx = pd.date_range(grp[date_col].min(), grp[date_col].max(), freq="D")
        grp = grp.set_index(date_col).reindex(full_idx)

        # Fill group identifier columns so they don't become NaN after reindex
        for col in group_cols:
            if col in grp.columns:
                grp[col] = grp[col].ffill().bfill()

        # Zero-fill numeric cols that represent "no sales this day"
        for col in [target_col, "sales_qty", "receipt_count",
                    "unique_items", "unique_stores", "unique_categories"]:
            if col in grp.columns:
                grp[col] = grp[col].fillna(0.0)

        # avg_unit_price: forward-fill price (price doesn't change to 0 on off days)
        if "avg_unit_price" in grp.columns:
            grp["avg_unit_price"] = grp["avg_unit_price"].ffill()

        grp.index.name = date_col
        grp = grp.reset_index()

        # Calendar & festival features
        grp = add_calendar_features(grp, date_col)
        grp = add_indian_festival_flags(grp, date_col)

        # Lag & rolling features (group-local, no cross-group bleed)
        grp = _add_lag_rolling_for_group(grp, target_col, lag_days, rolling_windows)

        results.append(grp)

    out = pd.concat(results, ignore_index=True)

    # Remove the synthetic grouping column for total-level
    if "_total_group" in out.columns:
        out.drop(columns=["_total_group"], inplace=True)

    # Drop warm-up rows where ALL lag features are NaN
    if drop_leading_nans:
        lag_cols = [c for c in out.columns if c.startswith("lag_")]
        if lag_cols:
            all_nan_mask = out[lag_cols].isna().all(axis=1)
            out = out[~all_nan_mask].copy()

    out.sort_values([*group_cols, date_col] if group_cols else [date_col], inplace=True)
    out.reset_index(drop=True, inplace=True)

    logger.info(
        "build_features_for_grain(group=%s, target=%s): %d rows, %d features.",
        group_cols, target_col, len(out), len(out.columns),
    )
    return out


def get_feature_columns(
    df: pd.DataFrame,
    target_col: str = "sales_value",
    exclude_extra: list[str] | None = None,
) -> list[str]:
    """
    Return the list of feature column names ready for model .fit() calls.

    Excludes: date, target, ID/group columns, raw transaction cols, and any
    columns explicitly listed in *exclude_extra*.
    """
    non_feature = {
        target_col, "date", "store_id", "item_id", "category", "brand",
        "division", "uom", "retail_product_code", "description", "base_uom",
        # raw transaction cols
        "net_amount", "quantity", "price", "net_price", "receipt_no",
        "trans_time", "is_return", "return_value",
        # aggregation metadata that would leak
        "unique_items", "unique_stores", "unique_categories",
    }
    if exclude_extra:
        non_feature.update(exclude_extra)

    feature_cols = [
        c for c in df.columns
        if c not in non_feature
        and not c.startswith("_")
        and df[c].dtype in [np.float64, np.float32, np.int64, np.int32, np.int8, np.uint8, "float64", "float32", "int64", "int32"]
    ]
    return feature_cols

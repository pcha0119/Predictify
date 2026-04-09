"""
Synchronous pipeline runner — executed in a thread pool to avoid blocking the event loop.
"""

from __future__ import annotations

import logging
import traceback
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    DATA_DIR, FORECAST_DIR, REPORT_DIR,
    WORKBOOK_PATH, IMPORTED_FLAT_PATH,
    MIN_ITEM_DAYS,
)
from api.state import _state
from api.helpers import _save_json

logger = logging.getLogger(__name__)


def _run_pipeline_sync() -> None:
    """
    Full synchronous pipeline: ingest → clean → feature eng → train → forecast.
    Called via run_in_executor so it does not block the event loop.
    """
    import datetime

    _state["started_at"] = datetime.datetime.utcnow().isoformat()
    _state["error"] = None
    _state["last_run_artifacts"] = {}

    try:
        from data_ingestion.loader import load_workbook, get_workbook_summary
        from data_cleaning.cleaner import (
            clean_sale_lines, clean_header, clean_item_master,
            build_fact_sales, aggregate_to_daily, save_cleaned_data,
        )
        from data_cleaning.quarantine import QuarantineLog
        from models.baselines import get_all_baselines
        from models.ridge_model import RidgeForecaster
        from models.lasso_elnet_model import LassoForecaster, ElasticNetForecaster
        from models.forecaster import ForecastingPipeline
        from evaluation.metrics import compute_all_metrics, save_evaluation_report

        # 1. Load data (flat import takes priority if newer)
        logger.info("[pipeline] Step 1: Loading data …")
        use_flat = (
            IMPORTED_FLAT_PATH.exists()
            and (
                not WORKBOOK_PATH.exists()
                or IMPORTED_FLAT_PATH.stat().st_mtime > WORKBOOK_PATH.stat().st_mtime
            )
        )
        if use_flat:
            from data_ingestion.loader import load_flat_data
            logger.info("[pipeline] Using imported flat data: %s", IMPORTED_FLAT_PATH)
            sheets = load_flat_data(IMPORTED_FLAT_PATH, fmt="csv")
        else:
            sheets = load_workbook()
        summary = get_workbook_summary(sheets)

        # 2. Clean
        logger.info("[pipeline] Step 2: Cleaning data …")
        q = QuarantineLog()
        sl_clean  = clean_sale_lines(sheets["sale_lines"], q)
        hdr_clean = clean_header(sheets["header"], q)
        im_clean  = clean_item_master(sheets["item_master"], q)

        q.flush(DATA_DIR / "quarantine.csv")
        summary["quarantine"] = q.summary()

        # 3. Build fact table + aggregates
        logger.info("[pipeline] Step 3: Building canonical tables …")
        fact = build_fact_sales(sl_clean, hdr_clean, im_clean)
        fs_daily, item_daily, store_daily, cat_daily, total_daily = aggregate_to_daily(fact)
        save_cleaned_data(fs_daily, item_daily, store_daily, cat_daily, total_daily)

        summary["fact_rows"] = len(fs_daily)
        summary["item_daily_rows"] = len(item_daily)
        summary["store_daily_rows"] = len(store_daily)
        summary["category_daily_rows"] = len(cat_daily)
        summary["total_daily_rows"] = len(total_daily)
        _save_json(summary, REPORT_DIR / "summary.json")
        _state["summary"] = summary

        # 4+5. Train models + walk-forward + forecast for each grain
        logger.info("[pipeline] Step 4–5: Training and forecasting …")
        HORIZONS = [7, 14, 30]
        all_metrics: list[dict] = []

        models_to_run = {
            "baselines": get_all_baselines(),
            "ridge": RidgeForecaster(),
            "lasso": LassoForecaster(),
            "elnet": ElasticNetForecaster(),
        }

        grains = {
            "total":    (total_daily, [],           "total",     "sales_value"),
            "store":    (store_daily, ["store_id"], "store_id",  "sales_value"),
            "category": (cat_daily,  ["category"],  "category", "sales_value"),
            "item":     (item_daily, ["item_id"],   "item_id",  "sales_qty"),
        }

        import pandas as pd
        FORECAST_DIR.mkdir(parents=True, exist_ok=True)

        for grain_name, (grain_df, group_cols, group_id_col, target_col) in grains.items():
            if grain_name == "item":
                item_day_counts = grain_df.groupby("item_id")["date"].nunique()
                eligible_items = item_day_counts[item_day_counts >= MIN_ITEM_DAYS].index.tolist()
                grain_df = grain_df[grain_df["item_id"].isin(eligible_items)].copy()
                logger.info(
                    "[pipeline] Item grain: %d items have >= %d days of data (out of %d total)",
                    len(eligible_items), MIN_ITEM_DAYS, len(item_day_counts),
                )

            groups = (
                grain_df[group_id_col].unique().tolist()
                if group_id_col in grain_df.columns
                else ["total"]
            )

            for model_tag, model_obj in models_to_run.items():
                if isinstance(model_obj, dict):
                    model_instances = model_obj
                else:
                    model_instances = {model_obj.get_model_name(): model_obj}

                for m_name, m_instance in model_instances.items():
                    all_fc_records = []
                    for grp_val in groups:
                        for horizon in HORIZONS:
                            pipeline = ForecastingPipeline(
                                model=m_instance,
                                grain=grain_name,
                                group_cols=group_cols,
                                target_col=target_col,
                            )
                            try:
                                result = pipeline.run(
                                    grain_df,
                                    horizon=horizon,
                                    group_value=grp_val if group_id_col in grain_df.columns else None,
                                    add_uncertainty=(model_tag != "baselines"),
                                    n_bootstrap=20,
                                )
                                fc_rows = result.forecast_df.copy()
                                fc_rows["grain"] = grain_name
                                fc_rows["group_key"] = grp_val
                                fc_rows["horizon"] = horizon
                                fc_rows["model_name"] = m_name
                                fc_rows["target_col"] = target_col

                                if grain_name == "item":
                                    item_meta = grain_df[grain_df["item_id"] == grp_val].iloc[0]
                                    fc_rows["uom"] = item_meta.get("uom", "")
                                    fc_rows["description"] = item_meta.get("description", "")
                                    fc_rows["category"] = item_meta.get("category", "")

                                all_fc_records.append(fc_rows)

                                if result.validation:
                                    all_metrics.append(result.validation.to_dict())

                            except Exception as exc:
                                logger.warning(
                                    "Pipeline failed for grain=%s group=%s model=%s h=%d: %s",
                                    grain_name, grp_val, m_name, horizon, exc,
                                )

                    if all_fc_records:
                        fc_df = pd.concat(all_fc_records, ignore_index=True)
                        fc_path = FORECAST_DIR / f"forecast_{grain_name}_{m_name}.csv"
                        fc_df.to_csv(fc_path, index=False)
                        _state["last_run_artifacts"][f"forecast_{grain_name}_{m_name}"] = str(fc_path)

        # 6. Save evaluation report
        if all_metrics:
            save_evaluation_report(all_metrics, grain="all")

        _state["status"] = "complete"
        _state["finished_at"] = datetime.datetime.utcnow().isoformat()
        logger.info("[pipeline] Complete.")

    except Exception as exc:
        _state["status"] = "error"
        _state["error"] = traceback.format_exc()
        logger.error("[pipeline] FAILED: %s", exc, exc_info=True)

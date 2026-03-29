"""
CLI entry point for the Sales Forecasting pipeline.

Usage
-----
  python run_pipeline.py                      # full pipeline, all grains, all horizons
  python run_pipeline.py --grain store        # only store-level
  python run_pipeline.py --horizon 7 14       # only 7 and 14-day horizons
  python run_pipeline.py --skip-training      # re-run forecast from saved artifacts
  python run_pipeline.py --serve              # run pipeline then start the API server

Implementation sequence (matches SOP §16):
  1.  Inspect workbook schema
  2.  Clean and normalise all three sheets
  3.  Join sheets into enriched fact table
  4.  Standardise revenue signs and UOM
  5.  Create canonical daily fact tables
  6.  Build naive baselines
  7.  Engineer time-based features
  8.  Train Ridge model
  9.  Train Lasso / Elastic Net
  10. Walk-forward validation
  11. Compare metrics
  12. Generate future forecasts
  13. Export results
  14. Print summary
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# -- logging setup (before any imports that use logging) -----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")

# -- project root on sys.path --------------------------------------------------
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import (
    DATA_DIR, FORECAST_DIR, REPORT_DIR,
    FORECAST_HORIZONS,
    WORKBOOK_PATH, WORKBOOK_PATH_RAW,
)


# -- argument parser -----------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sales Forecasting Pipeline — builds and evaluates forecasting models."
    )
    p.add_argument(
        "--grain",
        nargs="+",
        choices=["total", "store", "category"],
        default=["total", "store", "category"],
        help="Forecast grains to run (default: all three).",
    )
    p.add_argument(
        "--horizon",
        nargs="+",
        type=int,
        default=FORECAST_HORIZONS,
        help="Forecast horizons in days (default: 7 14 30).",
    )
    p.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training; re-run forecasting from already-saved models.",
    )
    p.add_argument(
        "--serve",
        action="store_true",
        help="After the pipeline completes, start the FastAPI server.",
    )
    p.add_argument(
        "--bootstrap",
        type=int,
        default=50,
        help="Number of bootstrap replicates for uncertainty intervals (default: 50).",
    )
    return p.parse_args()


# -- helpers -------------------------------------------------------------------

def _section(title: str) -> None:
    bar = "-" * 60
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)


def _check_data(key: str, df) -> None:
    print(f"    {key:30s} {len(df):>8,} rows  {len(df.columns):>4} cols")


# -- pipeline steps ------------------------------------------------------------

def run_pipeline(
    grains: list[str],
    horizons: list[int],
    skip_training: bool = False,
    n_bootstrap: int = 50,
) -> None:
    t0 = time.perf_counter()

    # -- Step 1: Load workbook -------------------------------------------------
    _section("Step 1 — Load workbook")
    from data_ingestion.loader import load_workbook, get_workbook_summary
    sheets = load_workbook()
    summary = get_workbook_summary(sheets)
    print(f"  Sheets loaded:")
    for k, v in sheets.items():
        _check_data(k, v)
    print(f"\n  Date range: {summary['date_range']['min']} -> {summary['date_range']['max']}"
          f"  ({summary['date_range']['calendar_days']} days)")
    print(f"  Stores: {summary['stores']['count']}  |  "
          f"Sold items: {summary['items']['sold_count']}  |  "
          f"Item master: {summary['items']['master_count']}")

    # -- Step 2-4: Clean all sheets --------------------------------------------
    _section("Step 2–4 — Clean, normalise, join")
    from data_cleaning.quarantine import QuarantineLog
    from data_cleaning.cleaner import (
        clean_sale_lines, clean_header, clean_item_master,
        build_fact_sales, aggregate_to_daily, save_cleaned_data,
    )

    q = QuarantineLog()
    sl_clean  = clean_sale_lines(sheets["sale_lines"], q)
    hdr_clean = clean_header(sheets["header"], q)
    im_clean  = clean_item_master(sheets["item_master"], q)

    print(f"  Quarantine summary: {q.summary() or 'no rows quarantined'}")
    q.flush(DATA_DIR / "quarantine.csv")

    # -- Step 5: Canonical tables ----------------------------------------------
    _section("Step 5 — Build canonical daily tables")
    fact = build_fact_sales(sl_clean, hdr_clean, im_clean)
    fs_daily, store_daily, cat_daily, total_daily = aggregate_to_daily(fact)
    save_cleaned_data(fs_daily, store_daily, cat_daily, total_daily)
    _check_data("fact_sales_daily", fs_daily)
    _check_data("fact_store_daily", store_daily)
    _check_data("fact_category_daily", cat_daily)
    _check_data("fact_total_daily", total_daily)

    # Persist workbook summary for the API /data/summary endpoint
    import json as _json
    summary["fact_rows"] = len(fs_daily)
    summary["store_daily_rows"] = len(store_daily)
    summary["category_daily_rows"] = len(cat_daily)
    summary["total_daily_rows"] = len(total_daily)
    summary["quarantine"] = q.summary()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_DIR / "summary.json", "w") as _f:
        _json.dump(summary, _f, indent=2, default=str)

    # -- Data quality checks ---------------------------------------------------
    _section("Data quality checks")
    _run_dq_checks(fact, fs_daily, store_daily, cat_daily, total_daily)

    # -- Step 6-9: Model setup -------------------------------------------------
    _section("Step 6–9 — Models")
    from models.baselines import get_all_baselines
    from models.ridge_model import RidgeForecaster
    from models.lasso_elnet_model import LassoForecaster, ElasticNetForecaster
    from models.forecaster import ForecastingPipeline

    model_registry = {
        **get_all_baselines(),
        "RidgeForecaster": RidgeForecaster(),
        "LassoForecaster": LassoForecaster(),
        "ElasticNetForecaster": ElasticNetForecaster(),
    }
    print(f"  Models registered: {list(model_registry.keys())}")

    grain_config = {
        "total":    (total_daily, [],           None),
        "store":    (store_daily, ["store_id"], "store_id"),
        "category": (cat_daily,  ["category"],  "category"),
    }

    # -- Step 10-13: Walk-forward CV + forecast --------------------------------
    _section("Step 10–13 — Walk-forward validation + forecast")
    import pandas as pd
    from evaluation.metrics import compute_all_metrics, save_evaluation_report

    FORECAST_DIR.mkdir(parents=True, exist_ok=True)
    all_metrics: list[dict] = []

    for grain_name in grains:
        if grain_name not in grain_config:
            logger.warning("Unknown grain '%s' — skipping.", grain_name)
            continue

        daily_df, group_cols, group_id_col = grain_config[grain_name]
        groups = (
            daily_df[group_id_col].unique().tolist()
            if group_id_col and group_id_col in daily_df.columns
            else ["total"]
        )

        print(f"\n  Grain: {grain_name.upper()}  ({len(groups)} groups × {len(horizons)} horizons "
              f"× {len(model_registry)} models)")

        for model_name, model_obj in model_registry.items():
            all_fc_rows = []

            for grp_val in groups:
                for horizon in horizons:
                    pipeline = ForecastingPipeline(
                        model=model_obj,
                        grain=grain_name,
                        group_cols=group_cols,
                        target_col="sales_value",
                    )
                    try:
                        result = pipeline.run(
                            daily_df,
                            horizon=horizon,
                            group_value=grp_val if group_id_col else None,
                            add_uncertainty=(model_name not in get_all_baselines()),
                            n_bootstrap=n_bootstrap,
                        )
                        fc = result.forecast_df.copy()
                        fc["grain"] = grain_name
                        fc["group_key"] = grp_val
                        fc["horizon"] = horizon
                        fc["model_name"] = model_name
                        all_fc_rows.append(fc)

                        if result.validation:
                            vd = result.validation.to_dict()
                            vd["group_key"] = grp_val
                            all_metrics.append(vd)

                        if result.warnings:
                            for w in result.warnings:
                                logger.warning("  [%s/%s/%s] %s", grain_name, grp_val, model_name, w)

                    except Exception as exc:
                        logger.warning(
                            "  FAILED: grain=%s group=%s model=%s h=%d — %s",
                            grain_name, grp_val, model_name, horizon, exc,
                        )

            if all_fc_rows:
                fc_df = pd.concat(all_fc_rows, ignore_index=True)
                fc_path = FORECAST_DIR / f"forecast_{grain_name}_{model_name}.csv"
                fc_df.to_csv(fc_path.with_suffix(".csv"), index=False)
                print(f"    -> {grain_name}/{model_name}: {len(fc_df)} forecast rows saved.")

    # -- Step 14: Save metrics + print summary ---------------------------------
    _section("Step 14 — Evaluation summary")
    if all_metrics:
        from evaluation.metrics import compare_models
        report_path = save_evaluation_report(all_metrics, grain="all")
        comparison = compare_models(all_metrics)
        if not comparison.empty:
            cols_to_show = [c for c in ["model_name", "grain", "horizon",
                                         "n_folds", "mean_mae", "mean_rmse"]
                            if c in comparison.columns]
            print(comparison[cols_to_show].head(20).to_string(index=False))
        print(f"\n  Full report: {report_path}")
    else:
        print("  No metrics collected (all models may have failed).")

    elapsed = time.perf_counter() - t0
    print(f"\n{'='*60}")
    print(f"  Pipeline complete in {elapsed:.1f}s")
    print(f"  Artifacts: {FORECAST_DIR}")
    print(f"{'='*60}")


def _run_dq_checks(fact, fs_daily, store_daily, cat_daily, total_daily) -> None:
    """Print data quality check results."""
    checks = [
        ("No null dates in fact table",
         fact["date"].isna().sum() == 0),
        ("No null store_id in fact table",
         fact["store_id"].isna().sum() == 0),
        ("No null item_id in fact table",
         fact["item_id"].isna().sum() == 0),
        ("All sales_value >= 0",
         (fact["sales_value"] < 0).sum() == 0),
        ("All sales_qty >= 0",
         (fact["sales_qty"] < 0).sum() == 0),
        ("Total daily rows > 0",
         len(total_daily) > 0),
        ("Store daily rows > 0",
         len(store_daily) > 0),
        ("Category daily rows > 0",
         len(cat_daily) > 0),
    ]
    for name, passed in checks:
        status = "PASS" if passed else "FAIL"
        symbol = "OK" if passed else "FAIL"
        print(f"    [{status}] {symbol} {name}")


# -- main ----------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    print("\n" + "=" * 60)
    print("  SALES FORECASTING PIPELINE")
    print("  Grains   :", args.grain)
    print("  Horizons :", args.horizon)
    print("  Bootstrap:", args.bootstrap)
    print("=" * 60)

    run_pipeline(
        grains=args.grain,
        horizons=args.horizon,
        skip_training=args.skip_training,
        n_bootstrap=args.bootstrap,
    )

    if args.serve:
        print("\nStarting API server at http://localhost:8000 …")
        print("Open http://localhost:8000/ui/index.html for the dashboard.")
        import uvicorn
        uvicorn.run(
            "api.app:app",
            host="0.0.0.0",
            port=8000,
            reload=False,
            log_level="info",
        )


if __name__ == "__main__":
    main()

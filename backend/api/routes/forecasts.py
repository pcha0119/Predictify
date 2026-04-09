"""Routes: /forecasts/{grain}, /forecasts/{grain}/export, /evaluation/metrics"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import DATA_DIR, FORECAST_DIR, REPORT_DIR
from api.helpers import _read_fc_csv, _read_act_csv

router = APIRouter(tags=["forecasts"])


@router.get("/forecasts/{grain}")
async def get_forecasts(
    grain: str,
    horizon: int = Query(7, ge=7, le=30),
    store_id: str | None = Query(None),
    category: str | None = Query(None),
    item_id: str | None = Query(None),
    model: str = Query("RidgeForecaster"),
) -> dict:
    """Return forecast + actuals for a given grain, horizon, and group (Plotly-ready)."""
    if grain not in ("total", "store", "category", "item"):
        raise HTTPException(400, f"Invalid grain '{grain}'. Use: total|store|category|item.")

    fc_path = FORECAST_DIR / f"forecast_{grain}_{model}.csv"
    if not fc_path.exists():
        available = list(FORECAST_DIR.glob(f"forecast_{grain}_*.csv"))
        if not available:
            raise HTTPException(404, f"No forecasts found for grain='{grain}'. Run the pipeline first.")
        fc_path = available[0]
        model = fc_path.stem.replace(f"forecast_{grain}_", "")

    df = _read_fc_csv(fc_path)
    df = df[df["horizon"] == horizon]
    if "group_key" in df.columns:
        df["group_key"] = df["group_key"].astype(str)

    group_key = "total"
    if grain == "store" and store_id:
        df = df[df["group_key"] == str(store_id)]
        group_key = store_id
    elif grain == "category" and category:
        df = df[df["group_key"] == str(category)]
        group_key = category
    elif grain == "item" and item_id:
        df = df[df["group_key"] == str(item_id)]
        group_key = item_id
    elif "group_key" in df.columns:
        first = df["group_key"].iloc[0] if len(df) > 0 else "total"
        df = df[df["group_key"] == first]
        group_key = first

    if df.empty:
        raise HTTPException(404, "No forecast data for the specified filters.")

    df = df.sort_values("date")

    actuals_map = {
        "total":    (DATA_DIR / "fact_total_daily.csv",    "sales_value"),
        "store":    (DATA_DIR / "fact_store_daily.csv",    "sales_value"),
        "category": (DATA_DIR / "fact_category_daily.csv", "sales_value"),
        "item":     (DATA_DIR / "fact_item_daily.csv",     "sales_qty"),
    }
    actuals_df = pd.DataFrame()
    act_path, act_col = actuals_map.get(grain, (None, "sales_value"))
    if act_path and act_path.exists():
        act = _read_act_csv(act_path)
        if grain == "store" and store_id and "store_id" in act.columns:
            act = act[act["store_id"] == store_id]
        elif grain == "category" and category and "category" in act.columns:
            act = act[act["category"] == category]
        elif grain == "item" and item_id and "item_id" in act.columns:
            act = act[act["item_id"] == item_id]
        actuals_df = act[["date", act_col]].rename(columns={act_col: "actual"}).sort_values("date")

    item_meta = {}
    if grain == "item" and not df.empty:
        item_meta = {
            "uom": df["uom"].iloc[0] if "uom" in df.columns else "",
            "description": df["description"].iloc[0] if "description" in df.columns else "",
            "category": df["category"].iloc[0] if "category" in df.columns else "",
            "target_col": "sales_qty",
        }

    return {
        "grain": grain,
        "group_key": group_key,
        "model_name": model,
        "horizon": horizon,
        "dates": df["date"].astype(str).tolist(),
        "forecast": df["forecast"].round(2).tolist(),
        "ci_lower": df["ci_lower"].round(2).tolist() if "ci_lower" in df.columns else [],
        "ci_upper": df["ci_upper"].round(2).tolist() if "ci_upper" in df.columns else [],
        "actuals_dates": actuals_df["date"].astype(str).tolist() if not actuals_df.empty else [],
        "actuals": actuals_df["actual"].round(2).tolist() if not actuals_df.empty else [],
        **item_meta,
    }


@router.get("/forecasts/{grain}/export")
async def export_forecasts(
    grain: str,
    horizon: int = Query(7, ge=7, le=30),
    model: str = Query("RidgeForecaster"),
) -> StreamingResponse:
    """Download forecast as XLSX."""
    fc_path = FORECAST_DIR / f"forecast_{grain}_{model}.csv"
    if not fc_path.exists():
        available = list(FORECAST_DIR.glob(f"forecast_{grain}_*.csv"))
        if not available:
            raise HTTPException(404, f"No forecasts for grain='{grain}'.")
        fc_path = available[0]

    df = _read_fc_csv(fc_path)
    df = df[df["horizon"] == horizon].sort_values(["group_key", "date"])

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="Forecasts", index=False)

    buf.seek(0)
    filename = f"forecast_{grain}_{model}_h{horizon}.xlsx"
    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/evaluation/metrics")
async def get_metrics(
    grain: str = Query("all"),
    model: str | None = Query(None),
) -> dict:
    """Return evaluation metrics, optionally filtered by model name."""
    metrics_path = REPORT_DIR / f"metrics_{grain}.json"
    if not metrics_path.exists():
        metrics_path = REPORT_DIR / "metrics_all.json"
    if not metrics_path.exists():
        raise HTTPException(404, "Metrics not found. Run the pipeline first.")

    with open(metrics_path) as f:
        records = json.load(f)

    if model:
        records = [r for r in records if r.get("model_name") == model]

    records = sorted(records, key=lambda r: r.get("mean_mae", float("inf")))
    return {"records": records, "count": len(records)}

"""Routes: /stores, /items, /items/forecast_summary, /categories"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import DATA_DIR, FORECAST_DIR
from api.helpers import _load_csv_or_404

router = APIRouter(tags=["catalog"])


@router.get("/stores")
async def list_stores() -> list[str]:
    df = _load_csv_or_404(DATA_DIR / "fact_store_daily.csv")
    return sorted(df["store_id"].dropna().unique().tolist())


@router.get("/items")
async def list_items() -> list[dict]:
    """Return all items with UOM, description, category, and recent sales stats."""
    df = _load_csv_or_404(DATA_DIR / "fact_item_daily.csv")

    agg: dict = dict(
        uom=("uom", "first"),
        category=("category", "first"),
        total_qty=("sales_qty", "sum"),
        total_value=("sales_value", "sum"),
        days_with_sales=("date", "nunique"),
        avg_daily_qty=("sales_qty", "mean"),
    )
    if "description" in df.columns:
        agg["description"] = ("description", "first")
    else:
        agg["description"] = ("item_id", "first")

    items = df.groupby("item_id", as_index=False).agg(**agg)
    items = items.sort_values("total_value", ascending=False)
    return items.to_dict(orient="records")


@router.get("/items/forecast_summary")
async def item_forecast_summary(
    horizon: int = Query(7, ge=7, le=30),
    model: str = Query("RidgeForecaster"),
) -> dict:
    """Production planning summary: forecasted daily qty per item over the given horizon."""
    fc_path = FORECAST_DIR / f"forecast_item_{model}.csv"
    if not fc_path.exists():
        available = list(FORECAST_DIR.glob("forecast_item_*.csv"))
        if not available:
            raise HTTPException(404, "No item-level forecasts found. Run the pipeline first.")
        fc_path = available[0]
        model = fc_path.stem.replace("forecast_item_", "")

    df = pd.read_csv(fc_path, parse_dates=["date"])
    df = df[df["horizon"] == horizon]
    if "group_key" in df.columns:
        df["group_key"] = df["group_key"].astype(str)

    if df.empty:
        raise HTTPException(404, f"No item forecasts for horizon={horizon}.")

    items = []
    for item_id, grp in df.groupby("group_key"):
        grp = grp.sort_values("date")
        uom = grp["uom"].iloc[0] if "uom" in grp.columns else ""
        desc = grp["description"].iloc[0] if "description" in grp.columns else ""
        category = grp["category"].iloc[0] if "category" in grp.columns else ""

        items.append({
            "item_id": item_id,
            "description": desc,
            "uom": uom,
            "category": category,
            "total_forecast_qty": round(grp["forecast"].sum(), 2),
            "avg_daily_qty": round(grp["forecast"].mean(), 2),
            "horizon": horizon,
            "model": model,
            "forecast_dates": grp["date"].astype(str).tolist(),
            "forecast_values": grp["forecast"].round(2).tolist(),
            "ci_lower": grp["ci_lower"].round(2).tolist() if "ci_lower" in grp.columns else [],
            "ci_upper": grp["ci_upper"].round(2).tolist() if "ci_upper" in grp.columns else [],
        })

    items.sort(key=lambda x: x["total_forecast_qty"], reverse=True)
    nos_items = [i for i in items if i["uom"] == "NOS"]
    kgs_items = [i for i in items if i["uom"] == "KGS"]
    other_items = [i for i in items if i["uom"] not in ("NOS", "KGS")]

    return {
        "model": model,
        "horizon": horizon,
        "total_items": len(items),
        "nos_items": nos_items,
        "nos_count": len(nos_items),
        "kgs_items": kgs_items,
        "kgs_count": len(kgs_items),
        "other_items": other_items,
        "all_items": items,
    }


@router.get("/categories")
async def list_categories() -> list[str]:
    df = _load_csv_or_404(DATA_DIR / "fact_category_daily.csv")
    return sorted(df["category"].dropna().unique().tolist())

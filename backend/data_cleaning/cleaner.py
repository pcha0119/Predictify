"""
Data cleaning pipeline.

Steps per sheet:
  1. Rename columns → snake_case using config maps.
  2. Parse dates, cast numerics.
  3. Quarantine invalid rows (null keys, null dates, impossible values).
  4. Sign-correct Net Amount → positive sales_value.
  5. Normalise UOM.
  6. Deduplicate where needed.

Then:
  - build_fact_sales : joins Sale Lines → Item Master → Header
  - aggregate_to_daily : produces the four canonical daily tables
  - save_cleaned_data : persists parquet + csv to DATA_DIR
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    RENAME_SALE_LINES,
    RENAME_HEADER,
    RENAME_ITEM_MASTER,
    UOM_NORMALISE,
    DATA_DIR,
)
from data_cleaning.quarantine import QuarantineLog

logger = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────

def _cast_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _strip_str_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Strip leading/trailing whitespace from all object columns."""
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip()
        # Replace the string literal "nan" produced by .astype(str) back to NaN
        df[c] = df[c].replace("nan", np.nan)
    return df


# ── sheet cleaners ────────────────────────────────────────────────────────────

def clean_sale_lines(
    df_raw: pd.DataFrame,
    quarantine: QuarantineLog,
) -> pd.DataFrame:
    """
    Clean the Sale Lines sheet into a standardised transaction-level DataFrame.

    Key transformations
    -------------------
    - Rename columns (config.RENAME_SALE_LINES).
    - Parse date, cast numeric columns.
    - Quarantine: null date, null store_id, null item_id, null net_amount.
    - Sign correction:
        * net_amount < 0  → sales_value = -net_amount  (true sale)
        * net_amount > 0  → is_return = True, sales_value = 0,
                            return_value = net_amount   (reversal)
        * net_amount == 0 → sales_value = 0
    - sales_qty = abs(quantity)  (returns have negative quantity)
    - Normalise UOM; unknowns → 'OTHER'.
    - Cast store_id / item_id to str.
    """
    df = df_raw.copy()

    # 1. Rename only columns that exist (workbook may have extra cols)
    rename_map = {k: v for k, v in RENAME_SALE_LINES.items() if k in df.columns}
    df.rename(columns=rename_map, inplace=True)

    # 2. Strip string columns
    df = _strip_str_cols(df)

    # 3. Parse date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # 4. Cast numerics
    df = _cast_numeric(df, ["net_amount", "quantity", "price", "net_price"])

    # 5. Quarantine: null date
    bad_date = df["date"].isna()
    quarantine.add(df[bad_date], reason="null_date", source_sheet="sale_lines")
    df = df[~bad_date].copy()

    # 6. Quarantine: null store_id
    bad_store = df["store_id"].isna() | (df["store_id"] == "")
    quarantine.add(df[bad_store], reason="null_store_id", source_sheet="sale_lines")
    df = df[~bad_store].copy()

    # 7. Quarantine: null item_id
    bad_item = df["item_id"].isna() | (df["item_id"] == "")
    quarantine.add(df[bad_item], reason="null_item_id", source_sheet="sale_lines")
    df = df[~bad_item].copy()

    # 8. Quarantine: null net_amount
    bad_amt = df["net_amount"].isna()
    quarantine.add(df[bad_amt], reason="null_net_amount", source_sheet="sale_lines")
    df = df[~bad_amt].copy()

    # 9. Sign correction
    df["is_return"] = df["net_amount"] > 0
    df["return_value"] = np.where(df["is_return"], df["net_amount"], 0.0)
    df["sales_value"] = np.where(
        df["net_amount"] < 0,
        -df["net_amount"],   # flip sign: negative net_amount → positive sales
        0.0,                 # returns and zero-amount rows contribute 0 to sales
    )

    # 10. Normalise quantity (returns have negative qty; take abs for volume tracking)
    df["sales_qty"] = df["quantity"].abs().fillna(0.0)

    # 11. Normalise UOM
    df["uom"] = df["uom"].map(lambda x: UOM_NORMALISE.get(str(x).strip(), "OTHER"))

    # 12. Cast IDs to string
    df["store_id"] = df["store_id"].astype(str).str.strip()
    df["item_id"] = df["item_id"].astype(str).str.strip()
    df["receipt_no"] = df["receipt_no"].astype(str).str.strip()

    logger.info(
        "clean_sale_lines: %d usable rows, %d quarantined.",
        len(df), quarantine.total,
    )
    return df.reset_index(drop=True)


def clean_header(
    df_raw: pd.DataFrame,
    quarantine: QuarantineLog,
) -> pd.DataFrame:
    """
    Clean the Header sheet into a slim receipt-level enrichment table.

    Used only for JOIN enrichment (transaction_type, gross_amount).
    The canonical date axis comes from Sale Lines, not Header.
    """
    df = df_raw.copy()

    rename_map = {k: v for k, v in RENAME_HEADER.items() if k in df.columns}
    df.rename(columns=rename_map, inplace=True)
    df = _strip_str_cols(df)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = _cast_numeric(df, ["header_net_amount", "gross_amount"])

    # Quarantine rows with null receipt_no
    if "receipt_no" in df.columns:
        bad = df["receipt_no"].isna() | (df["receipt_no"].astype(str).str.strip() == "")
        quarantine.add(df[bad], reason="null_receipt_no_header", source_sheet="header")
        df = df[~bad].copy()
        df["receipt_no"] = df["receipt_no"].astype(str).str.strip()

    # Quarantine null dates
    bad_date = df["date"].isna()
    quarantine.add(df[bad_date], reason="null_date_header", source_sheet="header")
    df = df[~bad_date].copy()

    # Deduplicate: keep first occurrence per (receipt_no, store_id)
    key_cols = [c for c in ["receipt_no", "store_id"] if c in df.columns]
    n_before = len(df)
    df = df.drop_duplicates(subset=key_cols, keep="first")
    dupes = n_before - len(df)
    if dupes:
        logger.debug("Header: dropped %d duplicate receipt rows.", dupes)

    if "store_id" in df.columns:
        df["store_id"] = df["store_id"].astype(str).str.strip()

    logger.info("clean_header: %d usable rows.", len(df))
    return df.reset_index(drop=True)


def clean_item_master(
    df_raw: pd.DataFrame,
    quarantine: QuarantineLog,
) -> pd.DataFrame:
    """
    Clean the Item Master sheet into a product dimension lookup table.
    """
    df = df_raw.copy()

    rename_map = {k: v for k, v in RENAME_ITEM_MASTER.items() if k in df.columns}
    df.rename(columns=rename_map, inplace=True)
    df = _strip_str_cols(df)

    # Quarantine null item_id
    bad = df["item_id"].isna() | (df["item_id"].astype(str).str.strip() == "")
    quarantine.add(df[bad], reason="null_item_id_master", source_sheet="item_master")
    df = df[~bad].copy()

    df["item_id"] = df["item_id"].astype(str).str.strip()

    # Fill missing category with UNKNOWN
    if "category" in df.columns:
        df["category"] = df["category"].fillna("UNKNOWN").replace("", "UNKNOWN")

    # Deduplicate on item_id (keep first — master data should be unique)
    n_before = len(df)
    df = df.drop_duplicates(subset=["item_id"], keep="first")
    dupes = n_before - len(df)
    if dupes:
        logger.debug("Item Master: dropped %d duplicate item_id rows.", dupes)

    # Keep only the columns we need to avoid pollution of the fact table
    keep = [c for c in ["item_id", "description", "base_uom", "category",
                         "brand", "division", "retail_product_code"] if c in df.columns]
    df = df[keep].copy()

    logger.info("clean_item_master: %d usable rows.", len(df))
    return df.reset_index(drop=True)


# ── join & build canonical fact table ─────────────────────────────────────────

def build_fact_sales(
    sale_lines: pd.DataFrame,
    header: pd.DataFrame,
    item_master: pd.DataFrame,
) -> pd.DataFrame:
    """
    Produce the enriched transaction-level fact table.

    Join order
    ----------
    1. sale_lines LEFT JOIN item_master  → adds category, brand, division
    2. result    LEFT JOIN header        → adds transaction_type, gross_amount

    Unmatched item rows get category='UNKNOWN'.
    Unmatched receipt rows get transaction_type=NaN (non-critical; header is enrichment only).
    """
    # -- Join 1: item dimension
    im_cols = ["item_id"] + [
        c for c in ["category", "brand", "division", "description",
                     "retail_product_code", "base_uom"]
        if c in item_master.columns
    ]
    fact = sale_lines.merge(
        item_master[im_cols],
        on="item_id",
        how="left",
        suffixes=("", "_master"),
    )

    # Prefer Item Master description over Sale Lines description
    # (Sale Lines often has generic "Imported item" placeholder)
    if "description_master" in fact.columns:
        fact["description"] = fact["description_master"].fillna(fact["description"])
        fact = fact.drop(columns=["description_master"])

    # Fill missing category from join
    if "category" in fact.columns:
        fact["category"] = fact["category"].fillna("UNKNOWN").replace("", "UNKNOWN")
    else:
        fact["category"] = "UNKNOWN"

    # -- Join 2: receipt header enrichment
    hdr_cols = [c for c in ["receipt_no", "store_id", "transaction_type",
                              "gross_amount", "nature_of_supply"]
                if c in header.columns]
    if len(hdr_cols) > 1:  # at least receipt_no + one enrichment column
        fact = fact.merge(
            header[hdr_cols],
            on=[c for c in ["receipt_no", "store_id"] if c in hdr_cols],
            how="left",
            suffixes=("", "_hdr"),
        )

    # Compute avg_unit_price safely (guard against divide-by-zero)
    safe_qty = np.maximum(fact["sales_qty"], 0.001)
    fact["avg_unit_price"] = (fact["sales_value"] / safe_qty).where(
        fact["sales_qty"] > 0, np.nan
    )

    logger.info("build_fact_sales: %d enriched rows.", len(fact))
    return fact.reset_index(drop=True)


# ── aggregate to daily canonical tables ───────────────────────────────────────

def aggregate_to_daily(
    fact_sales: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Aggregate the enriched fact table to five canonical daily tables.

    Returns
    -------
    fact_sales_daily    : date × store_id × item_id
    fact_item_daily     : date × item_id  (summed across stores — for item-level forecasting)
    fact_store_daily    : date × store_id
    fact_category_daily : date × category
    fact_total_daily    : date
    """

    # ── item-store-day granularity ────────────────────────────────────────────
    grp_item = fact_sales.groupby(
        ["date", "store_id", "item_id"], as_index=False
    ).agg(
        category=("category", "first"),
        brand=("brand", "first") if "brand" in fact_sales.columns else ("item_id", "count"),
        division=("division", "first") if "division" in fact_sales.columns else ("item_id", "count"),
        uom=("uom", "first"),
        sales_qty=("sales_qty", "sum"),
        sales_value=("sales_value", "sum"),
        receipt_count=("receipt_no", "nunique"),
    )
    # Fix brand/division columns if item_master lacked them
    for col in ["brand", "division"]:
        if col not in grp_item.columns:
            grp_item[col] = "UNKNOWN"
        else:
            grp_item[col] = grp_item[col].fillna("UNKNOWN")

    safe_qty = np.maximum(grp_item["sales_qty"], 0.001)
    grp_item["avg_unit_price"] = (grp_item["sales_value"] / safe_qty).where(
        grp_item["sales_qty"] > 0, np.nan
    )

    # ── item-day granularity (summed across all stores) ────────────────────────
    grp_item_agg = fact_sales.groupby(
        ["date", "item_id"], as_index=False
    ).agg(
        category=("category", "first"),
        brand=("brand", "first") if "brand" in fact_sales.columns else ("item_id", "count"),
        division=("division", "first") if "division" in fact_sales.columns else ("item_id", "count"),
        uom=("uom", "first"),
        sales_qty=("sales_qty", "sum"),
        sales_value=("sales_value", "sum"),
        receipt_count=("receipt_no", "nunique"),
        unique_stores=("store_id", "nunique"),
    )
    for col in ["brand", "division"]:
        if col not in grp_item_agg.columns:
            grp_item_agg[col] = "UNKNOWN"
        else:
            grp_item_agg[col] = grp_item_agg[col].fillna("UNKNOWN")

    # Bring in item description from item_master (via fact_sales if available)
    if "description" in fact_sales.columns:
        desc_map = (
            fact_sales.dropna(subset=["description"])
            .drop_duplicates(subset=["item_id"])[["item_id", "description"]]
        )
        grp_item_agg = grp_item_agg.merge(desc_map, on="item_id", how="left")
        grp_item_agg["description"] = grp_item_agg["description"].fillna("")

    safe_qty_item = np.maximum(grp_item_agg["sales_qty"], 0.001)
    grp_item_agg["avg_unit_price"] = (grp_item_agg["sales_value"] / safe_qty_item).where(
        grp_item_agg["sales_qty"] > 0, np.nan
    )

    # ── store-day granularity ─────────────────────────────────────────────────
    grp_store = fact_sales.groupby(
        ["date", "store_id"], as_index=False
    ).agg(
        sales_qty=("sales_qty", "sum"),
        sales_value=("sales_value", "sum"),
        receipt_count=("receipt_no", "nunique"),
        unique_items=("item_id", "nunique"),
    )

    # ── category-day granularity ──────────────────────────────────────────────
    grp_cat = fact_sales.groupby(
        ["date", "category"], as_index=False
    ).agg(
        sales_qty=("sales_qty", "sum"),
        sales_value=("sales_value", "sum"),
        receipt_count=("receipt_no", "nunique"),
        unique_items=("item_id", "nunique"),
        unique_stores=("store_id", "nunique"),
    )

    # ── total-day granularity ─────────────────────────────────────────────────
    grp_total = fact_sales.groupby(
        ["date"], as_index=False
    ).agg(
        sales_qty=("sales_qty", "sum"),
        sales_value=("sales_value", "sum"),
        receipt_count=("receipt_no", "nunique"),
        unique_stores=("store_id", "nunique"),
        unique_items=("item_id", "nunique"),
        unique_categories=("category", "nunique"),
    )

    for df in [grp_item, grp_item_agg, grp_store, grp_cat, grp_total]:
        df.sort_values("date", inplace=True)
        df.reset_index(drop=True, inplace=True)

    logger.info(
        "aggregate_to_daily: item-store-day=%d | item-day=%d | store-day=%d | "
        "category-day=%d | total-day=%d",
        len(grp_item), len(grp_item_agg), len(grp_store), len(grp_cat), len(grp_total),
    )
    return grp_item, grp_item_agg, grp_store, grp_cat, grp_total


# ── persistence ───────────────────────────────────────────────────────────────

def save_cleaned_data(
    fact_sales_daily: pd.DataFrame,
    item_daily: pd.DataFrame,
    store_daily: pd.DataFrame,
    category_daily: pd.DataFrame,
    total_daily: pd.DataFrame,
) -> None:
    """
    Save all five canonical tables to DATA_DIR as CSV.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    tables = {
        "fact_sales_daily": fact_sales_daily,
        "fact_item_daily": item_daily,
        "fact_store_daily": store_daily,
        "fact_category_daily": category_daily,
        "fact_total_daily": total_daily,
    }

    for name, df in tables.items():
        csv_path = DATA_DIR / f"{name}.csv"
        df.to_csv(csv_path, index=False)
        logger.info("Saved %s: %d rows -> %s", name, len(df), csv_path)

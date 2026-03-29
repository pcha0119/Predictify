"""
Data ingestion: load, validate, and summarise the Excel workbook.

Responsibilities:
- Load all three sheets in one pass.
- Drop unnamed/empty columns that Excel appends.
- Validate required columns exist per config.
- Return raw DataFrames keyed by logical name (no renaming here — that is the
  cleaner's job).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import pandas as pd

import sys, os
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    SHEET_HEADER,
    SHEET_SALE_LINES,
    SHEET_ITEM_MASTER,
    REQUIRED_COLS_HEADER,
    REQUIRED_COLS_SALE_LINES,
    REQUIRED_COLS_ITEM_MASTER,
    WORKBOOK_PATH_RAW,
    WORKBOOK_PATH,
    COLUMN_ALIASES,
    REQUIRED_FLAT_COLS,
    OPTIONAL_FLAT_COLS,
    IMPORTED_FLAT_PATH,
)

logger = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────

def _drop_unnamed_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns whose names match the Excel auto-generated pattern Unnamed: N."""
    mask = df.columns.str.match(r"^Unnamed")
    dropped = list(df.columns[mask])
    if dropped:
        logger.debug("Dropping unnamed columns: %s", dropped)
    return df.loc[:, ~mask]


def _validate_required_cols(
    df: pd.DataFrame, required: list[str], sheet_name: str
) -> None:
    """Raise ValueError listing every missing required column."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Sheet '{sheet_name}' is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )


# ── public API ────────────────────────────────────────────────────────────────

def load_workbook(
    path: Union[str, Path, None] = None,
) -> dict[str, pd.DataFrame]:
    """
    Load the three sheets from the workbook and return raw DataFrames.

    Parameters
    ----------
    path : str | Path | None
        Path to the .xlsx file.  If None, falls back to WORKBOOK_PATH then
        WORKBOOK_PATH_RAW.

    Returns
    -------
    dict with keys:  'header', 'sale_lines', 'item_master'
    """
    # resolve path
    if path is None:
        if WORKBOOK_PATH.exists():
            path = WORKBOOK_PATH
        else:
            path = WORKBOOK_PATH_RAW
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Workbook not found at: {path}")

    logger.info("Loading workbook: %s", path)

    # Load each sheet individually to avoid high memory with sheet_name=None
    # on very large files and to give clearer per-sheet error messages.
    raw: dict[str, pd.DataFrame] = {}
    sheet_map = {
        "header": SHEET_HEADER,
        "sale_lines": SHEET_SALE_LINES,
        "item_master": SHEET_ITEM_MASTER,
    }

    # Item Master has 2 extra metadata rows before the real column header
    # (row 0: "AI Item 27…", row 1: blank, row 2: actual column names)
    header_row_map = {
        "header":      0,
        "sale_lines":  0,
        "item_master": 2,
    }

    for key, sheet_name in sheet_map.items():
        try:
            df = pd.read_excel(
                path,
                sheet_name=sheet_name,
                engine="openpyxl",
                header=header_row_map[key],
            )
        except Exception as exc:
            raise ValueError(
                f"Could not read sheet '{sheet_name}' from {path}: {exc}"
            ) from exc

        df = _drop_unnamed_cols(df)
        logger.info("  %-12s -> %d rows, %d columns", sheet_name, len(df), len(df.columns))
        raw[key] = df

    # Validate required columns
    _validate_required_cols(raw["header"],      REQUIRED_COLS_HEADER,      SHEET_HEADER)
    _validate_required_cols(raw["sale_lines"],  REQUIRED_COLS_SALE_LINES,  SHEET_SALE_LINES)
    _validate_required_cols(raw["item_master"], REQUIRED_COLS_ITEM_MASTER, SHEET_ITEM_MASTER)

    logger.info("Workbook loaded and validated successfully.")
    return raw


def get_workbook_summary(sheets: dict[str, pd.DataFrame]) -> dict:
    """
    Return a metadata summary dict suitable for the /data/summary API endpoint.

    Parameters
    ----------
    sheets : dict returned by load_workbook()
    """
    sl = sheets.get("sale_lines", pd.DataFrame())
    hdr = sheets.get("header", pd.DataFrame())
    im = sheets.get("item_master", pd.DataFrame())

    # Date range from Sale Lines (the canonical time axis)
    date_col_sl = "Trans. Date"
    if date_col_sl in sl.columns:
        dates = pd.to_datetime(sl[date_col_sl], errors="coerce").dropna()
        date_min = str(dates.min().date()) if len(dates) else "N/A"
        date_max = str(dates.max().date()) if len(dates) else "N/A"
        date_range_days = (dates.max() - dates.min()).days + 1 if len(dates) else 0
    else:
        date_min = date_max = "N/A"
        date_range_days = 0

    # Store count
    store_col = "Store No."
    stores = list(sorted(sl[store_col].dropna().unique())) if store_col in sl.columns else []

    # Item count
    item_col = "Item No."
    items = int(sl[item_col].nunique()) if item_col in sl.columns else 0

    # Category count (from Item Master)
    cat_col = "Item Category Code"
    categories_in_master = int(im[cat_col].nunique()) if cat_col in im.columns else 0

    # UOM distribution
    uom_col = "Unit of Measure"
    uom_dist = (
        sl[uom_col].value_counts().to_dict()
        if uom_col in sl.columns
        else {}
    )

    # Net Amount sign distribution in Sale Lines
    amt_col = "Net Amount"
    if amt_col in sl.columns:
        amt = pd.to_numeric(sl[amt_col], errors="coerce")
        sign_dist = {
            "negative_count": int((amt < 0).sum()),
            "positive_count": int((amt > 0).sum()),
            "zero_count": int((amt == 0).sum()),
            "null_count": int(amt.isna().sum()),
        }
    else:
        sign_dist = {}

    return {
        "workbook_summary": {
            "header_rows": len(hdr),
            "sale_lines_rows": len(sl),
            "item_master_rows": len(im),
        },
        "date_range": {
            "min": date_min,
            "max": date_max,
            "calendar_days": date_range_days,
        },
        "stores": {
            "count": len(stores),
            "ids": stores,
        },
        "items": {
            "sold_count": items,
            "master_count": len(im),
        },
        "categories": {
            "count_in_master": categories_in_master,
        },
        "uom_distribution": uom_dist,
        "net_amount_sign_distribution": sign_dist,
    }


# ── flat data support (CSV / JSON / XML / API / DB) ─────────────────────────

def validate_flat_schema(
    df: pd.DataFrame,
) -> dict:
    """
    Validate that a flat DataFrame has the required columns for the pipeline.

    Normalises column names to lowercase, applies COLUMN_ALIASES, and checks
    that REQUIRED_FLAT_COLS are present.

    Returns
    -------
    dict with keys: valid, mapped_columns, missing, warnings, sample_rows
    """
    # Normalise column names to lowercase
    df.columns = [str(c).strip().lower() for c in df.columns]

    # Apply aliases
    mapped = {}
    for col in df.columns:
        if col in COLUMN_ALIASES:
            mapped[col] = COLUMN_ALIASES[col]
    df.rename(columns=mapped, inplace=True)

    # Check required columns
    present = set(df.columns)
    missing = [c for c in REQUIRED_FLAT_COLS if c not in present]
    optional_found = [c for c in OPTIONAL_FLAT_COLS if c in present]

    warnings = []
    if missing:
        warnings.append(f"Missing required columns: {missing}")

    # Sample rows for preview
    sample = df.head(5).to_dict(orient="records")

    return {
        "valid": len(missing) == 0,
        "mapped_columns": mapped,
        "columns": list(df.columns),
        "missing": missing,
        "optional_found": optional_found,
        "warnings": warnings,
        "row_count": len(df),
        "sample_rows": sample,
    }


def load_flat_data(
    source: Union[str, Path, pd.DataFrame],
    fmt: str = "csv",
) -> dict[str, pd.DataFrame]:
    """
    Load flat data (CSV / JSON / XML or a DataFrame) and construct synthetic
    sale_lines / header / item_master DataFrames matching the load_workbook()
    contract.

    Parameters
    ----------
    source : path to file, or a pre-loaded DataFrame (from API / DB)
    fmt    : "csv", "json", "xml", or "dataframe"

    Returns
    -------
    dict with keys: 'header', 'sale_lines', 'item_master'
    """
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    else:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Flat data file not found: {path}")
        if fmt == "csv":
            df = pd.read_csv(path)
        elif fmt == "json":
            df = pd.read_json(path)
        elif fmt == "xml":
            df = pd.read_xml(path)
        else:
            raise ValueError(f"Unsupported format: {fmt}")

    # Normalise column names and apply aliases
    df.columns = [str(c).strip().lower() for c in df.columns]
    alias_map = {c: COLUMN_ALIASES[c] for c in df.columns if c in COLUMN_ALIASES}
    df.rename(columns=alias_map, inplace=True)

    # Verify required columns
    missing = [c for c in REQUIRED_FLAT_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Flat data missing required columns: {missing}")

    # Fill optional columns with defaults
    if "receipt_no" not in df.columns:
        df["receipt_no"] = [f"FLAT-{i}" for i in range(len(df))]
    if "category" not in df.columns:
        df["category"] = "UNKNOWN"
    if "uom" not in df.columns:
        df["uom"] = "NOS"
    if "price" not in df.columns:
        df["price"] = 0.0

    # Build synthetic Sale Lines (raw column names matching the workbook)
    sale_lines = pd.DataFrame({
        "Store No.": df["store_id"],
        "Receipt No.": df["receipt_no"],
        "Item No.": df["item_id"],
        "Price": pd.to_numeric(df["price"], errors="coerce").fillna(0.0),
        "Quantity": pd.to_numeric(df["quantity"], errors="coerce").fillna(0.0),
        "Net Amount": pd.to_numeric(df["net_amount"], errors="coerce").fillna(0.0),
        "Trans. Date": pd.to_datetime(df["date"], errors="coerce"),
        "Unit of Measure": df["uom"],
    })

    # Build synthetic Header (one row per unique receipt + store + date)
    hdr_df = sale_lines[["Receipt No.", "Store No.", "Trans. Date", "Net Amount"]].copy()
    hdr_df = hdr_df.rename(columns={"Trans. Date": "Date"})
    hdr_agg = hdr_df.groupby(["Receipt No.", "Store No."], as_index=False).agg({
        "Date": "first",
        "Net Amount": "sum",
    })
    hdr_agg["Gross Amount"] = hdr_agg["Net Amount"].abs()
    header = hdr_agg

    # Build synthetic Item Master
    items = df[["item_id"]].drop_duplicates().copy()
    items.rename(columns={"item_id": "No."}, inplace=True)
    items["Description"] = "Imported item"
    if "category" in df.columns:
        cat_map = df.drop_duplicates(subset=["item_id"]).set_index("item_id")["category"]
        items["Item Category Code"] = items["No."].map(cat_map).fillna("UNKNOWN")
    else:
        items["Item Category Code"] = "UNKNOWN"

    logger.info(
        "load_flat_data: sale_lines=%d, header=%d, item_master=%d",
        len(sale_lines), len(header), len(items),
    )

    return {
        "header": header,
        "sale_lines": sale_lines,
        "item_master": items,
    }

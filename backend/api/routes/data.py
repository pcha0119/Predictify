"""Routes: /data/summary, /data/preview, /data/transform, /data/save, /data/prep/reset"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import REPORT_DIR, IMPORTED_FLAT_PATH, WORKBOOK_PATH, WORKBOOK_PATH_RAW
from api.state import _state, _prep

router = APIRouter(tags=["data"])


# ── Data prep helpers (local — only used by this module) ──────────────────────

def _load_prep_data() -> pd.DataFrame:
    """Load imported data into _prep['df'] if not already loaded."""
    if _prep["df"] is not None:
        return _prep["df"]

    if IMPORTED_FLAT_PATH.exists():
        _prep["df"] = pd.read_csv(IMPORTED_FLAT_PATH)
        _prep["source"] = "flat_csv"
    elif WORKBOOK_PATH.exists() or WORKBOOK_PATH_RAW.exists():
        from data_ingestion.loader import load_workbook
        path = WORKBOOK_PATH if WORKBOOK_PATH.exists() else WORKBOOK_PATH_RAW
        sheets = load_workbook(path)
        _prep["df"] = sheets["sale_lines"]
        _prep["source"] = "excel"
    else:
        raise HTTPException(404, "No data imported yet. Upload a file first.")

    _prep["history"] = []
    return _prep["df"]


def _column_profile(df: pd.DataFrame) -> list[dict]:
    """Build Tableau Prep-style column profiles."""
    profiles = []
    for col in df.columns:
        s = df[col]
        null_count = int(s.isna().sum())
        total = len(s)

        profile: dict[str, Any] = {
            "name": col,
            "dtype": str(s.dtype),
            "null_count": null_count,
            "null_pct": round(null_count / total * 100, 1) if total else 0,
            "unique_count": int(s.nunique()),
            "total_count": total,
        }

        if pd.api.types.is_numeric_dtype(s):
            clean = s.dropna()
            profile["kind"] = "numeric"
            profile["min"] = round(float(clean.min()), 2) if len(clean) else None
            profile["max"] = round(float(clean.max()), 2) if len(clean) else None
            profile["mean"] = round(float(clean.mean()), 2) if len(clean) else None
            profile["median"] = round(float(clean.median()), 2) if len(clean) else None
            if len(clean) > 0:
                import numpy as np
                counts, edges = np.histogram(clean, bins=min(10, len(clean.unique())))
                profile["histogram"] = {
                    "counts": counts.tolist(),
                    "edges": [round(float(e), 2) for e in edges],
                }
        elif pd.api.types.is_datetime64_any_dtype(s):
            clean = s.dropna()
            profile["kind"] = "datetime"
            profile["min"] = str(clean.min()) if len(clean) else None
            profile["max"] = str(clean.max()) if len(clean) else None
        else:
            profile["kind"] = "text"
            vc = s.value_counts().head(8)
            profile["top_values"] = [{"value": str(v), "count": int(c)} for v, c in vc.items()]

        profiles.append(profile)
    return profiles


def _push_undo():
    """Save current df state for undo."""
    if _prep["df"] is not None:
        _prep["history"].append(_prep["df"].copy())
        if len(_prep["history"]) > _prep["max_undo"]:
            _prep["history"].pop(0)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/data/summary")
async def data_summary() -> dict:
    """Return workbook metadata. 404 if pipeline has not run yet."""
    if _state["summary"]:
        return _state["summary"]

    summary_path = REPORT_DIR / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            return json.load(f)

    raise HTTPException(404, "Summary not available. Run the pipeline first.")


@router.get("/data/preview")
async def data_preview(
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=10, le=500),
    sort_col: str | None = Query(None),
    sort_asc: bool = Query(True),
) -> dict:
    """Return paginated data preview + column profiles."""
    df = _load_prep_data()

    if sort_col and sort_col in df.columns:
        df = df.sort_values(sort_col, ascending=sort_asc, na_position="last")

    total_rows = len(df)
    total_pages = max(1, (total_rows + page_size - 1) // page_size)
    start = (page - 1) * page_size
    end = start + page_size
    page_df = df.iloc[start:end]

    return {
        "columns": list(df.columns),
        "rows": page_df.fillna("").to_dict(orient="records"),
        "row_indices": list(range(start, min(end, total_rows))),
        "total_rows": total_rows,
        "total_pages": total_pages,
        "page": page,
        "page_size": page_size,
        "profiles": _column_profile(df),
        "can_undo": len(_prep["history"]) > 0,
    }


@router.post("/data/transform")
async def data_transform(body: dict) -> dict:
    """
    Apply a data transformation and return updated status.

    Supported operations (body.operation):
      rename_column, remove_column, change_type, fill_nulls, filter_rows,
      remove_duplicates, replace_values, sort, edit_cell, add_column, undo
    """
    df = _load_prep_data()
    op = body.get("operation", "")

    if op == "undo":
        if _prep["history"]:
            _prep["df"] = _prep["history"].pop()
            return {"status": "ok", "operation": "undo", "message": "Reverted to previous state."}
        else:
            raise HTTPException(400, "Nothing to undo.")

    _push_undo()

    try:
        if op == "rename_column":
            old, new = body["column"], body["new_name"]
            if old not in df.columns:
                raise HTTPException(400, f"Column '{old}' not found.")
            if new in df.columns:
                raise HTTPException(400, f"Column '{new}' already exists.")
            df.rename(columns={old: new}, inplace=True)

        elif op == "remove_column":
            col = body["column"]
            if col not in df.columns:
                raise HTTPException(400, f"Column '{col}' not found.")
            df.drop(columns=[col], inplace=True)

        elif op == "change_type":
            col, new_type = body["column"], body["new_type"]
            if col not in df.columns:
                raise HTTPException(400, f"Column '{col}' not found.")
            if new_type == "text":
                df[col] = df[col].astype(str).replace("nan", "")
            elif new_type == "number":
                df[col] = pd.to_numeric(df[col], errors="coerce")
            elif new_type == "date":
                df[col] = pd.to_datetime(df[col], errors="coerce")
            else:
                raise HTTPException(400, f"Unknown type '{new_type}'. Use: text, number, date.")

        elif op == "fill_nulls":
            col, strategy = body["column"], body.get("strategy", "zero")
            if col not in df.columns:
                raise HTTPException(400, f"Column '{col}' not found.")
            if strategy == "zero":
                df[col] = df[col].fillna(0)
            elif strategy == "mean":
                df[col] = df[col].fillna(df[col].mean())
            elif strategy == "median":
                df[col] = df[col].fillna(df[col].median())
            elif strategy == "mode":
                mode_val = df[col].mode()
                df[col] = df[col].fillna(mode_val.iloc[0] if len(mode_val) else "")
            elif strategy == "empty_string":
                df[col] = df[col].fillna("")
            elif strategy == "custom":
                df[col] = df[col].fillna(body.get("value", ""))
            else:
                raise HTTPException(400, f"Unknown strategy '{strategy}'.")

        elif op == "filter_rows":
            col, operator, val = body["column"], body["operator"], body.get("value", "")
            if col not in df.columns:
                raise HTTPException(400, f"Column '{col}' not found.")
            s = df[col]
            ops = {
                "eq":           lambda: s.astype(str) == str(val),
                "neq":          lambda: s.astype(str) != str(val),
                "gt":           lambda: pd.to_numeric(s, errors="coerce") > float(val),
                "lt":           lambda: pd.to_numeric(s, errors="coerce") < float(val),
                "gte":          lambda: pd.to_numeric(s, errors="coerce") >= float(val),
                "lte":          lambda: pd.to_numeric(s, errors="coerce") <= float(val),
                "contains":     lambda: s.astype(str).str.contains(str(val), case=False, na=False),
                "not_contains": lambda: ~s.astype(str).str.contains(str(val), case=False, na=False),
                "starts_with":  lambda: s.astype(str).str.startswith(str(val), na=False),
                "ends_with":    lambda: s.astype(str).str.endswith(str(val), na=False),
                "is_null":      lambda: s.isna() | (s.astype(str).str.strip() == "") | (s.astype(str) == "nan"),
                "is_not_null":  lambda: s.notna() & (s.astype(str).str.strip() != "") & (s.astype(str) != "nan"),
            }
            if operator not in ops:
                raise HTTPException(400, f"Unknown operator '{operator}'.")
            mask = ops[operator]()
            df = df[mask].reset_index(drop=True)

        elif op == "remove_duplicates":
            cols = body.get("columns", list(df.columns))
            df = df.drop_duplicates(subset=cols, keep="first").reset_index(drop=True)

        elif op == "replace_values":
            col = body["column"]
            if col not in df.columns:
                raise HTTPException(400, f"Column '{col}' not found.")
            df[col] = df[col].astype(str).str.replace(str(body["find"]), str(body["replace_with"]), regex=False)

        elif op == "sort":
            col = body["column"]
            if col not in df.columns:
                raise HTTPException(400, f"Column '{col}' not found.")
            df = df.sort_values(col, ascending=body.get("ascending", True), na_position="last").reset_index(drop=True)

        elif op == "edit_cell":
            row_idx, col, val = body["row_index"], body["column"], body["value"]
            if col not in df.columns:
                raise HTTPException(400, f"Column '{col}' not found.")
            if row_idx < 0 or row_idx >= len(df):
                raise HTTPException(400, f"Row index {row_idx} out of range.")
            df.at[row_idx, col] = val

        elif op == "add_column":
            name = body["name"]
            if name in df.columns:
                raise HTTPException(400, f"Column '{name}' already exists.")
            df[name] = body.get("default_value", "")

        else:
            if _prep["history"]:
                _prep["history"].pop()
            raise HTTPException(400, f"Unknown operation '{op}'.")

        _prep["df"] = df

    except HTTPException:
        raise
    except Exception as exc:
        if _prep["history"]:
            _prep["df"] = _prep["history"].pop()
        raise HTTPException(400, f"Transform failed: {exc}")

    return {
        "status": "ok",
        "operation": op,
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "message": f"Applied '{op}' successfully.",
    }


@router.post("/data/save")
async def data_save() -> dict:
    """Persist the current prep state back to imported_flat.csv."""
    if _prep["df"] is None:
        raise HTTPException(400, "No data loaded in prep editor.")

    df = _prep["df"]
    IMPORTED_FLAT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(IMPORTED_FLAT_PATH, index=False)

    return {
        "status": "saved",
        "row_count": len(df),
        "columns": list(df.columns),
        "saved_to": str(IMPORTED_FLAT_PATH),
        "message": f"Data saved ({len(df)} rows, {len(df.columns)} columns).",
    }


@router.post("/data/prep/reset")
async def data_prep_reset() -> dict:
    """Clear the in-memory prep state, forcing a reload from file on next access."""
    _prep["df"] = None
    _prep["history"] = []
    _prep["source"] = None
    return {"status": "reset", "message": "Prep state cleared."}

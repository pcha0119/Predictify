"""
Shared utility functions for route handlers.
"""

import json
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import HTTPException

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import REPORT_DIR


def _save_json(data: Any, path: Path) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def _load_csv_or_404(path: Path) -> pd.DataFrame:
    """Load a CSV artifact; raise 404 if not yet created by the pipeline."""
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Artifact not found: {path.name}. Run the pipeline first.",
        )
    return pd.read_csv(path, parse_dates=["date"] if "date" in pd.read_csv(path, nrows=0).columns else [])


def _read_fc_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"])


def _read_act_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"])

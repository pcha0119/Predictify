"""
Shared in-memory state for the single-user local demo.

Imported by pipeline_runner.py and all route modules that need to
read or update pipeline/prep status.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Any

# ── Pipeline state ────────────────────────────────────────────────────────────
_state: dict[str, Any] = {
    "status": "idle",        # idle | running | complete | error
    "job_id": None,
    "started_at": None,
    "finished_at": None,
    "error": None,
    "last_run_artifacts": {},
    "summary": None,
}

_executor = ThreadPoolExecutor(max_workers=1)

# ── Data prep state ───────────────────────────────────────────────────────────
_prep: dict[str, Any] = {
    "df": None,           # current DataFrame being prepped
    "history": [],        # list of previous df states for undo
    "source": None,       # "excel" or "flat_csv"
    "max_undo": 20,
}

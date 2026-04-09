"""Routes: /health, /pipeline/run, /pipeline/status"""

from __future__ import annotations

import asyncio
import uuid

from fastapi import APIRouter

from api.state import _state, _executor
from api.pipeline_runner import _run_pipeline_sync

router = APIRouter(tags=["pipeline"])


@router.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "version": "1.0.0",
        "pipeline_status": _state["status"],
    }


@router.post("/pipeline/run")
async def run_pipeline() -> dict:
    """Trigger the full pipeline asynchronously."""
    if _state["status"] == "running":
        return {"status": "already_running", "job_id": _state["job_id"]}

    job_id = str(uuid.uuid4())
    _state["status"] = "running"
    _state["job_id"] = job_id

    loop = asyncio.get_event_loop()
    loop.run_in_executor(_executor, _run_pipeline_sync)

    return {"status": "started", "job_id": job_id}


@router.get("/pipeline/status")
async def pipeline_status() -> dict:
    return {
        "status": _state["status"],
        "job_id": _state["job_id"],
        "started_at":  _state["started_at"],
        "finished_at": _state["finished_at"],
        "error": _state["error"],
    }

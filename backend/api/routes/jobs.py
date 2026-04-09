"""Routes: /jobs/save, /jobs/list, /jobs/load/{job_id}, DELETE /jobs/{job_id}"""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from api.state import _prep
from services.job_persistence import save_job, load_job, list_jobs, delete_job

router = APIRouter(tags=["jobs"])


@router.post("/jobs/save")
async def save_current_job(job_name: str = "untitled") -> dict:
    """Save the current prep state (DataFrame) to the database for later resumption."""
    if _prep["df"] is None:
        raise HTTPException(400, "No data loaded to save as a job.")

    job_id = str(uuid.uuid4())[:8]
    filename = job_name or "job"
    df = _prep["df"]

    success = save_job(
        job_id=job_id,
        filename=filename,
        stage="prepped",
        df=df,
        rows=len(df),
        cols=len(df.columns),
        metadata={"source": _prep["source"], "columns": list(df.columns)},
    )

    if success:
        return {
            "status": "saved",
            "job_id": job_id,
            "filename": filename,
            "rows": len(df),
            "cols": len(df.columns),
            "message": f"Job '{filename}' saved successfully.",
        }
    else:
        raise HTTPException(500, "Failed to save job to database.")


@router.get("/jobs/list")
async def list_saved_jobs() -> dict:
    jobs = list_jobs()
    return {"status": "success", "count": len(jobs), "jobs": jobs}


@router.post("/jobs/load/{job_id}")
async def load_saved_job(job_id: str) -> dict:
    """Load a previously saved job from the database and restore it to prep state."""
    job = load_job(job_id)
    if not job:
        raise HTTPException(404, f"Job '{job_id}' not found.")

    _prep["df"] = job["df"]
    _prep["source"] = job["metadata"].get("source", "unknown")

    return {
        "status": "loaded",
        "job_id": job_id,
        "filename": job["filename"],
        "stage": job["stage"],
        "rows": job["rows"],
        "cols": job["cols"],
        "message": f"Job '{job['filename']}' loaded successfully.",
    }


@router.delete("/jobs/{job_id}")
async def delete_saved_job(job_id: str) -> dict:
    if delete_job(job_id):
        return {"status": "deleted", "job_id": job_id}
    else:
        raise HTTPException(500, "Failed to delete job.")

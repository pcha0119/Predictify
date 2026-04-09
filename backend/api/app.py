"""
FastAPI application — router wiring only.

All route logic lives in api/routes/.
Shared state lives in api/state.py.
Pipeline runner lives in api/pipeline_runner.py.

Endpoints
---------
GET  /health                       pipeline.py
POST /pipeline/run                 pipeline.py
GET  /pipeline/status              pipeline.py
GET  /data/summary                 data.py
GET  /data/preview                 data.py
POST /data/transform               data.py
POST /data/save                    data.py
POST /data/prep/reset              data.py
GET  /stores                       catalog.py
GET  /items                        catalog.py
GET  /items/forecast_summary       catalog.py
GET  /categories                   catalog.py
GET  /forecasts/{grain}            forecasts.py
GET  /forecasts/{grain}/export     forecasts.py
GET  /evaluation/metrics           forecasts.py
POST /upload                       upload.py
GET  /import/status                upload.py
POST /connect/api                  connectors.py
POST /connect/database             connectors.py
POST /jobs/save                    jobs.py
GET  /jobs/list                    jobs.py
POST /jobs/load/{job_id}           jobs.py
DELETE /jobs/{job_id}              jobs.py
POST /chat                         llm.py
GET  /llm/config                   llm.py
POST /llm/config                   llm.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make backend/ importable from anywhere uvicorn is invoked
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.routes import pipeline, catalog, forecasts, data, upload, connectors, jobs, llm

app = FastAPI(
    title="Sales Forecasting API",
    version="1.0.0",
    description="Retail sales forecasting pipeline — Almond House",
)

# ── CORS (open for local demo — restrict in production) ───────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Serve the UI ──────────────────────────────────────────────────────────────
_repo_root = Path(__file__).resolve().parents[2]
for _ui_dir in [_repo_root / "frontend", Path(__file__).resolve().parent.parent / "ui"]:
    if _ui_dir.exists():
        app.mount("/ui", StaticFiles(directory=str(_ui_dir), html=True), name="ui")
        break

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(pipeline.router)
app.include_router(catalog.router)
app.include_router(forecasts.router)
app.include_router(data.router)
app.include_router(upload.router)
app.include_router(connectors.router)
app.include_router(jobs.router)
app.include_router(llm.router)

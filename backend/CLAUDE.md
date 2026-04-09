# backend/CLAUDE.md

Backend-specific context for the Predictify forecasting pipeline.

## Critical gotchas

**No Parquet — CSV only**: pyarrow has no Python 3.14 wheels. All artifact writes use CSV. Do not introduce Parquet.

**Sign convention in raw data**: negative `net_amount` = genuine sale; positive = return/reversal. `cleaner.py` handles this before any aggregation. Do not re-correct downstream.

**Workbook path resolution**: loader tries `WORKBOOK_PATH` (uploaded copy) first, then `WORKBOOK_PATH_RAW` (`data/raw/AI Forecasting Data.xlsx`). The demo workbook covers only ~36 days.

**Item Master sheet**: has extra header rows; skip logic is in `loader.py`. Don't remove it.

## Key design rules

**No future leakage**: lag features (1, 2, 3, 7, 14, 28 days) must reference only past rows. Feature computation is group-scoped (per store / category / item) — never compute across groups.

**Sparse series fallback**: items with fewer than `MIN_ITEM_DAYS` (10) distinct sales days skip item-level training and fall back through `SPARSE_FALLBACK_ORDER`. Intentional — don't paper over it.

**LLM tools call functions directly**: `llm_gateway.py` invokes FastAPI endpoint functions in-process, not via HTTP. Read-only tools auto-execute; `run_pipeline` and `upload_data` require user confirmation.

## Where things live

| What | Where |
|------|-------|
| All config values | `config.py` — edit here first |
| Shared API state | `api/state.py` (`_state`, `_prep`, `_executor`) |
| Shared API helpers | `api/helpers.py` (`_save_json`, `_load_csv_or_404`, etc.) |
| Pipeline sync runner | `api/pipeline_runner.py` |
| Route handlers | `api/routes/` — one file per domain |
| Job persistence (SQLite) | `services/job_persistence.py` → `artifacts/jobs.db` |
| LLM provider config | `llm_config.yaml` |
| Cleaning + fact tables | `data_cleaning/cleaner.py` |
| Feature engineering | `feature_engineering/features.py` |
| Model base class + walk-forward | `models/forecaster.py` |

## API routes map

| File | Endpoints |
|------|-----------|
| `routes/pipeline.py` | /health, /pipeline/run, /pipeline/status |
| `routes/catalog.py` | /stores, /items, /items/forecast_summary, /categories |
| `routes/forecasts.py` | /forecasts/{grain}, /forecasts/{grain}/export, /evaluation/metrics |
| `routes/data.py` | /data/summary, /data/preview, /data/transform, /data/save, /data/prep/reset |
| `routes/upload.py` | /upload, /import/status |
| `routes/connectors.py` | /connect/api, /connect/database |
| `routes/jobs.py` | /jobs/save, /jobs/list, /jobs/load/{job_id}, DELETE /jobs/{job_id} |
| `routes/llm.py` | /chat, /llm/config (GET + POST) |

## Artifacts layout

```
artifacts/
  data/       cleaned daily fact CSVs
  forecasts/  one CSV per grain × model (includes CI columns)
  models/     pickled sklearn pipelines
  reports/    metrics_all.json, summary.json
```

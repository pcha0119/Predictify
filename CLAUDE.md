# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Predictify — retail sales forecasting platform (Almond House, India). Multi-grain forecasts (total / store / category / item) via Ridge, Lasso, ElasticNet, and Naive baselines, served through a FastAPI backend and single-page dashboard.

## Commands

```bash
# Install
pip install -r backend/requirements.txt          # Python 3.11+

# Run API only
python -m uvicorn backend.api.app:app --host 127.0.0.1 --port 8000

# Run full pipeline then serve
cd backend && python run_pipeline.py --grain total store category --horizon 7 14 30 --serve

# Pipeline flags
# --skip-training    re-forecast from saved model artifacts
# --bootstrap N      uncertainty replicates (default 50)
```

Access: UI → http://127.0.0.1:8000/ui/ | Docs → /docs | Health → /health

## Data flow

```
data/raw/AI Forecasting Data.xlsx  (source — never overwritten)
  → loader.py          # validate, rename columns
  → cleaner.py         # sign-correct, normalise UOM, quarantine bad rows
  → 5 daily fact tables (total / store / category / item / store×category)
  → features.py        # calendar + festival flags + lag/rolling
  → forecaster.py      # walk-forward CV + train
  → artifacts/         # CSVs, pickled models, metrics JSON
  → app.py             # FastAPI router wiring → api/routes/*
```

All configuration (paths, hyperparams, thresholds) lives in `backend/config.py`. **Change values there, not inline.**

## Development workflow

Spec-driven. For every feature:
1. Ask all clarifying questions first — spec is approved before any code is written
2. Plan with Opus 4.6, execute with Sonnet
3. **After every code change, validate the build** — see @.claude/rules/build.md
4. On user go-green: commit + push to git
5. Run `/compact` between features to keep context lean

## Subdirectory context

@backend/CLAUDE.md
@frontend/CLAUDE.md

## Rules

@.claude/rules/build.md

## Updating CLAUDE.md and rules files

Ask permission before modifying. Codify only after something has bitten us — not speculatively. Claude updates rules files as new patterns are confirmed. Audit and trim stale entries periodically.

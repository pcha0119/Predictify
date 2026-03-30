# Predictify
Sales forecasting web app with a FastAPI backend and a static frontend.

## Project layout
- `backend/` - API, pipeline, models, ingestion, artifacts
- `frontend/` - UI (`index.html`)

## Run after cloning
### 1) Clone
```bash
git clone https://github.com/pcha0119/Predictify.git
cd Predictify
```

### 2) Create and activate virtual environment
Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3) Install dependencies
```bash
pip install -r backend/requirements.txt
```

### 4) Run the full project
From repo root:
```bash
python -m uvicorn backend.api.app:app --host 127.0.0.1 --port 8000
```

Open:
- UI: `http://127.0.0.1:8000/ui/index.html`
- API health: `http://127.0.0.1:8000/health`

The frontend is served by the backend at `/ui`, so one command starts the entire project.

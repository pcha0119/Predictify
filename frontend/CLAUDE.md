# frontend/CLAUDE.md

Frontend context for Predictify.

## Structure

Everything is `index.html` — a single self-contained file with embedded CSS and JS. No build step, no package manager, no bundler. Dependencies (Plotly, fonts) are loaded from CDN.

The file is served statically by FastAPI at `/ui/`. It talks to the backend via `fetch()` calls to `/api/*` endpoints.

## Editing

Edit `index.html` directly. If the file grows beyond ~3000 lines, raise it before splitting — the single-file approach is a deliberate demo choice.

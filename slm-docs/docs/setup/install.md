---
sidebar_position: 1
title: Install + boot
---

# Install + boot

BrewSLM runs as **two services**: a FastAPI backend on port 8000 and a Vite/React frontend on port 5173. Docs are a third (this Docusaurus site, port 3001). All three are optional but recommended for the first run.

## Prerequisites

- **Python 3.11+**. `python --version` should report 3.11 or newer.
- **Node 20+**. Both frontend (`frontend/`) and docs (`slm-docs/`) require it.
- **(Optional) NVIDIA GPU + drivers** if you plan to run real training rather than the built-in simulator. CPU works for evaluation, smoke tests, and small SFT runs on tiny models.

## Clone + install

```sh
git clone https://github.com/<you>/__SLM__.git brewslm
cd brewslm

# Backend
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Frontend
cd ../frontend
npm install

# Docs (optional but recommended)
cd ../slm-docs
npm install
```

## Database

SQLite is the default — no setup required. The first boot creates `backend/data/brewslm.db` automatically.

For Postgres, set:

```sh
export DATABASE_URL="postgresql+asyncpg://user:pass@host:5432/brewslm"
```

…before starting the backend. See [Environment](environment.md) for the full list of variables.

## Run all three services

Open three terminals from the repo root.

### Backend

```sh
cd backend
source .venv/bin/activate
uvicorn app.main:app --reload --port 8000
```

You should see `Uvicorn running on http://127.0.0.1:8000`. The Swagger UI lives at `http://localhost:8000/api/docs`.

### Frontend

```sh
cd frontend
npm run dev
```

The app boots at `http://localhost:5173`. The TopBar's `?` icon links to the docs at `http://localhost:3001/...` (configurable via `VITE_DOCS_URL`).

### Docs (this site)

```sh
cd slm-docs
npm run start
```

The Docusaurus dev server serves `http://localhost:3001/docs/...` with hot reload.

## First login

If `AUTH_ENABLED` is `false` (default in dev), there's no login screen — you land on the project list. Otherwise see [Auth + SSO](auth-and-sso.md).

## Boot smoke test

A 30-second sanity check from the terminal:

```sh
# Backend responds
curl http://localhost:8000/api/health

# CLI can hit the API
brewslm doctor --project 0 || true  # works even with no project; reports readiness
```

If both succeed, you're ready for the [Quickstart](../getting-started/quickstart.md).

## Common install pitfalls

- **`alembic.command.upgrade` error on boot** — your local SQLite DB is older than the current migration head. Either delete `backend/data/brewslm.db` and let the auto-create path rebuild, or run `alembic -c backend/alembic.ini upgrade head`. See [Common blockers](../reliability/common-blockers.md).
- **`ModuleNotFoundError: app`** when running CLI — you didn't activate the backend venv. CLI imports the backend services.
- **CORS errors in the browser** — frontend is calling a different `VITE_API_BASE` than the backend is serving from. Set `VITE_API_BASE=http://localhost:8000/api` in `frontend/.env.local` if needed.
- **Geist font flicker on first paint** — the docs fetch Geist from Google Fonts on each cold load. Disable network throttling or self-host the font under `slm-docs/static/`.

## Next

- [Auth + SSO](auth-and-sso.md) — Local, API key, and SSO auth modes.
- [Environment](environment.md) — every `export VAR=…` that changes behavior.
- [Quickstart](../getting-started/quickstart.md) — 10-minute hello world.

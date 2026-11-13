---
sidebar_position: 1
title: Install + boot
---

# Install + boot

BrewSLM runs as **two services**: a FastAPI backend on port 8000 and a Vite/React frontend on port 5173. Docs are a third (this Docusaurus site, port 3001). All three are optional but recommended for the first run.

## Prerequisites

- **Python 3.11+**. `python3 --version` should report 3.11 or newer (on Ubuntu the binary is `python3`).
- **Node 20.19+ or 22.12+**. The frontend (`frontend/`, Vite 7) requires one of these; the docs (`slm-docs/`) need Node 20+.
- **Redis plus a Celery worker** for real training: the default `TRAINING_BACKEND=external` dispatches training through Celery (`cd backend && celery -A app.worker.celery_app worker --loglevel=INFO --pool=threads --concurrency=2`), and without a worker a run stays queued. `TRAINING_BACKEND=simulate` with `ALLOW_SIMULATED_TRAINING=true` (set both in `backend/.env`; neither is the default) needs neither, and trains no weights: a loss curve only, no evaluation.
- **(Optional) NVIDIA GPU + drivers** if you plan to run real training rather than the built-in simulator. CPU works for evaluation, smoke tests, and small SFT runs on tiny models.

## Clone + install

```sh
git clone https://github.com/TensorGreed/__SLM__.git brewslm
cd brewslm

# Backend
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # the backend reads backend/.env; without it `admin` has no project access

# No GPU / no Redis? Simulated training (loss curve only, not evaluated):
# sed -i 's/^TRAINING_BACKEND=.*/TRAINING_BACKEND="simulate"/; s/^ALLOW_SIMULATED_TRAINING=.*/ALLOW_SIMULATED_TRAINING=true/' .env

# Frontend
cd ../frontend
npm ci

# Docs (optional but recommended)
cd ../slm-docs
npm install
```

## Database

SQLite is the default (`DATABASE_URL="sqlite+aiosqlite:///./data/slm_platform.db"`, a relative path resolves against the repo root, so the file is `data/slm_platform.db` there). The first boot creates the tables (`ALLOW_SQLITE_AUTOCREATE=true`). Then stop the backend and run `cd backend && alembic stamp head` once, so later `alembic upgrade head` runs apply only new migrations. Do not run `upgrade head` on a never-stamped SQLite DB: some early migrations use `ALTER COLUMN`, which SQLite refuses.

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

You should see `Uvicorn running on http://127.0.0.1:8000`. The Swagger UI lives at `http://localhost:8000/docs`.

### Frontend

```sh
cd frontend
npm run dev
```

The app boots at `http://localhost:5173`. The TopBar's `?` icon links to the docs at `http://localhost:3001/...` (configurable via `VITE_DOCS_URL`).

Other ports: the UI calls `/api` on its own origin and Vite proxies it to `VITE_API_PROXY_TARGET` (default `http://localhost:8000`). For a backend on 8100 and the UI on 5180: `uvicorn app.main:app --reload --port 8100`, then `VITE_API_PROXY_TARGET=http://localhost:8100 npm run dev -- --port 5180 --strictPort` (without `--strictPort`, Vite silently moves to the next free port when the one you asked for is taken).

### Docs (this site)

```sh
cd slm-docs
npm run start
```

The Docusaurus dev server serves `http://localhost:3001/docs/...` with hot reload.

## First login

Auth is on by default (`AUTH_ENABLED=true`): sign in as `admin` with the `API_KEY` from `backend/.env` (default `sk-mock-admin-key`). `admin` has admin rights only when `AUTH_BOOTSTRAP_API_KEY` is set (`backend/.env.example` sets it); any other new username is created as an engineer with no project access. With `AUTH_ENABLED=false` there's no login screen — you land on the project list. See [Auth + SSO](auth-and-sso.md).

## Boot smoke test

A 30-second sanity check from the terminal:

```sh
# Backend responds (no auth needed)
curl http://localhost:8000/api/health

# CLI can hit the API: log in once (auth is on), then any command works.
# ./brewslm is at the repo root; it is not on your PATH.
./brewslm auth login --username admin --password sk-mock-admin-key --save
./brewslm auth whoami
# Backend on another port: BREWSLM_API_BASE=http://127.0.0.1:8100 ./brewslm auth whoami
```

If both succeed, you're ready for the [Quickstart](../getting-started/quickstart.md).

## Common install pitfalls

- **`alembic.command.upgrade` error on boot** — your local SQLite DB is older than the current migration head. If the DB was never stamped (it was created by the auto-create path), run `cd backend && alembic stamp head` instead of `upgrade head`; otherwise run `cd backend && alembic upgrade head`. Or move `data/slm_platform.db` aside and let the auto-create path rebuild it (this loses its data). See [Common blockers](../reliability/common-blockers.md).
- **`ModuleNotFoundError: app`** when running CLI — run `./brewslm` from the repo root (it uses `backend/.venv`), or activate the backend venv.
- **Every request fails in the UI** — the Vite dev proxy points at a different backend. Set `VITE_API_PROXY_TARGET=http://localhost:<backend port>` when you start `npm run dev` (the client always calls `/api` on its own origin, so there is no CORS to configure).
- **Geist font flicker on first paint** — the docs fetch Geist from Google Fonts on each cold load. Disable network throttling or self-host the font under `slm-docs/static/`.

## Next

- [Auth + SSO](auth-and-sso.md) — Local, API key, and SSO auth modes.
- [Environment](environment.md) — every `export VAR=…` that changes behavior.
- [Quickstart](../getting-started/quickstart.md) — 10-minute hello world.

---
sidebar_position: 2
title: Quickstart
---

# Quickstart

About ten minutes from `git clone` to a completed training run with a loss curve, most of it installs. This path needs no GPU and no Redis: it uses **simulated training** on the bundled **Demo · Support FAQ** starter. For real training, see [Real training](#real-training) at the end.

## Prerequisites

- Python 3.11+ (on Ubuntu the binary is `python3`) · Node 20.19+ or 22.12+ (Vite 7 requires one of these) · `pip`, `npm`.
- Two terminals (backend, frontend); a third for these docs is optional.

## 1. Clone + install

```sh
git clone https://github.com/TensorGreed/__SLM__.git brewslm
cd brewslm/backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

The backend reads `backend/.env` (start it from `backend/`). `.env.example` sets `AUTH_BOOTSTRAP_API_KEY`, which gives the `admin` login admin rights; without a `.env`, `admin` is created as an engineer with **no project access** and you are stuck after login.

### No GPU, no Redis: turn on simulated training

The defaults are **not** simulated: `TRAINING_BACKEND="external"` runs real training through a Celery worker, and `ALLOW_SIMULATED_TRAINING=false`. Without a worker, a run stays queued. For this quickstart set both in `backend/.env`:

```sh
sed -i 's/^TRAINING_BACKEND=.*/TRAINING_BACKEND="simulate"/; s/^ALLOW_SIMULATED_TRAINING=.*/ALLOW_SIMULATED_TRAINING=true/' .env
```

(Or edit the two lines by hand: `TRAINING_BACKEND="simulate"` and `ALLOW_SIMULATED_TRAINING=true`.) Optional: `DEBUG=false` stops the SQL echo that the example's `DEBUG=true` prints.

A simulated run walks the real pipeline and draws a loss curve, but it trains **no weights**. It is not evaluated against the base model, so there is no verdict on **Results** for it.

## 2. Create the database and boot the backend

```sh
# still in backend/, venv active
uvicorn app.main:app --port 8000
# wait for "Application startup complete", then Ctrl-C
alembic stamp head
uvicorn app.main:app --reload --port 8000
```

The first boot creates `data/slm_platform.db` at the repo root and all its tables (`ALLOW_SQLITE_AUTOCREATE=true`). `alembic stamp head` marks that fresh DB as current, so later `alembic upgrade head` runs only apply new migrations. Do not run `alembic upgrade head` on a never-stamped DB: some early migrations use `ALTER COLUMN`, which SQLite refuses.

Check: `curl http://localhost:8000/api/health`. The Swagger UI is at `http://localhost:8000/docs`.

## 3. Boot the frontend

In a second terminal, from the repo root:

```sh
cd frontend
npm ci
npm run dev
```

Open `http://localhost:5173`.

### Other ports

The UI calls `/api` on its own origin and Vite proxies it to `VITE_API_PROXY_TARGET` (default `http://localhost:8000`). To use other ports, for example backend 8100 and UI 5180:

```sh
# backend/
uvicorn app.main:app --reload --port 8100
# frontend/
VITE_API_PROXY_TARGET=http://localhost:8100 npm run dev -- --port 5180 --strictPort
```

Pass `--strictPort`: otherwise, when the port is taken, Vite silently starts on the next one. The browser only ever talks to the Vite origin, so no CORS change is needed; `CORS_ORIGINS` matters only for a page on another origin calling the API directly.

## 4. Log in

Auth is on by default (`AUTH_ENABLED=true`). Sign in as `admin` with the `API_KEY` from `backend/.env` (default `sk-mock-admin-key`).

## 5. Open the demo and train

1. On the project list, find **Demo · Support FAQ** and click **Open demo**. It creates a project with a CSV of support Q&A rows, a hand-labelled answer key and prepared train / validation / test splits.
2. On the project's **Overview**, follow the next step, **Run and monitor training**.
3. On **Train**, keep the defaults (base model SmolLM2-135M-Instruct, one run), click **Train**, then **Launch** in the confirmation dialog. (The dialog shows cost estimates and the run page shows CUDA / bitsandbytes preflight warnings; in simulate mode they do not apply.)
4. The run completes in under a minute. The **Train** page shows its loss curve (train and eval loss) and status **COMPLETED**.

A blank project (**+ New Project**) has no data yet: add some on **Data** (upload a file, import a dataset or generate examples) before Train or the Autopilot planner can run. The planner says so ("No training data yet … Add data first") with a link to **Data**.

## 6. API and CLI

With auth on, API calls need a Bearer token. Log in once:

```sh
TOKEN=$(curl -s -X POST http://localhost:8000/api/auth/local/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "sk-mock-admin-key"}' \
  | python3 -c 'import sys, json; print(json.load(sys.stdin)["token"])')

curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/api/projects
```

The CLI is `./brewslm` at the repo root (it is not on your `PATH`). `auth login --save` stores the token in `~/.brewslm/token`, and later commands use it:

```sh
./brewslm auth login --username admin --password sk-mock-admin-key --save
./brewslm doctor --project 1
```

The CLI talks to `http://127.0.0.1:8000/api` by default. For a backend elsewhere, set `BREWSLM_API_BASE` or pass `--api-base` (a bare `http://host:port` gets `/api` added):

```sh
BREWSLM_API_BASE=http://127.0.0.1:8100 ./brewslm doctor --project 1
```

## 7. (Optional) Boot the docs

```sh
cd slm-docs
npm install
npm run start
```

This site is then at `http://localhost:3001`. The `?` icon in the app's TopBar links here.

## Real training

To train real weights and get a verdict:

1. Set `TRAINING_BACKEND="external"` and `ALLOW_SIMULATED_TRAINING=false` in `backend/.env` (the `.env.example` values), and restart the backend.
2. Start Redis and a Celery worker in another terminal: `cd backend && source .venv/bin/activate && celery -A app.worker.celery_app worker --loglevel=INFO --pool=threads --concurrency=2`. `.env.example` uses Redis DB 0 (broker) and 1 (results); change the index in `REDIS_URL`, `CELERY_BROKER_URL` and `CELERY_RESULT_BACKEND` if another local app uses them.
3. A GPU is strongly recommended; a tiny base model (the demo's SmolLM2-135M) also trains on CPU, slowly.

When a real run finishes, the automatic lift check evaluates the base model and the fine-tuned run on the same test examples (the bell shows it). **Results** → **Verdict** then says whether the run beat its base model and whether the change is more than noise.

## What you just did

- Cloned the repo, installed dependencies and created the database.
- Brought up the backend and the frontend.
- Opened a demo project with data, an answer key and prepared splits.
- Ran a simulated training run end to end and saw its loss curve.
- Called the API and the CLI with a token.

The next page narrates a real project in more detail — see **[Build your first project](first-project.md)**.

## Common first-run hiccups

| Symptom | Fix |
|---|---|
| `python: command not found` | Use `python3 -m venv .venv`; inside the venv, `python` works. |
| After login you see no projects and can't create one | No `backend/.env`, so `admin` is an engineer with no project access. `cp .env.example .env` in `backend/`, restart the backend, log in again. |
| A training run stays **queued** | The default backend is `external`, which needs Redis + a Celery worker. Start one, or switch to simulated training (step 1). |
| `Alembic head mismatch` on backend startup | Never-stamped DB (made by auto-create): `cd backend && alembic stamp head`. Otherwise `cd backend && alembic upgrade head`. Or move `data/slm_platform.db` aside to rebuild it (loses its data). |
| `401 Authentication required` from curl / the CLI | Log in first (step 6) and send `Authorization: Bearer $TOKEN`; for the CLI, `./brewslm auth login --username admin --save`. |
| The UI loads but every request fails | The Vite proxy points at the wrong backend: set `VITE_API_PROXY_TARGET` to the backend's URL and restart `npm run dev`. |
| `ModuleNotFoundError: app` from the CLI | Use `./brewslm` from the repo root (it uses `backend/.venv`), or activate the backend venv. |
| Geist font flickers on first paint | Loaded fresh from Google Fonts. Self-host via `@fontsource/geist` if it bothers you. |
| The terminal scrolls SQL non-stop | `DEBUG=true` in `.env` echoes every query; set `DEBUG=false`. |

## Next

- [Build your first project](first-project.md) — narrated end-to-end walkthrough.
- [Pipeline overview](../workflows/pipeline-overview.md) — the 11-stage pipeline.
- [Concepts → Architecture](../concepts/architecture.md) — the mental model.

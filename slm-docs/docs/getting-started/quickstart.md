---
sidebar_position: 2
title: Quickstart
---

# Quickstart

Ten minutes from `git clone` to a trained model. Optimised for a new ML engineer who wants to see the whole loop end-to-end before reading any other docs.

## Prerequisites

- Python 3.11+ · Node 20+ · `pip`, `npm`.
- A terminal multiplexer or three windows (backend / frontend / docs).
- **Optional**: NVIDIA GPU for real training. CPU is fine for the simulator runtime that ships built-in.

## 1. Clone + install

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

# Docs (this site)
cd ../slm-docs
npm install
```

The default DB is SQLite at `backend/data/brewslm.db` — created on first boot. No setup needed.

## 2. Boot the backend

```sh
cd backend
source .venv/bin/activate
uvicorn app.main:app --reload --port 8000
```

Should print `Uvicorn running on http://127.0.0.1:8000`. The Swagger UI is at `http://localhost:8000/api/docs` if you want to poke endpoints directly.

## 3. Boot the frontend

```sh
cd frontend
npm run dev
```

Open `http://localhost:5173`. With `AUTH_ENABLED=false` (default in dev) there's no login screen — you land on the project list.

## 4. (Optional) Boot the docs

```sh
cd slm-docs
npm run start
```

Now this site is live at `http://localhost:3001`. The `?` icon in the app's TopBar links here.

## 5. Create your first project

### UI

Click **New Project** on the project list. The wizard asks for a name, template (`general` / `support` / `legal`), and whether to start in **beginner mode** (recommended for a first project). Submit.

### CLI

```sh
brewslm project create --name "Quickstart" --template general
```

### API

```sh
curl -X POST http://localhost:8000/api/projects \
  -H "Content-Type: application/json" \
  -d '{"name": "Quickstart", "template": "general"}'
```

You should now see the project in the sidebar.

## 6. Run the autopilot one-click flow

The fastest path to a trained model is the **Autopilot** — describe what you want, accept the plan, click run.

### UI

1. Open your project workspace.
2. Click **Autopilot Planner** in the Training rail.
3. Type a brief: *"Build a small Q&A assistant from a CSV of FAQ rows."*
4. Click **Plan**. The planner shows the proposed dataset adapter, base model, training recipe, and target profile.
5. Review the **provenance** column — `measured` vs `estimated` per component.
6. Click **One-click run**. The autopilot creates the project artifacts, starts a training job in the built-in simulator, and routes you to the live monitor.

### CLI

```sh
brewslm project bootstrap \
  --name "Quickstart Q&A" \
  --brief "Build a small Q&A assistant from a CSV of FAQ rows." \
  --target edge_gpu \
  --create-project
# Then:
brewslm train start --project 1 --autopilot --one-click \
  --intent "Build a small Q&A assistant from a CSV of FAQ rows."
```

### API

```sh
# Plan
curl -X POST http://localhost:8000/api/projects/1/autopilot/plan \
  -H "Content-Type: application/json" \
  -d '{"intent": "Build a small Q&A assistant from a CSV of FAQ rows."}'

# Run the returned plan
curl -X POST http://localhost:8000/api/projects/1/autopilot/run \
  -H "Content-Type: application/json" \
  -d '{"plan_id": "auto_..."}'
```

## 7. Watch the run

### UI

The Autopilot Planner page switches to live mode when training starts. You see loss curves, per-step telemetry, and a **stop / pause / resume** control. When it finishes, the **Eval** tab fires automatically.

### CLI

```sh
brewslm logs tail --project 1 --run-id exp-1
```

Streams every RunEvent for that run as it lands.

### API

```sh
curl http://localhost:8000/api/run-events/run/exp-1
```

## 8. Inspect the result

Once training is done:

- **Models** under the Training rail lists the new checkpoint + its eval pass rate.
- **Observability** shows the full [Run Timeline](../observability/timeline.md) for the autopilot session → training → eval tree.
- **Deployments** (still empty) is where you'd next [plan + smoke + promote](../deployment/plan.md) the checkpoint.

## What you just did

In about ten minutes you've:

- Cloned the repo + installed dependencies.
- Brought up three services (backend, frontend, docs).
- Created a project with a starter template.
- Run the autopilot loop: brief → plan → train → eval.
- Inspected the result on the UI + CLI + API.

The next page narrates this same workflow in more detail with a real dataset — see **[Build your first project](first-project.md)**.

## Common first-run hiccups

| Symptom | Fix |
|---|---|
| `Alembic head mismatch` on backend startup | Delete `backend/data/brewslm.db` (auto-recreate path takes over) OR run `alembic -c backend/alembic.ini upgrade head`. |
| `ModuleNotFoundError: app` from the CLI | Forgot to activate the backend venv. `source backend/.venv/bin/activate`. |
| Frontend CORS errors in browser console | Set `VITE_API_BASE=http://localhost:8000/api` in `frontend/.env.local`. |
| Geist font flickers on first paint | Loaded fresh from Google Fonts. Self-host via `@fontsource/geist` if it bothers you. |
| GPU not detected | The default `simulate` runtime doesn't need a GPU. To use a real one, set `TRAINING_BACKEND=external` + configure Celery. |

## Next

- [Build your first project](first-project.md) — narrated end-to-end walkthrough.
- [Pipeline overview](../workflows/pipeline-overview.md) — the 11-stage pipeline.
- [Concepts → Architecture](../concepts/architecture.md) — the mental model.

---
sidebar_position: 2
title: Quickstart
---

# Quickstart

This is the fastest way to run BrewSLM locally.

## Prerequisites

- Python 3.10+
- Node.js 18+
- `pip` and `npm`
- Optional: NVIDIA GPU for faster training

## 1) Start Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-base.txt
pip install -r requirements.txt
alembic upgrade head
uvicorn app.main:app --reload --port 8000
```

## 2) Start Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs on `http://localhost:5173` by default.

## 3) (Optional) Start Worker for Long Jobs

```bash
cd backend
source .venv/bin/activate
celery -A app.celery_app worker -l info
```

## 4) Login

- Open `http://localhost:5173/login`
- Use:
  - Username: `admin`
  - Password/API key: value of `API_KEY` in `backend/.env`

## 5) Verify System Health

- Backend API docs: `http://localhost:8000/docs`
- Create one project from UI.
- Confirm you can open Project Wizard, Training, and Export tabs.

## 6) Start Docs Site (for Help button)

```bash
cd slm-docs
npm install
npm start
```

Docs run at `http://localhost:3001/`.

## Common First-Run Issues

- **`ENOSPC` file watchers on frontend**: increase Linux inotify watches.
- **Alembic head mismatch**: run `alembic upgrade head` again.
- **Auth failures**: confirm `AUTH_ENABLED` and `API_KEY` in backend env.
- **GPU not detected**: start with CPU mode first, then add CUDA stack.

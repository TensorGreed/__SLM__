# SLM Platform

> Build, evaluate, compress, govern, and export domain-specific Small Language Models (SLMs) with a guided full-stack workflow.

This repository contains a FastAPI backend + React frontend for end-to-end SLM lifecycle operations: ingestion, cleaning, dataset prep, training, evaluation, compression, export packaging, model registry promotion gates, and project-scoped secret management.

## Latest Updates (March 5, 2026)

- Added **Model Registry** lifecycle with readiness snapshots, promotion gates, and deploy tracking.
- Added **Project Secrets** APIs with masked listing and encrypted-at-rest values.
- Added **Domain Profiles** with a typed contract and project-level assignment.
- Added **Domain Profile Manager UI** (list/create/edit/assign) in project detail and profile selection during project creation.
- Added runtime transparency in split/training responses (applied profile + resolved defaults).
- Added **Duplicate-as-new-version** flow for domain profiles from the project UI.
- Added dedicated **Resolved Defaults** panels in Dataset Prep and Training tabs.
- Added queued **remote ingestion** jobs with task status/cancel APIs and WebSocket log streaming.
- Added queued **compression** jobs (quantize/merge/benchmark) with task status/cancel APIs and WebSocket logs.
- Added **experiment comparison** API/UI for side-by-side metrics and loss-history visualization.
- Added auth UX updates for **SSO + local login** flow.
- Added Alembic revisions:
  - `20260305_0002` for registry + secrets tables
  - `20260305_0003` for domain profiles + project binding

---

## Quick Start (Local Dev)

### Prerequisites

- Python 3.11+
- Node.js 18+
- Redis (required for queued training/compression/remote-import jobs)
- Optional GPU/CUDA for real training and inference benchmarking

### 1. Backend API

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

# Optional (recommended for Postgres/prod-like setups):
# alembic upgrade head

uvicorn app.main:app --reload --port 8000
```

Notes:
- Default local DB is SQLite (`sqlite+aiosqlite:///./data/slm_platform.db`).
- With `ALLOW_SQLITE_AUTOCREATE=true` (default), tables auto-create on startup.

### 2. Worker (Celery)

Run this in a second terminal if you want queued jobs (training external runtime, compression, remote imports):

```bash
cd backend
source .venv/bin/activate
celery -A app.worker.celery_app worker --loglevel=INFO --pool=threads --concurrency=2
```

### 3. Frontend

```bash
cd frontend
npm install
npm run dev
```

### 4. Login and First Project

1. Open `http://localhost:5173/login`
2. If OIDC is configured, use **Sign in with SSO**
3. Otherwise use local login:
   - Username: any value (new users are auto-provisioned)
   - Password: `API_KEY` from `backend/.env` (default `sk-mock-admin-key`)
4. Create a project and walk tabs:
   - Data -> Cleaning -> Gold Set -> Synthetic -> Dataset Prep -> Tokenization -> Training -> Evaluation -> Compression -> Export

### 5. API Docs

With backend running, visit: `http://localhost:8000/docs`

---

## Optional Docker Compose

Repo includes `docker-compose.yml` for:
- Postgres
- Redis
- Backend API
- Celery worker
- Frontend

```bash
docker compose up --build
```

If your frontend image build fails, verify `frontend/nginx.conf` exists and matches your deployment routing needs.

---

## Authentication and Authorization

### Auth modes

- **SSO mode**: enable OIDC vars (`OIDC_CLIENT_ID`, `OIDC_CLIENT_SECRET`, `OIDC_DISCOVERY_URL`)
- **Local mode**: `/api/auth/local/login` returns a JWT token (used by frontend)

### Access control

- Global roles: `admin`, `engineer`, `viewer`
- Project membership roles: `owner`, `editor`, `viewer`
- Request auth supports either:
  - `Authorization: Bearer <jwt-or-api-key>`
  - `x-api-key: <api-key>`
- Audit logs are captured for mutating API calls (configurable via `AUDIT_LOG_ENABLED`)

---

## Pipeline and API Surface

All project-scoped routes are under `/api/projects/{project_id}/...`.

| Domain | Route Prefix | Key Capabilities |
|---|---|---|
| Projects | `/projects` | CRUD, stats, base model metadata |
| Domain Profiles | `/domain-profiles` | Create/list/get/update contract profiles |
| Pipeline | `/projects/{id}/pipeline` | Stage status, advance, rollback |
| Ingestion | `/projects/{id}/ingestion` | Upload/batch upload, remote import (sync + queued), document lifecycle, job status/cancel, WS logs |
| Cleaning | `/projects/{id}/cleaning` | Clean, clean-batch, chunk inspection |
| Gold Set | `/projects/{id}/gold` | Add/import/list/lock evaluation gold data |
| Synthetic | `/projects/{id}/synthetic` | Teacher-model generation + save workflow |
| Dataset Prep | `/projects/{id}/dataset` | Split, preview, schema/profile diagnostics (profile-aware defaults) |
| Tokenization | `/projects/{id}/tokenization` | Token stats + vocab sample |
| Training | `/projects/{id}/training` | Experiments, start/cancel, status, task status, WS telemetry/logs (profile-aware defaults) |
| Comparison | `/projects/{id}/training/compare` | Compare up to 5 experiments side by side |
| Evaluation | `/projects/{id}/evaluation` | Exact Match/F1/Safety, LLM judge, held-out evaluation, scorecards |
| Compression | `/projects/{id}/compression` | Quantize/merge/benchmark queue, report status, task status/cancel, WS logs |
| Export | `/projects/{id}/export` | Create/run/list exports with manifests |
| Registry | `/projects/{id}/registry` | Register, readiness snapshot, promote with gates, deploy metadata (profile-aware gate defaults) |
| Secrets | `/projects/{id}/secrets` | Upsert/list/delete project secrets with encrypted storage |
| Auth | `/auth` | Config, me, SSO flow, local login, user/member management |
| Audit | `/audit` | Audit log listing |

---

## Domain Profile Runtime Wiring

Domain profiles are not just metadata; they are applied during runtime.

- Default bootstrap profile: `generic-domain-v1`
- New projects auto-attach to `generic-domain-v1` when available.
- You can re-assign profile per project:
  - `PUT /api/projects/{project_id}/domain-profile` with `{ "profile_id": "..." }`
- Server-side duplication:
  - `POST /api/domain-profiles/{profile_id}/duplicate`
  - Auto-generates next `profile_id`/`version` unless explicitly overridden in request body.
- Frontend wiring:
  - Project detail page exposes Domain Profile Manager to list/create/edit/assign contracts.
  - Selected profile can be duplicated as a new version (server-generated ID/version, default status `draft`) and opened immediately in the editor.
  - New project modal allows selecting a profile (or auto-assign default).
  - Dataset split and training forms can omit untouched fields so profile defaults are actually applied.
  - Dataset Prep and Training tabs show dedicated "Resolved Defaults" panels with applied profile, fields sourced from profile defaults, and resolved config payloads.

Current enforced behavior:

- Dataset split defaults:
  - `POST /api/projects/{project_id}/dataset/split`
  - `POST /api/projects/{project_id}/dataset/split/effective-config` previews resolved config pre-run.
  - If omitted in request body, `train_ratio`, `val_ratio`, `test_ratio`, `seed` are pulled from profile `dataset_split`.
  - If `chat_template` is omitted, it is taken from profile `training_defaults.chat_template`.
  - Response now includes:
    - `domain_profile_applied`
    - `profile_split_defaults`
    - `resolved_split_config`
    - `profile_defaults_applied`
- Training experiment defaults:
  - `POST /api/projects/{project_id}/training/experiments`
  - `POST /api/projects/{project_id}/training/experiments/effective-config` previews resolved config pre-create.
  - For omitted config fields, defaults come from profile `training_defaults` (e.g. `batch_size`, `num_epochs`, `learning_rate`, `chat_template`, `use_lora`, `training_mode`).
  - Response now includes:
    - `domain_profile_applied`
    - `profile_training_defaults`
    - `resolved_training_config`
    - `profile_defaults_applied`
- Registry promotion gates:
  - `POST /api/projects/{project_id}/registry/models/{model_id}/promote`
  - Gate defaults are pulled from profile `registry_gates.to_staging` / `registry_gates.to_production`.
  - Explicit request `gates` still override profile/default values.

## Production-Oriented Capabilities

- Strict runtime modes (no silent fallbacks unless explicitly enabled)
- Alembic migration head enforcement (`DB_REQUIRE_ALEMBIC_HEAD=true`)
- Queued background jobs with task-level status/cancel APIs
- Real-time worker logs over Redis PubSub WebSockets
- Dataset normalization for heterogeneous schemas
- Export run packaging with checksums + versioned run directories
- Model registry governance with promotion gating and regression checks
- Domain profile-driven runtime defaults and policy gates
- Project-level encrypted secret storage for connectors/providers

---

## Environment Configuration

Common backend env vars (see `backend/.env.example`):

### Core and DB

- `DATABASE_URL` (default: `sqlite+aiosqlite:///./data/slm_platform.db`)
- `DB_AUTO_CREATE` (default: `false`)
- `ALLOW_SQLITE_AUTOCREATE` (default: `true`)
- `DB_REQUIRE_ALEMBIC_HEAD` (default: `true`)
- `ALEMBIC_CONFIG_FILE` (default: `alembic.ini`)
- `DATA_DIR`

### Auth and security

- `AUTH_ENABLED`
- `AUTH_BOOTSTRAP_API_KEY`
- `AUTH_BOOTSTRAP_USERNAME` (default: `admin`)
- `AUTH_BOOTSTRAP_ROLE` (default: `admin`)
- `API_KEY` (used by local login password)
- `JWT_SECRET`
- `AUDIT_LOG_ENABLED`
- `OIDC_CLIENT_ID`, `OIDC_CLIENT_SECRET`, `OIDC_DISCOVERY_URL`
- `SECRETS_ENCRYPTION_KEY` (optional; falls back to `JWT_SECRET` if unset)

### Worker/broker

- `REDIS_URL`
- `CELERY_BROKER_URL`
- `CELERY_RESULT_BACKEND`

### Ingestion / synthetic / judge

- `ALLOW_SIMULATED_INGESTION_FALLBACK`
- `TEACHER_MODEL_API_URL`, `TEACHER_MODEL_API_KEY`
- `ALLOW_SYNTHETIC_DEMO_FALLBACK`
- `JUDGE_MODEL_API_URL`, `JUDGE_MODEL_API_KEY`

### Training runtime

- `TRAINING_BACKEND` (`simulate` or `external`)
- `ALLOW_SIMULATED_TRAINING`
- `TRAINING_EXTERNAL_CMD`

### Compression runtime

- `COMPRESSION_BACKEND` (`external` or `stub`)
- `ALLOW_STUB_COMPRESSION`
- `QUANTIZE_EXTERNAL_CMD`
- `MERGE_LORA_EXTERNAL_CMD`
- `BENCHMARK_EXTERNAL_CMD`
- `EXTERNAL_COMMAND_TIMEOUT_SECONDS`

---

## External Runtime Command Templates

Example training command template:

```bash
TRAINING_BACKEND=external
TRAINING_EXTERNAL_CMD='python "{backend_dir}/scripts/train.py" --project {project_id} --experiment {experiment_id} --output "{output_dir}" --base-model "{base_model}" --config "{config_path}" --train-file "{train_file}" --val-file "{val_file}"'
```

`backend/scripts/train.py` performs real HuggingFace Trainer fine-tuning and expects prepared split files (for example `train.jsonl`).

Example compression template:

```bash
COMPRESSION_BACKEND=external
QUANTIZE_EXTERNAL_CMD='python "{backend_dir}/scripts/quantize.py" --project {project_id} --model "{model_path}" --bits {bits} --format {output_format} --out "{output_model_path}"'
```

`backend/scripts/quantize.py` and `backend/scripts/benchmark.py` are starter external runtimes you can replace with your own production commands.

---

## Database Migrations (Alembic)

Run migrations:

```bash
cd backend
alembic upgrade head
```

Create a new migration:

```bash
cd backend
alembic revision --autogenerate -m "describe change"
alembic upgrade head
```

Current revisions:
- `20260304_0001` baseline schema
- `20260305_0002` model registry + project secrets
- `20260305_0003` domain profiles + project binding

From repo root:

```bash
alembic -c backend/alembic.ini upgrade head
```

---

## Testing

Backend unit tests:

```bash
cd backend
python -m unittest discover -s tests -v
```

Frontend production build check:

```bash
cd frontend
npm run build
```

CI workflow (`.github/workflows/ci.yml`) runs backend tests and frontend build on pushes/PRs.

---

## Repository Layout

```text
__SLM__/
├── backend/
│   ├── app/
│   │   ├── api/          # FastAPI routers
│   │   ├── models/       # SQLAlchemy ORM models
│   │   ├── schemas/      # Pydantic schemas
│   │   ├── services/     # Business logic
│   │   ├── pipeline/     # Stage orchestration
│   │   └── utils/        # Parsers/helpers
│   ├── alembic/          # DB migrations
│   └── scripts/          # External runtime scripts
├── frontend/src/
│   ├── components/       # Pipeline and dashboard UI
│   ├── pages/            # Login, project list, project detail
│   ├── stores/           # Zustand state stores
│   └── api/              # Axios client
├── data/                 # Runtime data, artifacts, manifests
└── docker-compose.yml
```

---

## License

MIT

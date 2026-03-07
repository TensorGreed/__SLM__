# SLM Platform

> Build, evaluate, compress, govern, and export domain-specific Small Language Models (SLMs) with a guided full-stack workflow.

This repository contains a FastAPI backend + React frontend for end-to-end SLM lifecycle operations: ingestion, cleaning, dataset prep, training, evaluation, compression, export packaging, model registry promotion gates, and project-scoped secret management.

## Latest Updates (March 5, 2026)

- Added **Model Registry** lifecycle with readiness snapshots, promotion gates, and deploy tracking.
- Added **Project Secrets** APIs with masked listing and encrypted-at-rest values.
- Added **Domain Profiles** with a typed contract and project-level assignment.
- Added **Domain Packs** (pluggable overlays + default profile pointer) with project-level assignment.
- Added **Domain Pack Hooks** for custom normalizers, validators, and evaluators.
- Added **Domain Profile Manager UI** (list/create/edit/assign) in project detail and profile selection during project creation.
- Added **Domain Pack Manager UI** (list/create/edit/assign) in project detail and pack selection during project creation.
- Added **Workflow Graph Preview** (read-only visual stage graph + step contracts) in project detail.
- Added **Workflow Contract Runtime (Phase 2)** APIs: graph validate, dry-run, run active step, and run history.
- Added **Visual Pipeline Editor (Phase 3)** with project-scoped graph contract save/load/reset and compile diagnostics.
- Added **Step Contract SDK v1** runtime fields (`runtime_requirements`) with compile/dry-run/run-step enforcement.
- Added **Typed Artifact Registry (Phase 4)** with versioned artifacts and pipeline run-step artifact publication.
- Added **Workflow Runner v1 (Phase 5)** with persisted DAG runs, node retries, dependency tracking, and backend adapters (`local`, `celery`, `external`).
- Added **Workflow Templates + Run Monitor UI (Phase 6)** on the Workflow page.
- Refactored project UI into a cleaner workspace IA:
  - `Pipeline` page
  - `Workflow Graph` page
  - `Domain Contracts` page
  - left sidebar navigation with pipeline stages as submenu
- Added runtime transparency in split/training responses (applied profile + resolved defaults).
- Added **Duplicate-as-new-version** flow for domain profiles from the project UI.
- Added dedicated **Resolved Defaults** panels in Dataset Prep and Training tabs.
- Added queued **remote ingestion** jobs with task status/cancel APIs and WebSocket log streaming.
- Added queued **compression** jobs (quantize/merge/benchmark) with task status/cancel APIs and WebSocket logs.
- Added real GGUF/ONNX compression runtime paths in `backend/scripts/quantize.py`.
- Added **experiment comparison** API/UI for side-by-side metrics and loss-history visualization.
- Added auth UX updates for **SSO + local login** flow.
- Added training runtime generalization:
  - trainer backend abstraction (`auto`, `hf_trainer`, `trl_sft`)
  - task/data adapter contract (`causal_lm`, `seq2seq`, `classification`)
  - CUDA OOM auto-retry planner with progressive memory downscaling
- Added **Training Capability Matrix + Preflight**:
  - explicit config preflight endpoint before experiment creation/start
  - model/task/trainer compatibility checks
  - dependency/runtime/data-file preflight checks
  - start-time server guard that blocks incompatible training runs
- Added **Training Runtime Plugin SDK v1**:
  - runtime registry + catalog endpoint (`/training/runtimes`)
  - pluggable runtime selection via `training_runtime_id`
  - built-in runtimes: `builtin.simulate`, `builtin.external_celery`
  - legacy `TRAINING_BACKEND` remains default fallback for backward compatibility
- Added **Training Recipe System v1**:
  - built-in domain-agnostic recipe catalog (safe/balanced SFT, LoRA-fast, classification, seq2seq)
  - recipe resolve endpoint with runtime/profile default merge + optional preflight
  - Training UI recipe starter picker with one-click apply
- Added **Evaluation Packs + Auto Gates v1**:
  - built-in gate packs (`general`, `strict`, `fast-iteration`) + domain-profile-derived dynamic pack
  - project-level evaluation pack preference with fallback to defaults
  - experiment gate evaluation endpoint and pipeline status auto-gate summary
- Added **Preflight Plan Suggestions** (`safe`, `balanced`, `max_quality`) with one-click config apply in Training UI.
- Added project-persisted training plan preference (`preferred_plan_profile`) so recommended profile choice is remembered per project.
- Added project/domain-pack adapter preset resolution for dataset split defaults (`adapter_id`, `adapter_config`, `field_mapping`).
- Added strict dataset contract checks in training preflight (task-shape coverage gate with actionable fix hints).
- Expanded Training experiment form controls for memory/runtime tuning (`gradient_accumulation_steps`, `max_seq_length`, `save_steps`, `eval_steps`, `fp16`/`bf16`, `flash_attention`, `sequence_packing`).
- Added split backend dependency profiles for CPU/GPU installs (`requirements-base.txt`, `requirements.txt`, `requirements-gpu.txt`, `requirements-gpu-cu128.txt`).
- Added live external-training telemetry streaming (`epoch`, `step`, loss metrics) from Celery worker to Training dashboard via WebSocket.
- Added Alembic revisions:
  - `20260305_0002` for registry + secrets tables
  - `20260305_0003` for domain profiles + project binding
  - `20260305_0004` for domain packs + project binding
  - `20260305_0005` for artifact registry table
  - `20260306_0007` for project training preference persistence
  - `20260306_0008` for project dataset adapter preset persistence
  - `20260307_0009` for project evaluation pack preference persistence

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
# CPU/default install profile:
pip install -r requirements.txt

# GPU profile (all non-torch deps):
# pip install -r requirements-gpu.txt
# then install CUDA-enabled torch for your platform

# Example only (Linux x86_64 + CUDA 12.8):
# pip install -r requirements-gpu-cu128.txt
cp .env.example .env

# Recommended whenever you pull latest backend schema changes on an existing DB:
# alembic upgrade head

uvicorn app.main:app --reload --port 8000
```

Notes:
- Default local DB is SQLite (`sqlite+aiosqlite:///./data/slm_platform.db`).
- With `ALLOW_SQLITE_AUTOCREATE=true` (default), tables auto-create on startup.
- `requirements.txt` is the default CPU-friendly profile.
- For GPU training, use `requirements-gpu.txt` and then install a CUDA-enabled torch wheel that matches your OS/arch/CUDA stack.
  - For Linux aarch64 systems, use your vendor/official CUDA torch wheel (PyPI `torch` is often CPU-only).
- Verify torch runtime before training:

```bash
cd backend
source .venv/bin/activate
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_devices:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device0:", torch.cuda.get_device_name(0))
PY
```

### 2. Worker (Celery)

Run this in a second terminal if you want queued jobs (training external runtime, compression, remote imports):

```bash
cd backend
source .venv/bin/activate
celery -A app.worker.celery_app worker --loglevel=INFO --pool=threads --concurrency=2
```

If long-running tasks (for example training jobs >1 hour) are re-received repeatedly with the same task id, set `CELERY_VISIBILITY_TIMEOUT_SECONDS` in `backend/.env` to a larger value (for example `43200`).

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

### 6. Synthetic Generation with Ollama

Use these defaults in the **Synthetic** tab:

- Provider: `Local (Ollama)`
- API URL: `http://localhost:11434/v1/chat/completions`
- Model: `llama3.1:latest` (or any installed Ollama chat model)

Quick checks:

```bash
ollama serve
ollama list
```

If you see `Teacher model response was not valid JSON for Q&A extraction`:

- Update to the latest backend code (parser now tolerates fenced/wrapped JSON and `Q:/A:` text).
- Keep `Pairs to Generate` small at first (`3-8`) and start with a smaller source selection.
- Ensure the model is available locally (`ollama list`) and the API URL is reachable from backend host.

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
| Domain Packs | `/domain-packs` | Create/list/get/update pack overlays, hook catalog/reload |
| Domain Profiles | `/domain-profiles` | Create/list/get/update contract profiles |
| Pipeline | `/projects/{id}/pipeline` | Stage status (+ auto-gate summary), read-only graph preview, graph runtime actions, advance, rollback |
| Ingestion | `/projects/{id}/ingestion` | Upload/batch upload, remote import (sync + queued), document lifecycle, job status/cancel, WS logs |
| Cleaning | `/projects/{id}/cleaning` | Clean, clean-batch, chunk inspection |
| Gold Set | `/projects/{id}/gold` | Add/import/list/lock evaluation gold data |
| Synthetic | `/projects/{id}/synthetic` | Teacher-model generation + save workflow |
| Dataset Prep | `/projects/{id}/dataset` | Split, preview, schema/profile diagnostics (runtime-aware defaults) |
| Tokenization | `/projects/{id}/tokenization` | Token stats + vocab sample |
| Training | `/projects/{id}/training` | Experiments, config preflight, start/cancel, status, task status, WS telemetry/logs (runtime-aware defaults) |
| Comparison | `/projects/{id}/training/compare` | Compare up to 5 experiments side by side |
| Evaluation | `/projects/{id}/evaluation` | Exact Match/F1/Safety, LLM judge, held-out evaluation, scorecards, eval pack preferences, auto-gate reports |
| Compression | `/projects/{id}/compression` | Quantize/merge/benchmark queue, report status, task status/cancel, WS logs |
| Export | `/projects/{id}/export` | Create/run/list exports with manifests |
| Artifacts | `/projects/{id}/artifacts` | Publish/list/version typed artifacts and resolve latest keys |
| Registry | `/projects/{id}/registry` | Register, readiness snapshot, promote with gates, deploy metadata (runtime-aware gate defaults) |
| Secrets | `/projects/{id}/secrets` | Upsert/list/delete project secrets with encrypted storage |
| Auth | `/auth` | Config, me, SSO flow, local login, user/member management |
| Audit | `/audit` | Audit log listing |

Pipeline graph preview endpoint:
- `GET /api/projects/{project_id}/pipeline/graph`
- Returns a read-only node/edge graph with stage status and step contract metadata (`input_artifacts`, `output_artifacts`, `config_schema_ref`).
- Node contracts also include `runtime_requirements` (`execution_modes`, `required_services`, `required_env`, `required_settings`, `requires_gpu`, `min_vram_gb`).
- `GET /api/projects/{project_id}/pipeline/graph/stage-catalog`
- `GET /api/projects/{project_id}/pipeline/graph/templates`
- `GET /api/projects/{project_id}/pipeline/graph/contract`
- `PUT /api/projects/{project_id}/pipeline/graph/contract`
- `DELETE /api/projects/{project_id}/pipeline/graph/contract`
- `POST /api/projects/{project_id}/pipeline/graph/validate`
- `POST /api/projects/{project_id}/pipeline/graph/compile`
- `POST /api/projects/{project_id}/pipeline/graph/dry-run`
- `POST /api/projects/{project_id}/pipeline/graph/run-step`
- `GET /api/projects/{project_id}/pipeline/graph/runs`
- `POST /api/projects/{project_id}/pipeline/graph/run`
- `POST /api/projects/{project_id}/pipeline/graph/run-async`
- `GET /api/projects/{project_id}/pipeline/graph/workflow-runs`
- `GET /api/projects/{project_id}/pipeline/graph/workflow-runs/{run_id}`
- Runtime mode executes active-step gating with contract checks + stage guardrails; completed step runs now publish declared outputs as versioned artifact records.
- Dry-run/compile artifact availability resolves from typed artifact registry (plus existing inferred fallback checks).
- Compile and run-step now also enforce active-node runtime requirements (missing env/settings/services/GPU checks surface in API responses).
- Workflow DAG runs persist in DB (`workflow_runs` + `workflow_run_nodes`) and support:
  - dependency-aware node ordering
  - per-node retries (`max_retries`)
  - stop-on-block / stop-on-failure controls
  - backend execution mode selection (`local`, `celery`, `external`)
  - `celery` backend dispatches each node attempt to worker task `run_workflow_node_job` and waits for task result (requires worker + broker/result backend).
  - executable custom step type `core.data_adapter_preview` for adapter coverage gating in DAG runs (`local` and `celery` backends).
- Workflow page now includes:
  - template picker in Visual Pipeline Editor (`SFT`, `LoRA`, `Distillation`, `Eval-only`)
  - stage palette entry for `data_adapter_preview` and node-level JSON config editing in inspector
  - node config presets in inspector for `core.training`, `core.evaluation`, and `core.export` to avoid manual JSON authoring
  - executable node config modes (local + celery):
    - `core.training`: `noop` (default), `create_and_start`, `start_existing`
    - `core.evaluation`: `noop` (default), `heldout`
    - `core.export`: `noop` (default), `create_and_run`, `run_existing`
  - conservative defaults to preserve existing behavior: if no mode is set, these nodes remain `noop`
  - Workflow Run Monitor panel with run controls, background DAG queueing, auto-refresh polling, recent run history, and per-node attempt status
- Phase 3 adds persisted per-project graph overrides and compile-time graph diagnostics before save/run.

Artifact registry endpoints:
- `POST /api/projects/{project_id}/artifacts/publish`
- `POST /api/projects/{project_id}/artifacts/publish-batch`
- `GET /api/projects/{project_id}/artifacts`
- `GET /api/projects/{project_id}/artifacts/keys`
- `GET /api/projects/{project_id}/artifacts/latest/{artifact_key}`
- Artifacts are immutable per `(project_id, artifact_key, version)` and include `producer_stage`, `producer_run_id`, `schema_ref`, and metadata.

---

## Domain Runtime Wiring

Domain packs and profiles are runtime primitives. The system always resolves an effective contract with fallback.

- Default bootstrap profile: `generic-domain-v1`
- Default bootstrap pack: `general-pack-v1` (default profile pointer -> `generic-domain-v1`)
- New projects auto-attach to `general-pack-v1` when available.
- New projects auto-attach to the selected pack's default profile when available, otherwise fall back to `generic-domain-v1`.
- You can inspect resolved runtime contract:
  - `GET /api/projects/{project_id}/domain-runtime`
  - Response includes `pack_hooks` and merged `effective_contract`.
- You can re-assign pack and profile per project:
  - `PUT /api/projects/{project_id}/domain-pack` with `{ "pack_id": "...", "adopt_pack_default_profile": true }`
  - `PUT /api/projects/{project_id}/domain-profile` with `{ "profile_id": "..." }`
- Merge precedence is deterministic:
  1. Core platform default profile contract
  2. Resolved domain profile contract (project -> pack default profile -> platform default)
  3. Domain pack overlay (project pack -> platform default pack)
- Server-side pack duplication:
  - `POST /api/domain-packs/{pack_id}/duplicate`
- Hook catalog:
  - `GET /api/domain-packs/hooks/catalog`
- Hook plugin reload (admin/engineer):
  - `POST /api/domain-packs/hooks/reload`
- Server-side duplication:
  - `POST /api/domain-profiles/{profile_id}/duplicate`
  - Auto-generates next `profile_id`/`version` unless explicitly overridden in request body.
- Pack contract shape (example):

```json
{
  "$schema": "slm.domain-pack/v1",
  "pack_id": "general-pack-v1",
  "version": "1.0.0",
  "display_name": "General Domain Pack",
  "default_profile_id": "generic-domain-v1",
  "hooks": {
    "normalizer": { "id": "default-normalizer", "config": {} },
    "validator": { "id": "default-validator", "config": {} },
    "evaluator": { "id": "default-evaluator", "config": {} }
  },
  "overlay": {
    "dataset_split": { "train": 0.8, "val": 0.1, "test": 0.1, "seed": 42 },
    "training_defaults": { "training_mode": "sft", "chat_template": "llama3" },
    "registry_gates": {
      "to_staging": { "min_metrics": { "f1": 0.65, "llm_judge_pass_rate": 0.75 } },
      "to_production": { "min_metrics": { "f1": 0.7, "llm_judge_pass_rate": 0.8, "safety_pass_rate": 0.92 } }
    }
  }
}
```

- Built-in hook IDs (initial set):
  - normalizers: `default-normalizer`, `qa-required-normalizer`, `min-text-length-normalizer`, `strip-markdown-normalizer`
  - validators: `default-validator`, `min-text-length-validator`, `qa-pair-validator`
  - evaluators: `default-evaluator`, `pass-rate-band-evaluator`, `weighted-score-evaluator`
- External/plugin hooks:
  - Configure `DOMAIN_HOOK_PLUGIN_MODULES` with Python module paths.
  - On startup, modules are imported and any exported hooks are registered.
  - Catalog response includes plugin load status and load errors.
  - Example module: `app.plugins.domain_hooks.example_hooks`.
  - Module can expose either:
    - `register_domain_hooks(register)` callback API
    - `get_domain_hooks()` returning `{ normalizers, validators, evaluators }`
    - or constants `NORMALIZER_HOOKS` / `VALIDATOR_HOOKS` / `EVALUATOR_HOOKS`
- Frontend wiring:
  - Project detail page exposes Domain Pack Manager to list/create/edit/assign pack contracts.
  - Domain Pack Manager includes Hook Catalog view and "Reload Plugins" action for runtime hook operations.
  - Domain Pack contract editor includes a Hook Helper to load/apply hook IDs into `hooks.normalizer|validator|evaluator`.
  - Project detail page exposes Domain Profile Manager to list/create/edit/assign contracts.
  - Selected pack can be duplicated as a new version and opened immediately in the editor.
  - Selected profile can be duplicated as a new version (server-generated ID/version, default status `draft`) and opened immediately in the editor.
  - New project modal allows selecting a pack/profile (or auto-assign defaults).
  - Dataset split and training forms can omit untouched fields so runtime defaults are actually applied.
  - Dataset Prep and Training tabs show dedicated "Resolved Defaults" panels with applied pack/profile, fields sourced from runtime defaults, and resolved config payloads.

Current enforced behavior:

- Dataset split defaults:
  - `POST /api/projects/{project_id}/dataset/split`
  - `POST /api/projects/{project_id}/dataset/split/effective-config` previews resolved config pre-run.
  - If omitted in request body, `train_ratio`, `val_ratio`, `test_ratio`, `seed` are pulled from resolved runtime `dataset_split`.
  - If `chat_template` is omitted, it is taken from resolved runtime `training_defaults.chat_template`.
  - Response now includes:
    - `domain_pack_applied`
    - `domain_pack_source`
    - `domain_profile_applied`
    - `domain_profile_source`
    - `profile_split_defaults`
    - `resolved_split_config`
    - `profile_defaults_applied`
- Dataset profile diagnostics:
  - `POST /api/projects/{project_id}/dataset/profile`
  - Response now includes:
    - `domain_hooks`
    - `validator_report`
- Dataset adapter SDK + preview:
  - `GET /api/projects/{project_id}/dataset/adapters/catalog` lists built-in/plugin adapters and schema hints.
  - `POST /api/projects/{project_id}/dataset/adapters/preview` samples project data and reports adapter mapping coverage, drop/error counts, and mapped-row previews.
  - `POST /api/projects/{project_id}/dataset/adapters/reload` reloads adapter plugins from env-configured modules.
  - `GET /api/projects/{project_id}/dataset/adapter-preference` resolves adapter preset fallback chain (`project` -> `domain_pack` -> `default`).
  - `PUT /api/projects/{project_id}/dataset/adapter-preference` persists a project-level adapter preset.
  - `POST /api/projects/{project_id}/dataset/adapter-preference/auto-detect` auto-detects adapter from sampled rows and can save it as project preset.
  - `POST /api/projects/{project_id}/dataset/split` now accepts `adapter_id`, `adapter_config`, and `field_mapping` (all optional; default adapter fallback remains active).
  - Dataset Prep UI now has dedicated views (`Overview`, `Adapter Lab`, `Split`) to reduce clutter.
  - Adapter Lab and split flows both support JSON adapter config input.
  - Split UI can run with explicit adapter contract (`adapter_id` + `adapter_config`) instead of only default normalization.
- Cleaning behavior:
  - Remote-imported structured documents (`jsonl/json/csv`) can be cleaned directly.
  - If `.extracted.txt` is missing, cleaning will synthesize extractable text from structured rows.
- Training experiment defaults:
  - `POST /api/projects/{project_id}/training/experiments`
  - `POST /api/projects/{project_id}/training/experiments/effective-config` previews resolved config pre-create.
  - `POST /api/projects/{project_id}/training/experiments/preflight` resolves config + runs capability/runtime preflight before create/start.
    - Includes dataset contract validation against requested `task_type` (`causal_lm`, `seq2seq`, `classification`) and emits explicit fix hints when coverage is insufficient.
  - `POST /api/projects/{project_id}/training/experiments/preflight/plan` returns suggested configs (`safe`, `balanced`, `max_quality`) with estimated VRAM risk and per-profile preflight output.
  - `GET /api/projects/{project_id}/training/runtimes` lists registered training runtime plugins and server default runtime.
  - `GET /api/projects/{project_id}/training/recipes` lists built-in training recipes.
  - `POST /api/projects/{project_id}/training/recipes/resolve` applies recipe patch over base config, then resolves runtime defaults and preflight.
  - `GET /api/projects/{project_id}/training/preferences` reads persisted project training UI preferences.
  - `PUT /api/projects/{project_id}/training/preferences` updates persisted project training UI preferences (currently `preferred_plan_profile`).
  - `GET /api/projects/{project_id}/training/experiments/{experiment_id}/preflight` runs preflight on an existing experiment.
  - For omitted config fields, defaults come from resolved runtime `training_defaults` (e.g. `batch_size`, `num_epochs`, `learning_rate`, `chat_template`, `use_lora`, `training_mode`).
  - Runtime config now also supports:
    - `training_runtime_id`: runtime plugin id (`auto` uses server default)
    - `task_type`: `causal_lm` | `seq2seq` | `classification`
    - `trainer_backend`: `auto` | `hf_trainer` | `trl_sft`
    - `auto_oom_retry`, `max_oom_retries`, `oom_retry_seq_shrink`
  - Task-specific trainer runtime:
    - `causal_lm` uses `AutoModelForCausalLM` (`hf_trainer` or `trl_sft` when available).
    - `seq2seq` uses `AutoModelForSeq2SeqLM` + seq2seq collator.
    - `classification` uses `AutoModelForSequenceClassification` + label mapping + eval accuracy/macro-F1.
    - If `trainer_backend=trl_sft` is requested for non-`causal_lm`, runtime falls back to `hf_trainer` with a warning.
  - `POST /start` now enforces preflight server-side and returns `400` with `Training preflight failed: ...` when blocked.
  - Training UI includes:
    - `Run Capability Preflight` for pass/fail + warnings
    - `Run Preflight Plan` for recommended configs and one-click apply
  - Response now includes:
    - `domain_pack_applied`
    - `domain_pack_source`
    - `domain_profile_applied`
    - `domain_profile_source`
    - `profile_training_defaults`
    - `resolved_training_config`
    - `profile_defaults_applied`
- Evaluation hook application:
  - `POST /api/projects/{project_id}/evaluation/run`
  - `POST /api/projects/{project_id}/evaluation/llm-judge`
  - `GET /api/projects/{project_id}/evaluation/packs`
  - `GET /api/projects/{project_id}/evaluation/pack-preference`
  - `PUT /api/projects/{project_id}/evaluation/pack-preference`
  - `GET /api/projects/{project_id}/evaluation/gates/{experiment_id}`
  - Metrics are post-processed by the active pack evaluator hook (for example adds `quality_band`).
- Pipeline status now returns `auto_gate` summary:
  - `GET /api/projects/{project_id}/pipeline/status`
  - includes latest experiment gate pass/fail, failed gate IDs, and missing required metrics.
- Registry promotion gates:
  - `POST /api/projects/{project_id}/registry/models/{model_id}/promote`
  - Gate defaults are pulled from resolved runtime `registry_gates.to_staging` / `registry_gates.to_production`.
  - Explicit request `gates` still override profile/default values.

Remote import request notes:
- `source_type` accepts `huggingface`, `kaggle`, `url` (also normalizes `hf` and minor formatting variants).
- `max_samples <= 0` is treated as omitted/default instead of failing request validation.
- Remote import request supports `adapter_id` and `adapter_config`; default fallback is `default-canonical`.
- Ingestion UI includes an **Adapter Mapping** block for optional `adapter_config` JSON overrides during remote imports.

## Production-Oriented Capabilities

- Strict runtime modes (no silent fallbacks unless explicitly enabled)
- Alembic migration head enforcement (`DB_REQUIRE_ALEMBIC_HEAD=true`)
- Queued background jobs with task-level status/cancel APIs
- Real-time worker logs over Redis PubSub WebSockets
- Dataset normalization for heterogeneous schemas
- Export run packaging with checksums + versioned run directories
- Model registry governance with promotion gating and regression checks
- Domain pack + profile-driven runtime defaults and policy gates
- Project-level encrypted secret storage for connectors/providers

---

## Environment Configuration

Common backend env vars (see `backend/.env.example`):

### Core and DB

- `DATABASE_URL` (default: `sqlite+aiosqlite:///./data/slm_platform.db`)
  - If using SQLite, prefer an absolute path if you launch `uvicorn` from different directories.
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
- `CELERY_VISIBILITY_TIMEOUT_SECONDS` (increase for long-running tasks; default `43200`)

### Ingestion / synthetic / judge

- `ALLOW_SIMULATED_INGESTION_FALLBACK`
- `TEACHER_MODEL_API_URL`, `TEACHER_MODEL_API_KEY`
- `ALLOW_SYNTHETIC_DEMO_FALLBACK`
- `JUDGE_MODEL_API_URL`, `JUDGE_MODEL_API_KEY`

### Training runtime

- `TRAINING_BACKEND` (`simulate` or `external`)
- `ALLOW_SIMULATED_TRAINING`
- `TRAINING_EXTERNAL_CMD`
- `TRAINING_RUNTIME_PLUGIN_MODULES` (JSON array of Python module paths)

### Compression runtime

- `COMPRESSION_BACKEND` (`external` or `stub`)
- `ALLOW_STUB_COMPRESSION`
- `QUANTIZE_EXTERNAL_CMD`
- `MERGE_LORA_EXTERNAL_CMD`
- `BENCHMARK_EXTERNAL_CMD`
- `LLAMA_CPP_DIR`
- `LLAMA_CPP_CONVERT_SCRIPT`
- `LLAMA_CPP_QUANTIZE_BIN`
- `ONNX_EXPORT_TASK`
- `PYTHON_EXECUTABLE`
- `EXTERNAL_COMMAND_TIMEOUT_SECONDS`

### Domain hook plugins

- `DOMAIN_HOOK_PLUGIN_MODULES` (JSON array of Python module paths)
- Example: `["app.plugins.domain_hooks.example_hooks"]`

### Data adapter plugins

- `DATA_ADAPTER_PLUGIN_MODULES` (JSON array of Python module paths)
- Example: `["app.plugins.data_adapters.example_adapters"]`

---

## External Runtime Command Templates

Example training command template:

```bash
TRAINING_BACKEND=external
TRAINING_EXTERNAL_CMD='python "{backend_dir}/scripts/train.py" --project {project_id} --experiment {experiment_id} --output "{output_dir}" --base-model "{base_model}" --config "{config_path}" --train-file "{train_file}" --val-file "{val_file}"'
```

`backend/scripts/train.py` performs real HuggingFace Trainer fine-tuning and expects prepared split files (for example `train.jsonl`).

Runtime plugin notes:

- Experiment config can set `training_runtime_id`; use `auto` to inherit server default runtime.
- Current built-ins:
  - `builtin.simulate`
  - `builtin.external_celery`
- Legacy mapping is preserved:
  - `TRAINING_BACKEND=simulate` -> `builtin.simulate`
  - `TRAINING_BACKEND=external` -> `builtin.external_celery`
- Custom runtimes can be loaded with `TRAINING_RUNTIME_PLUGIN_MODULES` and should expose:
  - `register_training_runtime_plugins(register_fn)`

Example compression template:

```bash
COMPRESSION_BACKEND=external
QUANTIZE_EXTERNAL_CMD='python "{backend_dir}/scripts/quantize.py" --project {project_id} --model "{model_path}" --bits {bits} --format {output_format} --out "{output_model_path}"'
```

If worker `PATH` does not include `python`, queued training/compression jobs automatically fall back to the active runtime interpreter (typically your backend `.venv` Python).

Real compression notes (`output_format=gguf|onnx`):

- This runtime now performs real conversion/quantization:
  1) HF model dir/id -> FP16 GGUF (`convert_hf_to_gguf.py`)
  2) FP16 GGUF -> quantized GGUF (`llama-quantize`)
  3) HF model dir/id -> ONNX -> INT8 ONNX (`optimum` + `onnxruntime`)
- Required tooling:
  - `llama.cpp` built locally (or binaries/scripts available in `PATH`)
  - `transformers` + `huggingface_hub` in backend env
  - for ONNX path: `optimum`, `onnx`, `onnxruntime` (or `onnxruntime-gpu`)
- Tool discovery (first match wins):
  - `LLAMA_CPP_CONVERT_SCRIPT` (absolute path to `convert_hf_to_gguf.py`)
  - `LLAMA_CPP_QUANTIZE_BIN` (absolute path to `llama-quantize`)
  - `LLAMA_CPP_DIR` (expects `convert_hf_to_gguf.py` and `build/bin/llama-quantize`)
- Optional:
  - `ONNX_EXPORT_TASK` (default `auto`, example: `text-generation-with-past`)
  - `PYTHON_EXECUTABLE` to force python used for conversion script
- If set in `backend/.env`, these values are injected into queued compression commands automatically.
- Supported formats in default runtime: `gguf`, `onnx`, and `merged` (LoRA merge).
- ONNX path currently supports `bits=8` (dynamic INT8 quantization).
- For other custom flows, provide your own command template.

Quick setup example:

```bash
git clone https://github.com/ggerganov/llama.cpp.git
cmake -S llama.cpp -B llama.cpp/build
cmake --build llama.cpp/build -j
export LLAMA_CPP_DIR="$(pwd)/llama.cpp"
pip install optimum onnx onnxruntime
```

`backend/scripts/benchmark.py` remains a pluggable external runtime you can replace with your own production command.

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
- `20260305_0004` domain packs + project binding
- `20260305_0005` artifact registry
- `20260305_0006` workflow run tracking tables
- `20260306_0007` project training preference persistence
- `20260306_0008` project dataset adapter preset persistence

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

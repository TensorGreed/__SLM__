# SLM Platform

> **Build, evaluate, compress, and export domain-specific Small Language Models (SLMs)** through a guided, production-oriented workflow.

A modular platform for ML engineers to go from arbitrary source data to releasable SLM artifacts with RBAC, audit logging, dataset normalization, training/eval orchestration, and export manifests.

---

## Quick Start

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows PowerShell
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173** in your browser.

---

## Architecture

```
__SLM__/
├── backend/
│   └── app/
│       ├── api/           # REST routers (auth, audit, ingestion, cleaning, training, eval, export, ...)
│       ├── models/        # SQLAlchemy ORM (project, dataset, experiment, export, auth)
│       ├── schemas/       # Pydantic request/response models
│       ├── services/      # Business logic and pipeline operations
│       ├── pipeline/      # Orchestrator + stage definitions
│       └── utils/         # Parsers, metrics, helpers
├── frontend/
│   └── src/
│       ├── components/    # Pipeline UI panels
│       ├── pages/         # Project list + detail
│       ├── stores/        # Zustand stores
│       └── api/           # Axios client
└── data/                  # Runtime data (projects, models, exports)
```

## Pipeline Modules

| Module | Description | Backend Service | API Routes |
|--------|-------------|-----------------|------------|
| **Data Ingestion** | Upload/import PDF, DOCX, TXT, MD, CSV, JSON, JSONL; remote HF/Kaggle/URL | `ingestion_service.py` | `/ingestion` |
| **Data Cleaning** | PII redaction, dedup, quality scoring, chunking | `cleaning_service.py` | `/cleaning` |
| **Gold Dataset** | Manual Q&A creation/import/lock for evaluation | `gold_service.py` | `/gold` |
| **Synthetic Gen** | Teacher model Q&A generation (incl. local providers) | `synthetic_service.py` | `/synthetic` |
| **Dataset Prep** | Normalize, combine, profile, split train/val/test + manifest | `dataset_service.py` | `/dataset` |
| **Tokenization** | Token stats + vocab inspection | `tokenization_service.py` | `/tokenization` |
| **Training** | SFT/LoRA experiment management + checkpoints | `training_service.py` | `/training` |
| **Evaluation** | Exact match, F1, safety, LLM-judge | `evaluation_service.py` | `/evaluation` |
| **Compression** | Quantization and model-size optimization | `compression_service.py` | `/compression` |
| **Export** | GGUF/ONNX/HF/Docker packaging + run manifests | `export_service.py` | `/export` |

---

## Production-Oriented Capabilities

### 1. Auth, RBAC, and Audit
- API-key based authentication with global roles: `admin`, `engineer`, `viewer`
- Project-level memberships: `owner`, `editor`, `viewer`
- Project-scoped authorization checks across API routes
- HTTP audit logging for mutating API calls

### 2. Generic Dataset Normalization
- Works with heterogeneous source schemas from any domain
- Optional `field_mapping` to map custom schema fields into canonical training keys
- Heuristic fallback normalization for unknown schemas
- Coverage diagnostics and dropped-record visibility during import/profile

### 3. Dataset Profiling Before Training
- `POST /api/projects/{project_id}/dataset/profile`
- Inspect top fields, normalization coverage, text-length distribution, and sample previews
- Profile raw documents directly or prepared datasets

### 4. Evaluation with Deterministic + Provider Judge Modes
- LLM-judge path supports:
  - deterministic local fallback scoring
  - optional OpenAI-compatible remote judge endpoint via env config
- Bounded prediction payload handling and explicit request schema validation

### 5. Export Run Manifests for Release
- Each export run produces a versioned `run-<timestamp>` directory
- Manifest includes:
  - run metadata
  - experiment summary
  - artifact checksums (SHA-256)
- Root export manifest tracks latest run for backward compatibility

---

## Environment Configuration

Common backend environment variables:

- `DATABASE_URL` (default: `sqlite+aiosqlite:///./data/slm_platform.db`)
- `AUTH_ENABLED` (`true`/`false`)
- `AUTH_BOOTSTRAP_API_KEY`
- `AUTH_BOOTSTRAP_USERNAME` (default: `admin`)
- `AUTH_BOOTSTRAP_ROLE` (default: `admin`)
- `AUDIT_LOG_ENABLED` (`true`/`false`)
- `JUDGE_MODEL_API_URL` (optional, OpenAI-compatible endpoint)
- `JUDGE_MODEL_API_KEY` (optional bearer token)

When `AUTH_ENABLED=true`, set a bootstrap API key and call secured APIs with:
- `x-api-key: <your-key>`

---

## Testing

```bash
cd backend
.venv/bin/python -m unittest discover -s tests -v
```

---

## API Documentation

With backend running, visit:
- **http://localhost:8000/docs**

---

## License

MIT

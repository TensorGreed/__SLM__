# SLM Platform

> **Build, evaluate, compress, and export domain-specific Small Language Models (SLMs)** through a guided UI-driven workflow.

A modular platform enabling ML engineers and domain teams to take raw data all the way to a production-ready, quantized small language model — with full evaluation, safety testing, and version tracking.

---

## Quick Start

### Backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate       # Windows
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
├── backend/               # Python / FastAPI
│   └── app/
│       ├── api/           # REST endpoints (10 routers)
│       ├── models/        # SQLAlchemy ORM (Project, Dataset, Experiment, Export)
│       ├── schemas/       # Pydantic request/response models
│       ├── services/      # Business logic (8 service modules)
│       ├── pipeline/      # Orchestrator + stage definitions
│       └── utils/         # File parsers, PII detection, metrics
├── frontend/              # React 18 + Vite + TypeScript
│   └── src/
│       ├── components/    # Module panels (8 tabs)
│       ├── pages/         # Project list + detail
│       ├── stores/        # Zustand state management
│       └── api/           # Axios client
└── data/                  # Runtime data (projects, models, exports)
```

## Pipeline Modules

| Module | Description | Backend Service | API Routes |
|--------|-------------|-----------------|------------|
| **Data Ingestion** | Upload PDF, DOCX, TXT, MD, CSV | `ingestion_service.py` | `/ingestion` |
| **Data Cleaning** | PII redaction, dedup, quality scoring, chunking | `cleaning_service.py` | `/cleaning` |
| **Gold Dataset** | Manual Q&A creation, import, lock | `gold_service.py` | `/gold` |
| **Synthetic Gen** | Teacher model Q&A generation with confidence scoring | `synthetic_service.py` | `/synthetic` |
| **Dataset Prep** | Combine, split train/val/test, manifest | `dataset_service.py` | — |
| **Tokenization** | Token stats, vocabulary inspection | `tokenization_service.py` | — |
| **Training** | SFT/LoRA experiment management | `training_service.py` | `/training` |
| **Evaluation** | Exact match, F1, safety tests, scorecard | `evaluation_service.py` | `/evaluation` |
| **Compression** | 4/8-bit quantization, LoRA merge, benchmarks | `compression_service.py` | `/compression` |
| **Export** | GGUF/ONNX/HF/Docker packaging with manifest | `export_service.py` | `/export` |

## Tech Stack

- **Backend**: Python 3.11+ / FastAPI / SQLAlchemy / Celery
- **Frontend**: React 18 / Vite / TypeScript / Zustand / Recharts
- **ML**: HuggingFace Transformers + PEFT + bitsandbytes
- **Database**: SQLite (dev) / PostgreSQL (prod)

## API Documentation

With the backend running, visit **http://localhost:8000/docs** for interactive Swagger UI.

## License

MIT
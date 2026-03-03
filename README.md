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
| **Gold Dataset** | Manual Q&A creation (e.g., Q: "How to reset auth?", A: "Call `reset()`"), import, lock | `gold_service.py` | `/gold` |
| **Synthetic Gen** | Teacher model Q&A generation (Supports local LLMs via Ollama) | `synthetic_service.py` | `/synthetic` |
| **Dataset Prep** | Combine, split train/val/test, manifest | `dataset_service.py` | — |
| **Tokenization** | Token stats, vocabulary inspection | `tokenization_service.py` | — |
| **Training** | SFT/LoRA experiment management | `training_service.py` | `/training` |
| **Evaluation** | Exact match, F1, safety tests, scorecard | `evaluation_service.py` | `/evaluation` |
| **Compression** | 4/8-bit quantization, LoRA merge, benchmarks | `compression_service.py` | `/compression` |
| **Export** | GGUF/ONNX/HF/Docker packaging with manifest | `export_service.py` | `/export` |

### Understanding Core Data Concepts

#### 1. Gold Evaluation Dataset
The Gold Dataset is your **ground-truth evaluation set**. It consists of hand-curated, high-quality Q&A pairs used *only* for testing the model's performance, never for training.

**Example Usage**:
If building a customer support SLM, your Gold Set should contain exact expert answers:
- **Question (Prompt)**: *"What is the refund policy for annual subscriptions cancelled after 30 days?"*
- **Gold Answer (Reference)**: *"Annual subscriptions cancelled after 30 days are non-refundable, but the account remains active until the end of the billing cycle. (See Terms section 4.2)"*

During the Evaluation stage, the SLM's generated answer will be compared against this Gold Answer using metrics like ROUGE or an LLM-as-a-Judge.

#### 2. Synthetic Data Generation
When you don't have enough manual training data, you can use a larger "Teacher" model to automatically generate Q&A pairs from your raw documents. 

**Ollama Support**: The platform supports local inference via **Ollama**. If you have a local Llama-3 or Mixtral model running on `localhost:11434`, you can use it as the Teacher model without paying API costs or sending data to the cloud.

**How it works**:
1. **Source Text**: The platform feeds raw document chunks to the Teacher. 
   *(Example source: "The HTTP 429 Too Many Requests response status code indicates the user has sent too many requests in a given amount of time ('rate limiting').")*
2. **Teacher Output**: The Teacher generates training pairs.
   *(Example generated pair -> Question: "What does an HTTP 429 status code mean?", Answer: "It means Too Many Requests, indicating the user has hit a rate limit.")*

## Tech Stack

- **Backend**: Python 3.11+ / FastAPI / SQLAlchemy / Celery
- **Frontend**: React 18 / Vite / TypeScript / Zustand / Recharts
- **ML**: HuggingFace Transformers + PEFT + bitsandbytes
- **Database**: SQLite (dev) / PostgreSQL (prod)

## API Documentation

With the backend running, visit **http://localhost:8000/docs** for interactive Swagger UI.

## License

MIT
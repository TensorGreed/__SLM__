# Dataset And Pipeline Map

Discovery date: 2026-05-19.

This map records what the repo actually implements or declares. It does not
assume a standard ML lifecycle where the repo has no evidence.

Status legend: verified, partial, simulated, estimated, conceptual, unsupported,
unknown.

## Official Demo Seeder

Status: verified.

Evidence:

- `backend/app/services/demo_project_service.py`
- `backend/app/api/demo_projects.py`
- `frontend/src/components/dashboard/DemoProjectTiles.tsx`

When `POST /api/demo-projects/{slug}` is called, the backend:

1. Resolves `backend/data/demo_samples/{slug}`.
2. Reads `manifest.json`.
3. Creates or reuses an active project by sample name.
4. Sets `pipeline_stage=PipelineStage.TRAINING`.
5. Stores `task_profile`, target profile, training/eval preferences, and
   `dataset_adapter_preset` from the manifest.
6. Copies the sample CSV into project raw data.
7. Creates a RAW dataset and one `RawDocument` per CSV row.
8. Reads `gold.jsonl`.
9. Writes a legacy `gold_dev.jsonl`.
10. Creates a locked `GOLD_DEV` dataset, a locked `GoldSetVersion`, and
    approved `GoldSetRow` records.
11. Writes canonical prepared `train.jsonl`, `val.jsonl`, `test.jsonl`, and
    prepared `manifest.json`.
12. Creates TRAIN, VALIDATION, and TEST dataset records and versions.

Important recording implication:

- Seeded demos do not start as empty projects. They already contain raw rows,
  gold rows, and prepared splits. A teaching demo should clearly distinguish
  "what the official sample seeds for you" from "how the UI can run import,
  cleaning, synthetic generation, and split preparation manually."

## Official Sample Sizes And Seeded Splits

Counts are from the sample files and the seeder's deterministic 70/15/15 split
logic with minimum validation/test rows where possible.

| Sample | Source rows | Gold rows | Seeded train | Seeded val | Seeded test | Status |
|---|---:|---:|---:|---:|---:|---|
| `support-faq` | 20 | 200 | 16 | 2 | 2 | verified from files/code |
| `pii-detector` | 61 | 200 | 45 | 8 | 8 | verified from files/code |
| `sentiment-classifier` | 30 | 200 | 22 | 4 | 4 | verified from files/code |

Open documentation mismatch:

- `pii-detector/manifest.json` describes 60 snippets, but the CSV has 61 data
  rows.
- `support-faq/manifest.json` and `sentiment-classifier/manifest.json` mention
  smaller gold-set counts than the current 200-row `gold.jsonl` files.

## Sample To Adapter Mapping

Status: verified.

Evidence: `_adapter_for_task` in `backend/app/services/demo_project_service.py`.

| Manifest task profile | Adapter id used by seeder | Samples |
|---|---|---|
| `classification` | `classification-label` | `sentiment-classifier` |
| `structured_extraction` or `extraction` | `structured-extraction` | `pii-detector` |
| Any other task profile | `qa-pair` | `support-faq` (`instruction_sft`) |

Seeder canonical row behavior:

- All prepared rows include `text`, `source_text`, and `target_text`.
- Classification rows also include `label`.
- Non-classification rows are shaped with `question` and `answer`.
- PII rows preserve structured output schema and entity types in the prepared
  manifest.

## Frontend Route And Pipeline Map

Status: verified.

Evidence:

- `frontend/src/App.tsx`
- `frontend/src/pages/ProjectWorkspaceLayout.tsx`
- `frontend/src/pages/ProjectPipelinePage.tsx`

Project route root: `/project/:id`.

| Route/tab | Frontend component | Main backend surface | Recording status |
|---|---|---|---|
| `/project/:id/pipeline/data` | `IngestionPanel` | `/api/projects/{id}/ingestion/*`, `/api/projects/{id}/dataset-import/*` | verified surface, selector pass needed |
| `/project/:id/pipeline/cleaning` | `CleaningPanel` | `/api/projects/{id}/cleaning/*` | verified surface |
| `/project/:id/pipeline/goldset` | `GoldSetPanel` | `/api/projects/{id}/gold/*` | verified surface |
| `/project/:id/pipeline/synthetic` | `SyntheticPanel` | `/api/projects/{id}/synthetic/*` | partial runtime |
| `/project/:id/pipeline/dataprep` | `DatasetPrepPanel` | `/api/projects/{id}/dataset/*` | verified surface |
| `/project/:id/pipeline/tokenization` | `TokenizationPanel` | `/api/projects/{id}/tokenization/*` | partial runtime due tokenizer deps |
| `/project/:id/pipeline/training` | `TrainingPanel` | `/api/projects/{id}/training/*` | partial runtime |
| `/project/:id/training-config` | `ProjectTrainingConfigPage` | `/api/projects/{id}/training/*` | verified config surface |
| `/project/:id/pipeline/eval` | `EvalPanel` | `/api/projects/{id}/evaluation/*` | partial runtime/artifact |
| `/project/:id/pipeline/compression` | `CompressionPanel` | `/api/projects/{id}/compression/*` | partial runtime |
| `/project/:id/pipeline/export` | `ExportPanel` | `/api/projects/{id}/export/*`, `/api/projects/{id}/registry/*` | partial artifact/runtime |
| `/project/:id/playground` | `ProjectPlaygroundPage`, `ChatPlaygroundPanel` | training playground endpoints | partial final-model usage |
| `/project/:id/deployments` | `ProjectDeploymentsPage` | deployment APIs | partial telemetry/deployment data |

## Manual Dataset Ingestion And Import

Status: verified for surfaces, partial for external sources.

Evidence:

- `backend/app/api/ingestion.py`
- `frontend/src/components/data/IngestionPanel.tsx`
- `backend/app/api/dataset_import.py`
- `backend/app/services/dataset_import/service.py`
- `backend/app/services/dataset_import/sources/*`
- `backend/app/services/dataset_import/mappers/*`
- `frontend/src/components/data/DatasetImportWizard.tsx`

Ingestion supports:

- Upload single file.
- Upload batch.
- Remote inspect/import/queue.
- Import task status/cancel/logs.
- Document list and document sample.
- EDA and outlier removal endpoint.
- Per-document process/delete.

Dataset import wizard supports:

- Source catalog and mapper catalog.
- Introspection.
- Preview with accepted/rejected rows.
- Run with saved configs.
- Built-in mappers including text-only, Q&A, classification labels, chat
  messages, preferences, RAG, BIO spans, and key-value structured mapping.

Important behavior:

- `backend/app/services/dataset_import/service.py` persists accepted transformed
  rows to a synthetic dataset file (`synthetic.jsonl`) with `source=dataset_import`.
  It is not the same path as raw file ingestion.

External/heavy dependencies:

- Hugging Face sources need the `datasets` package and model/data access.
- Kaggle sources need Kaggle package/credentials/cache.
- Queued remote import requires Redis/Celery.

## Cleaning, PII Handling, And Deduplication

Status: verified for cleaning/PII regex, partial for deduplication.

Evidence:

- `backend/app/services/cleaning_service.py`
- `backend/app/api/cleaning.py`
- `frontend/src/components/data/CleaningPanel.tsx`
- `backend/app/services/gold_workbench_service.py`
- `backend/app/services/dataset_service.py`
- `slm-docs/docs/workflows/data-ingestion.md`

Verified cleaning behavior:

- Materializes extracted text from structured rows.
- Removes boilerplate by regex.
- Detects PII patterns such as email, phone, SSN, credit card, IP address,
  API key, and AWS key.
- Optionally redacts detected PII as `[REDACTED_TYPE]`.
- Detects and optionally masks toxicity spans.
- Computes a quality score.
- Computes `text_hash`.
- Chunks text.
- Writes cleaned text, chunk JSONL, and project-level `cleaned.jsonl`.
- Updates raw document metadata with PII/toxicity/chunk/hash information.

Partial/unsupported dedup evidence:

- Cleaning computes a normalized text hash, but this pass did not find code that
  removes duplicate rows during cleaning.
- Gold workbench sampling skips duplicate source-row keys.
- Dataset semantic analysis surfaces can report redundancy/diversity.
- `slm-docs/docs/workflows/data-ingestion.md` describes cleaning-stage
  deduplication, but code evidence for automatic row removal was not verified.

Recording guidance:

- Show PII redaction as a cleaning option only.
- Show PII detector as a model task only.
- Do not conflate cleaning redaction with training a PII detector.
- Mark duplicate removal as partial/unknown unless a later run proves it.

## Dataset Preparation, Normalization, And Splits

Status: verified.

Evidence:

- `backend/app/api/dataset.py`
- `backend/app/services/dataset_service.py`
- `frontend/src/components/data/DatasetPrepPanel.tsx`
- `backend/app/services/demo_project_service.py`

Verified behavior:

- Adapter preference and auto-detection APIs exist.
- Adapter catalog, preview, mapping acceptance, and profile endpoints exist.
- Dataset preview exists.
- Semantic intelligence analysis endpoint exists.
- `split_dataset()` combines selected dataset types, normalizes rows through the
  chosen adapter, applies optional domain normalizer hooks, shuffles with seed,
  writes train/val/test JSONL, creates/updates datasets and versions, and writes
  a prepared manifest.
- If cleaned and synthetic rows are both requested and synthetic rows exist,
  `resolve_training_dataset_types()` may auto-exclude cleaned rows.
- If expected prepared inputs are empty, the service can fall back to RAW rows.

Recording guidance:

- Browser can show adapter preview, profile, semantic analysis, split settings,
  and prepared dataset creation.
- For official seeded demos, prepared splits already exist before manual split.

## Runtime And Job Map

| Area | Status | Evidence | Runtime model |
|---|---|---|---|
| Cleaning async tasks | verified | `backend/app/api/cleaning.py`, `backend/app/services/cleaning_service.py` | In-process async registry |
| Synthetic span async tasks | partial | `backend/app/api/synthetic.py`, `backend/app/services/synthetic_service.py` | In-process async registry, teacher/fallback dependent |
| Remote import queued jobs | partial | `backend/app/api/ingestion.py`, README | Redis/Celery |
| Training | partial | `backend/app/services/training_runtime_service.py`, `backend/scripts/train.py` | Default external Celery runtime; simulated runtime disabled unless explicitly allowed |
| Evaluation | partial | `backend/app/services/evaluation_service.py` | Local/remote inference or supplied predictions; judge model optional/path-dependent |
| Compression | partial | `backend/app/services/compression_service.py`, `backend/scripts/quantize.py` | Default external Celery runtime; stub disabled unless explicitly allowed |
| Export | partial | `backend/app/services/export_service.py` | Requires real artifacts; can package and validate target profiles |
| Serve runs | partial | `backend/app/services/serve_runtime_service.py` | Local subprocess with runtime-specific command |

## Pipeline Stage Evidence Summary

- Project setup: verified.
- Sample import: verified.
- Dataset ingestion: verified.
- Raw inspection: verified surface.
- Cleaning: verified.
- Normalization: verified through adapter/prep services, but not as one single
  named UI button.
- Deduplication: partial.
- PII handling: verified for cleaning redaction, partial for model pipeline.
- Gold set: verified.
- Synthetic generation: partial runtime.
- Dataset prep and splits: verified.
- Tokenization: verified surface, heavy tokenizer runtime.
- Training: partial runtime.
- Evaluation/gates: partial runtime/artifact.
- Compression/export/registry/final usage: partial runtime/artifact.

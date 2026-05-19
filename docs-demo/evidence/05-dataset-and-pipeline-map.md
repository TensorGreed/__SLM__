# Dataset And Pipeline Map

## Official Demo Seeder

Evidence: `backend/app/services/demo_project_service.py`.

When `POST /api/demo-projects/{slug}` is called, the seeder:
- Resolves `backend/data/demo_samples/{slug}`.
- Reads `manifest.json`.
- Creates an active project.
- Sets `pipeline_stage=TRAINING`.
- Stores `dataset_adapter_preset` with demo slug, suggested brief, adapter id, task profile, and field mapping.
- Copies the sample CSV into project raw data.
- Creates a RAW dataset and one RawDocument per CSV row.
- Reads `gold.jsonl`.
- Writes a legacy gold JSONL file under project data.
- Creates a GOLD_DEV dataset, a locked GoldSetVersion, and approved GoldSetRow records.
- Writes canonical prepared `train.jsonl`, `val.jsonl`, `test.jsonl`, and a prepared `manifest.json`.
- Creates TRAIN, VALIDATION, and TEST dataset records.

## Sample To Adapter Mapping

Evidence: `_adapter_for_task` in `backend/app/services/demo_project_service.py`.

| Task profile | Adapter id |
|---|---|
| `classification` | `classification-label` |
| `structured_extraction` or `extraction` | `structured-extraction` |
| Other | `qa-pair` |

## Pipeline Tabs

Evidence: `frontend/src/pages/ProjectPipelinePage.tsx`.

| Tab | Component | Backend surface |
|---|---|---|
| data | `IngestionPanel` | `/projects/{project_id}/ingestion/*` |
| cleaning | `CleaningPanel` | `/projects/{project_id}/cleaning/*` |
| goldset | `GoldSetPanel` | `/projects/{project_id}/gold/*` |
| synthetic | `SyntheticPanel` | `/projects/{project_id}/synthetic/*` |
| dataprep | `DatasetPrepPanel` | `/projects/{project_id}/dataset/*` |
| tokenization | `TokenizationPanel` | `/projects/{project_id}/tokenization/*` |
| training | `TrainingPanel` plus link to Training Config | `/projects/{project_id}/training/*` |
| eval | `EvalPanel` | `/projects/{project_id}/evaluation/*` |
| compression | `CompressionPanel` | `/projects/{project_id}/compression/*` |
| export | `ExportPanel` | `/projects/{project_id}/export/*`, `/projects/{project_id}/registry/*` |

## Discovery Notes

- Demo seeding bypasses a manually recorded import/clean/split path by pre-creating prepared splits. For full teaching demos, record both "what the seed gives you" and "how the normal UI supports doing it yourself" only after running the UI.
- Pipeline stage is set to training for seeded demos, likely to unlock later tabs. Verify how this affects beginner walkthroughs.


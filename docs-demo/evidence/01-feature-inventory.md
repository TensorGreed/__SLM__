# Feature Inventory

Status legend: verified, partial, simulated, estimated, conceptual, unknown.

## Verified Repo Evidence

| Feature area | Evidence | Status | Notes |
|---|---|---|---|
| Official demo catalog | `backend/app/services/demo_project_service.py`, `backend/app/api/demo_projects.py`, `frontend/src/components/dashboard/DemoProjectTiles.tsx` | verified | UI lists demos from `backend/data/demo_samples` and POSTs `/api/demo-projects/{slug}`. |
| Official samples | `backend/data/demo_samples/pii-detector`, `backend/data/demo_samples/sentiment-classifier`, `backend/data/demo_samples/support-faq` | verified | Each sample has `manifest.json`, a CSV source file, and `gold.jsonl`. |
| Demo seeding | `backend/app/services/demo_project_service.py` | verified | Seeder creates project, raw dataset, raw document rows, locked gold set, and prepared train/val/test files. |
| Pipeline UI tabs | `frontend/src/pages/ProjectPipelinePage.tsx` | verified | Tabs: data, cleaning, goldset, synthetic, dataprep, tokenization, training, eval, compression, export. |
| Dataset import | `frontend/src/components/data/IngestionPanel.tsx`, `frontend/src/components/data/DatasetImportWizard.tsx`, `backend/app/api/ingestion.py` | verified | Upload, batch upload, remote import inspection, queued remote import, document sample, and EDA endpoints exist. |
| Cleaning | `frontend/src/components/data/CleaningPanel.tsx`, `backend/app/api/cleaning.py` | verified | UI starts `clean-batch-async` and polls tasks. |
| Gold sets | `frontend/src/components/data/GoldSetPanel.tsx`, `frontend/src/components/evaluation/GoldSetWorkbenchPanel.tsx`, `backend/app/api/gold.py`, `backend/app/services/gold_workbench_service.py` | partial | Legacy gold tab and newer workbench exist. Demo seeder locks gold rows. Recording details need route-level UI pass. |
| Synthetic generation | `frontend/src/components/data/SyntheticPanel.tsx`, `backend/app/api/synthetic.py`, `backend/app/services/synthetic_service.py` | verified | Q&A, conversation, and span-generation endpoints exist. Demo fallback is controlled by env. |
| Dataset preparation | `frontend/src/components/data/DatasetPrepPanel.tsx`, `backend/app/api/dataset.py` | verified | Split, preview, profile, semantic analysis, adapter catalog, auto-detect, and adapter preview exist. |
| Tokenization | `frontend/src/components/training/TokenizationPanel.tsx`, `backend/app/api/tokenization.py` | verified | UI can analyze tokens and fetch vocab sample. |
| Training config and runs | `frontend/src/pages/ProjectTrainingConfigPage.tsx`, `frontend/src/components/training/TrainingPanel.tsx`, `backend/app/api/training.py`, `backend/app/services/training_runtime_service.py` | partial | Full UI/API surface exists. Real vs simulated runtime depends on env. |
| Evaluation and gates | `frontend/src/components/evaluation/EvalPanel.tsx`, `backend/app/api/evaluation.py`, `backend/app/services/evaluation_pack_service.py` | partial | Held-out eval, scorecard, gates, safety scorecard, remediation endpoints exist. Needs run-through. |
| Compression | `frontend/src/components/compression/CompressionPanel.tsx`, `backend/app/api/compression.py`, `backend/app/services/compression_service.py` | partial | Quantize, merge, benchmark endpoints exist. Real or stub depends on runtime env. |
| Export | `frontend/src/components/export/ExportPanel.tsx`, `backend/app/api/export.py`, `backend/app/services/export_service.py` | partial | Export, deploy validation, serve plan, serve runs, optimization matrix exist. Needs run-through. |
| Registry | `frontend/src/components/export/ExportPanel.tsx`, `backend/app/api/registry.py`, `backend/app/services/registry_service.py` | partial | Export UI can register/promote/deploy models. Needs run-through. |
| Final usage/testing | `frontend/src/pages/ProjectPlaygroundPage.tsx`, `frontend/src/components/training/ChatPlaygroundPanel.tsx`, `backend/app/api/training.py` playground endpoints, `backend/app/api/export.py` serve endpoints | partial | Playground and local serve plans exist; end-to-end trained model usage needs verification. |

## To Verify

- Exact visual path after local login, because frontend currently redirects to `/login` until `slm_token` is set in local storage.
- Whether demo seeded projects begin at `training` pipeline stage but still allow earlier tabs to be shown clearly.
- Which training/compression/export paths run locally without GPU, Redis worker, teacher model, or external tools.
- Which API routes are most stable for narrated demos versus UI-only flows.


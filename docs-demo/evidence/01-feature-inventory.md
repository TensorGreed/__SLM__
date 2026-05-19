# Feature Inventory

Discovery date: 2026-05-19.
Last cross-validation: 2026-05-19 (repo audit agent).

## Cross-validation log

The 2026-05-19 repo audit confirmed every "verified" row in this file
against the cited evidence paths. Spot checks:

- `PIPELINE_TABS` in `frontend/src/types/index.ts:597-609` matches the
  ten tab keys claimed across this file (`data`, `cleaning`, `goldset`,
  `synthetic`, `dataprep`, `tokenization`, `training`, `eval`,
  `compression`, `export`).
- 70/15/15 split ratio confirmed in
  `backend/app/services/demo_project_service.py:465`.
- Demo-project routes confirmed in `backend/app/api/demo_projects.py:43,48`
  (`GET /api/demo-projects` and `POST /api/demo-projects/{slug}`).
- Local-login password gate confirmed in
  `backend/app/api/auth.py:168-169` against `settings.API_KEY`; default
  `sk-mock-admin-key` confirmed in `backend/.env.example:18`.
- Demo sample file counts confirmed: support-faq tickets.csv = 20 data
  rows, pii-detector pii_records.csv = 61 data rows, sentiment-classifier
  reviews.csv = 30 data rows, all three `gold.jsonl` = 200 rows.

No "verified" rows have been downgraded. No new "unsupported" rows
discovered. The single remaining manifest/gold-count documentation
mismatch ("Manifest prose mentions … gold rows while gold.jsonl has
200") is correctly captured in `10-open-questions.md` Q2.

This inventory is based on repo files only. No product behavior was changed, no
database seeding was performed during this pass, and no recording scripts were
created from these findings.

Status legend:

- verified: repo code/data directly implements or declares the feature.
- partial: surface exists, but a real run depends on runtime setup, artifacts, or
  a later UI pass.
- simulated: repo has an explicit fake/demo/mock mode.
- estimated: repo labels values as estimated or derives them without a measured
  run.
- conceptual: useful explanation, but not product-specific evidence.
- unsupported: repo evidence points away from the feature.
- unknown: not found or not resolved yet.

## Product Surface Inventory

| Feature area | Status | Frontend evidence | Backend/service evidence | Runtime/external dependencies | Demo evidence and what can be recorded |
|---|---|---|---|---|---|
| Official demo catalog and one-click seed | verified | `frontend/src/components/dashboard/DemoProjectTiles.tsx` | `backend/app/api/demo_projects.py`, `backend/app/services/demo_project_service.py` | Running backend/frontend and auth token | Browser can show the three official tiles and project creation. API can call `GET /api/demo-projects` and `POST /api/demo-projects/{slug}`. |
| Official demo samples | verified | Loaded through dashboard tiles | `backend/data/demo_samples/pii-detector`, `backend/data/demo_samples/sentiment-classifier`, `backend/data/demo_samples/support-faq` | None beyond file access | Browser can seed only these three official samples. Do not invent other sample templates. |
| Local auth and startup | verified | `frontend/src/pages/SSOLoginPage.tsx`, `frontend/src/api/client.ts` | `README.md`, `backend/app/config.py`, `backend/.env.example` | Auth is enabled by default; local password is `API_KEY` unless env changes | Demo can show login, or Playwright can set an auth token later after selector discovery. |
| Project workspace and pipeline tabs | verified | `frontend/src/App.tsx`, `frontend/src/pages/ProjectWorkspaceLayout.tsx`, `frontend/src/pages/ProjectPipelinePage.tsx` | Project APIs mounted from `backend/app/main.py` | Running app | Browser can show tabs: data, cleaning, goldset, synthetic, dataprep, tokenization, training, eval, compression, export. |
| Dataset upload and remote ingestion | verified | `frontend/src/components/data/IngestionPanel.tsx` | `backend/app/api/ingestion.py` | File upload is local; remote/HF/Kaggle/queued imports may need packages, credentials, Redis/Celery | Browser can show upload, remote source inspection/import, documents, samples, EDA, and import logs. |
| Generic dataset import wizard | verified | `frontend/src/components/data/DatasetImportWizard.tsx`, `frontend/src/api/datasetImport.ts` | `backend/app/api/dataset_import.py`, `backend/app/services/dataset_import/service.py` | HF/Kaggle sources need optional deps/credentials; run writes accepted rows to a synthetic dataset | Browser can show source selection, mapper selection, preview, confidence, and run. Narration must say this path lands transformed rows in synthetic output, not raw ingestion. |
| Raw data inspection | verified | `frontend/src/components/data/IngestionPanel.tsx`, `frontend/src/components/data/DocumentSampleAccordion.tsx` | `backend/app/api/ingestion.py` | None for seeded rows | Browser can show document list and sample rows after seeding/import. Durable selectors still need a later UI pass. |
| Cleaning, chunking, quality scoring | verified | `frontend/src/components/data/CleaningPanel.tsx` | `backend/app/api/cleaning.py`, `backend/app/services/cleaning_service.py` | None for local regex/text cleanup; async task registry is in-process | Browser can choose chunk size, PII/secrets redaction, toxicity masking, start async cleaning, and show quality/chunk/PII/toxicity results. |
| Regex PII redaction during cleaning | verified | `frontend/src/components/data/CleaningPanel.tsx` | `backend/app/services/cleaning_service.py` | Regex patterns only | Browser can show cleaning options and results. Keep this distinct from the `pii-detector` model sample, which is a structured extraction task. |
| Deduplication | partial | Gold workbench and dataset prep surfaces mention duplicate/redundancy concepts | `backend/app/services/cleaning_service.py`, `backend/app/services/gold_workbench_service.py`, `backend/app/services/dataset_service.py` | Semantic analysis may need embeddings/vector dependencies | Code computes `text_hash`, gold sampling skips duplicate source-row keys, and semantic analysis exists. A cleaning-stage duplicate removal action was not verified. |
| Gold set creation/import | verified | `frontend/src/components/data/GoldSetPanel.tsx`, `frontend/src/components/evaluation/GoldSetWorkbenchPanel.tsx` | `backend/app/api/gold.py`, `backend/app/services/gold_service.py`, `backend/app/services/gold_workbench_service.py` | None for local rows | Browser can show legacy gold rows, lock/import actions, and workbench sampling/review. Seeder creates locked approved gold rows for official samples. |
| Synthetic generation | partial | `frontend/src/components/data/SyntheticPanel.tsx` | `backend/app/api/synthetic.py`, `backend/app/services/synthetic_service.py` | Teacher model URL/API key, or explicit `ALLOW_SYNTHETIC_DEMO_FALLBACK=true` for fallback mode | Browser can show Q&A, conversation, and PII/NER span modes. Real generated content requires a teacher model or clearly labeled fallback. |
| Dataset preparation, adapters, normalization | verified | `frontend/src/components/data/DatasetPrepPanel.tsx` | `backend/app/api/dataset.py`, `backend/app/services/dataset_service.py` | Domain normalizer hooks may depend on selected profile | Browser can show adapter catalog/preview, profile, semantic analysis, split config, and prepared dataset preview. |
| Train/validation/test splits | verified | `frontend/src/components/data/DatasetPrepPanel.tsx` | `backend/app/services/demo_project_service.py`, `backend/app/services/dataset_service.py` | None | Seeder creates deterministic prepared split files. Manual split endpoint can reshuffle with seed/ratios. |
| Tokenization | verified | `frontend/src/components/training/TokenizationPanel.tsx` | `backend/app/api/tokenization.py`, `backend/app/services/tokenization_service.py` | `transformers`, tokenizer downloads/cache, and any gated model access | Browser can show tokenizer preset selection, split selection, length stats, histogram, and vocab sample if tokenizer loads. |
| Training configuration and preflight | verified | `frontend/src/pages/ProjectTrainingConfigPage.tsx`, `frontend/src/components/training/TrainingPanel.tsx` | `backend/app/api/training.py`, `backend/app/services/training_service.py` | Depends on runtime choice and prepared data | Browser can show runtimes, recipes/config, preflight, effective config, and start controls. |
| Training run and monitoring | partial | `frontend/src/components/training/TrainingPanel.tsx`, observability pages | `backend/app/services/training_runtime_service.py`, `backend/scripts/train.py`, Celery task wiring | Default backend is external; likely needs Redis/Celery, HF/torch stack, and model access. Simulation is off by default unless `ALLOW_SIMULATED_TRAINING=true`. | Browser can show monitor/logs only after a configured run. Any simulated path must be labeled simulated. |
| Evaluation, scorecards, gates | partial | `frontend/src/components/evaluation/EvalPanel.tsx` | `backend/app/api/evaluation.py`, `backend/app/services/evaluation_service.py`, `backend/app/services/evaluation_pack_service.py`, `backend/app/services/eval_task_handler_service.py` | Held-out model eval needs trained/exported model or local/remote inference. LLM judge needs local/remote judge or fallback path. | Browser can show workbench, eval controls, results, scorecards, and gates after a runnable model/prediction source exists. |
| Compression | partial | `frontend/src/components/compression/CompressionPanel.tsx` | `backend/app/api/compression.py`, `backend/app/services/compression_service.py`, `backend/scripts/quantize.py`, `backend/scripts/benchmark.py` | Default external backend, Redis/Celery, llama.cpp/optimum/onnxruntime/transformers as needed. Stub is off by default unless `ALLOW_STUB_COMPRESSION=true`. | Browser can show quantize/merge/benchmark surfaces. A real compressed artifact requires heavy runtime setup. |
| Export and packaging | partial | `frontend/src/components/export/ExportPanel.tsx` | `backend/app/api/export.py`, `backend/app/services/export_service.py`, `backend/app/models/export.py` | Requires completed training artifacts, and compressed artifacts for quantized formats | Browser can create/run exports after an artifact exists. Supported enum values are `gguf`, `onnx`, `tensorrt`, `huggingface`, `docker`. |
| Deployment target validation | partial | `frontend/src/components/export/ExportPanel.tsx` | `backend/app/services/deployment_target_service.py` | Depends on artifact format and local tools such as Docker, vLLM, TGI, Ollama, llama.cpp, ONNX Runtime, or TensorRT | Browser can show deploy/SDK/serve plans after export. Real smoke tests depend on installed runtimes. |
| Model registry | partial | `frontend/src/components/export/ExportPanel.tsx` | `backend/app/api/registry.py`, `backend/app/services/registry_service.py` | Requires completed experiment and usually export/eval data | Browser can register, promote, and mark deployment metadata after artifacts exist. Promotion gates can block production. |
| Final model testing in UI | partial | `frontend/src/pages/ProjectPlaygroundPage.tsx`, `frontend/src/components/training/ChatPlaygroundPanel.tsx` | Playground endpoints in `backend/app/api/training.py`, `backend/app/services/playground_service.py`, `backend/app/services/playground_session_service.py` | Mock provider is simulated; real calls need OpenAI-compatible, llama.cpp, Ollama/vLLM/TGI, or other endpoint | Browser can show prompt playground and feedback logging. A real trained model usage demo needs an artifact/serve endpoint. |
| Final model API/runtime usage | partial | Export panel serve-run controls | `backend/app/services/serve_service.py`, `backend/app/services/serve_runtime_service.py`, export/registry serve endpoints | Exported artifact and local runtime command availability | Demo can show generated curl smoke tests and optionally start a local serve subprocess after export. |
| Deployment telemetry/drift | partial | `frontend/src/pages/ProjectDeploymentsPage.tsx` | `backend/app/api/deployments.py` | Telemetry must be pushed/generated | Browser can show deployment versions, telemetry summaries, drift checks, and score APIs after data exists. |

## Official Sample Task Inventory

| Sample | Status | Manifest task profile | Target profile | Source file | Gold file | Main verified story |
|---|---|---|---|---|---|---|
| `support-faq` | partial | `instruction_sft` | `vllm_server` | `tickets.csv` with `question,answer` | `gold.jsonl` | FAQ assistant SFT-style Q&A. Seeder maps to `qa-pair`, but exact eval handler for `instruction_sft` still needs verification. |
| `pii-detector` | partial | `structured_extraction` | `vllm_server` | `pii_records.csv` with `text,entities_json` | `gold.jsonl` | Span-set structured extraction for PII/PCI entities. Synthetic span mode and structured eval services exist. |
| `sentiment-classifier` | partial | `classification` | `mobile_cpu` | `reviews.csv` with `text,label` | `gold.jsonl` | Three-class product review classifier. Classification eval handler and ONNX/mobile-oriented export surfaces exist, but a real ONNX output has not been run. |

## Unsupported Or Not Yet Verified Claims

- Automatic cleaning-stage duplicate removal was not verified. Evidence supports hash
  computation, gold-sampling dedup, and semantic redundancy analysis.
- A real end-to-end training run was not verified in this pass.
- A real compressed/exported artifact was not verified in this pass.
- A trained model appearing automatically in the playground was not verified.
- Any final browser recording must distinguish measured, estimated, simulated,
  conceptual, and unknown paths.

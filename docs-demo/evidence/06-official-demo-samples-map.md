# Official Demo Samples Map

Only these folders are official demo templates for this demo workspace:
- `backend/data/demo_samples/pii-detector`
- `backend/data/demo_samples/sentiment-classifier`
- `backend/data/demo_samples/support-faq`

Evidence sources:
- sample folders listed above
- `backend/app/services/demo_project_service.py`
- `backend/app/api/demo_projects.py`
- `frontend/src/components/dashboard/DemoProjectTiles.tsx`
- `frontend/src/pages/ProjectPipelinePage.tsx`

## support-faq

| Field | Value |
|---|---|
| Sample name | Demo - Support FAQ |
| Exact folder path | `backend/data/demo_samples/support-faq` |
| Files present | `manifest.json`, `tickets.csv`, `gold.jsonl` |
| Data format | CSV source with `question,answer`; JSONL gold rows with `key`, `input`, `expected`, `rationale` |
| Apparent task type | `instruction_sft` from manifest |
| Labels/classes | None discovered |
| Dataset size | 20 CSV rows; 200 gold JSONL rows |
| Config files | `manifest.json` |
| Expected pipeline use | Seeder maps task to `qa-pair`, creates raw documents, locked gold set, and prepared train/val/test splits |
| Related frontend pages | Project list demo tiles; `/project/{id}/pipeline/data`; pipeline tabs in `ProjectPipelinePage.tsx` |
| Related backend APIs | `GET /api/demo-projects`; `POST /api/demo-projects/support-faq`; project pipeline APIs |
| Related services/jobs | `demo_project_service.seed_demo_project`; downstream cleaning/synthetic/dataset/training services are available but not yet run for this sample |
| What can be recorded in UI | Demo tile seeding, project open, source rows/documents, gold set, pipeline tabs, prepared dataset views if visible |
| What must be done via API/CLI if UI is missing | Direct seed via `POST /api/demo-projects/support-faq`; verify prepared files via project data paths if UI does not expose them |
| Status | partial |
| Open questions | Manifest prose mentions 6 hand-labelled gold rows, but `gold.jsonl` currently counts 200 rows. Does the UI expose the seeded prepared manifest clearly? Which downstream runs complete locally without extra services? |

Manifest-backed details:
- `target_profile`: `vllm_server`
- `training_preferred_plan_profile`: `balanced`
- `evaluation_preferred_pack_id`: `evalpack.general.default`
- `dataset_input_field`: `question`
- `dataset_output_field`: `answer`
- Suggested brief describes a concise SaaS FAQ support assistant.

## pii-detector

| Field | Value |
|---|---|
| Sample name | Demo - PII / PCI Detector |
| Exact folder path | `backend/data/demo_samples/pii-detector` |
| Files present | `manifest.json`, `pii_records.csv`, `gold.jsonl`, `_generate_bundle.py`, `kaggle_pii_to_brewslm.py` |
| Data format | CSV source with `text,entities_json`; JSONL gold rows with structured entity expectations |
| Apparent task type | `structured_extraction` from manifest |
| Labels/classes | Entity types: email, phone, ssn, credit_card, person_name, street_address, date_of_birth, ip_address, api_key, bank_account |
| Dataset size | 61 CSV rows counted; 200 gold JSONL rows counted |
| Config files | `manifest.json`; generator/converter scripts document source construction |
| Expected pipeline use | Seeder maps task to `structured-extraction`, forwards `output_schema` and `entity_types` into prepared manifest, creates raw/gold/prepared datasets |
| Related frontend pages | Demo tiles; pipeline data/gold/synthetic/dataset prep/training/eval tabs |
| Related backend APIs | `GET /api/demo-projects`; `POST /api/demo-projects/pii-detector`; synthetic span endpoints; evaluation structured extraction handler evidence in services |
| Related services/jobs | `demo_project_service.seed_demo_project`; `synthetic_service` has span-generation and demo fallback helpers |
| What can be recorded in UI | Demo tile seeding, raw span JSON inspection, gold rows, output schema story, synthetic span generation if prerequisites are met |
| What must be done via API/CLI if UI is missing | Direct seed via API; optional Kaggle conversion via `kaggle_pii_to_brewslm.py` is a converter script, not an official template |
| Status | partial |
| Open questions | Manifest description says 60 snippets, but the CSV currently counts 61 data rows. Verify whether this is intentional or stale text. |

Manifest-backed details:
- `target_profile`: `vllm_server`
- `training_preferred_plan_profile`: `balanced`
- `evaluation_preferred_pack_id`: `evalpack.general.default`
- `dataset_input_field`: `text`
- `dataset_output_field`: `entities_json`
- `output_schema.scoring_mode`: `span_set`

## sentiment-classifier

| Field | Value |
|---|---|
| Sample name | Demo - Sentiment classifier |
| Exact folder path | `backend/data/demo_samples/sentiment-classifier` |
| Files present | `manifest.json`, `reviews.csv`, `gold.jsonl` |
| Data format | CSV source with `text,label`; JSONL gold rows with `expected.label` |
| Apparent task type | `classification` from manifest |
| Labels/classes | positive, neutral, negative |
| Dataset size | 30 CSV rows; 200 gold JSONL rows |
| Config files | `manifest.json` |
| Expected pipeline use | Seeder maps task to `classification-label`, forwards labels into prepared manifest, creates raw/gold/prepared datasets |
| Related frontend pages | Demo tiles; pipeline data/gold/dataset prep/tokenization/training/evaluation/export tabs |
| Related backend APIs | `GET /api/demo-projects`; `POST /api/demo-projects/sentiment-classifier`; dataset, training, evaluation, export APIs |
| Related services/jobs | `demo_project_service.seed_demo_project`; classification eval handler is referenced through evaluation services |
| What can be recorded in UI | Demo tile seeding, raw review/label inspection, label set explanation, training/eval/export path after runtime verification |
| What must be done via API/CLI if UI is missing | Direct seed via API; inspect prepared manifest/files if UI does not show labels |
| Status | partial |
| Open questions | Manifest prose mentions a 10-row gold set, but `gold.jsonl` currently counts 200 rows. Manifest says target is mobile CPU and ONNX-INT8 export, but real export support must be verified before demo claims. |

Manifest-backed details:
- `target_profile`: `mobile_cpu`
- `training_preferred_plan_profile`: `fast-iteration`
- `evaluation_preferred_pack_id`: `evalpack.classification.default`
- `dataset_input_field`: `text`
- `dataset_output_field`: `label`

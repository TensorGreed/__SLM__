# Official Demo Samples Map

Discovery date: 2026-05-19.

Only these folders are official demo templates for this demo workspace:

- `backend/data/demo_samples/pii-detector`
- `backend/data/demo_samples/sentiment-classifier`
- `backend/data/demo_samples/support-faq`

Do not invent other official samples. Helper scripts inside a sample folder are
evidence for that sample only; they are not additional official templates.

Status legend: verified, partial, simulated, estimated, conceptual, unsupported,
unknown.

Shared evidence:

- `backend/app/services/demo_project_service.py`
- `backend/app/api/demo_projects.py`
- `frontend/src/components/dashboard/DemoProjectTiles.tsx`
- `frontend/src/pages/ProjectPipelinePage.tsx`

## Summary

| Sample | Status | Source rows | Gold rows | Task profile | Target profile | Main demo role |
|---|---:|---:|---|---|---|---|
| `support-faq` | partial | 20 | 200 | `instruction_sft` | `vllm_server` | Q&A/FAQ fine-tuning style walkthrough |
| `pii-detector` | partial | 61 | 200 | `structured_extraction` | `vllm_server` | PII/PCI span extraction walkthrough |
| `sentiment-classifier` | partial | 30 | 200 | `classification` | `mobile_cpu` | Three-way sentiment classifier walkthrough |

## support-faq

| Field | Evidence-backed value |
|---|---|
| Sample name | Demo - Support FAQ |
| Exact folder path | `backend/data/demo_samples/support-faq` |
| Files present | `manifest.json`, `tickets.csv`, `gold.jsonl` |
| Data format | CSV source with `question,answer`; JSONL gold rows with `key`, `input`, `expected`, `rationale` |
| Apparent task type | `instruction_sft` from `manifest.json` |
| Labels/classes | None discovered |
| Dataset size | 20 CSV rows; 200 gold JSONL rows |
| Config files | `manifest.json` |
| Expected pipeline use | Seeder maps this task to `qa-pair`, creates raw documents, locked gold set, and prepared train/val/test splits |
| Related frontend pages | Dashboard demo tile; `/project/{id}/pipeline/data`; `/goldset`; `/dataprep`; `/training`; `/eval`; `/export` |
| Related backend APIs | `GET /api/demo-projects`; `POST /api/demo-projects/support-faq`; project pipeline APIs |
| Related services/jobs | `demo_project_service.seed_demo_project`; dataset adapter/prep services; training/eval/export services are available but runtime-dependent |
| What can be recorded in UI | Seed tile, project open, raw Q&A rows if exposed, gold rows, split/prep surfaces, training/eval/export surfaces |
| What requires API/CLI | Direct seed API if UI is missing; file/path inspection if prepared manifest is not visible |
| Heavy/external dependencies | Real training, held-out evaluation, compression/export/serve |
| Viewer should see | A support FAQ assistant dataset with customer questions and target answers |
| Status | partial |

Manifest-backed details:

- `target_profile`: `vllm_server`
- `training_preferred_plan_profile`: `balanced`
- `evaluation_preferred_pack_id`: `evalpack.general.default`
- `dataset_input_field`: `question`
- `dataset_output_field`: `answer`
- Suggested brief describes a concise SaaS FAQ support assistant.

Seeder-backed details:

- Expected prepared split counts: 16 train, 2 validation, 2 test.
- Prepared rows are canonicalized with `text`, `source_text`, `target_text`,
  `question`, and `answer`.

Open questions:

- Manifest prose mentions 6 hand-labelled gold rows, but the current
  `gold.jsonl` contains 200 rows.
- Which eval task handler is used for `instruction_sft` after prepared manifest
  and adapter resolution?
- Does the UI expose seeded prepared split files clearly enough for recording?

## pii-detector

| Field | Evidence-backed value |
|---|---|
| Sample name | Demo - PII / PCI Detector |
| Exact folder path | `backend/data/demo_samples/pii-detector` |
| Files present | `manifest.json`, `pii_records.csv`, `gold.jsonl`, `_generate_bundle.py`, `kaggle_pii_to_brewslm.py` |
| Data format | CSV source with `text,entities_json`; JSONL gold rows with structured entity expectations |
| Apparent task type | `structured_extraction` from `manifest.json` |
| Labels/classes | Entity types: email, phone, ssn, credit_card, person_name, street_address, date_of_birth, ip_address, api_key, bank_account |
| Dataset size | 61 CSV rows counted; 200 gold JSONL rows counted |
| Config files | `manifest.json`; generator/converter scripts support this sample |
| Expected pipeline use | Seeder maps this task to `structured-extraction`, forwards `output_schema` and entity types into prepared manifest, creates raw/gold/prepared datasets |
| Related frontend pages | Dashboard demo tile; pipeline data/gold/synthetic/dataprep/tokenization/training/eval/compression/export tabs |
| Related backend APIs | `GET /api/demo-projects`; `POST /api/demo-projects/pii-detector`; synthetic span endpoints; evaluation endpoints |
| Related services/jobs | `demo_project_service`; `synthetic_service` span generation; `eval_task_handler_service` structured extraction handler |
| What can be recorded in UI | Seed tile, raw text plus entity JSON, gold rows, span-generation mode if configured, structured eval surfaces if predictions/model exist |
| What requires API/CLI | Direct seed API; optional helper scripts for generating/converting data; runtime calls for real synthetic/training/eval |
| Heavy/external dependencies | Teacher model for real synthetic spans; external/simulated training; held-out model inference; export/serve tools |
| Viewer should see | A span-set extraction task that identifies typed PII/PCI entities in text |
| Status | partial |

Manifest-backed details:

- `target_profile`: `vllm_server`
- `training_preferred_plan_profile`: `balanced`
- `evaluation_preferred_pack_id`: `evalpack.general.default`
- `dataset_input_field`: `text`
- `dataset_output_field`: `entities_json`
- `output_schema.scoring_mode`: `span_set`

CSV entity counts:

| Entity type | Count |
|---|---:|
| `person_name` | 27 |
| `email` | 23 |
| `phone` | 14 |
| `date_of_birth` | 7 |
| `credit_card` | 8 |
| `bank_account` | 7 |
| `street_address` | 8 |
| `ssn` | 7 |
| `ip_address` | 10 |
| `api_key` | 5 |

Gold entity counts:

| Entity type | Count |
|---|---:|
| `person_name` | 138 |
| `date_of_birth` | 36 |
| `street_address` | 42 |
| `ssn` | 28 |
| `ip_address` | 34 |
| `bank_account` | 29 |
| `email` | 72 |
| `phone` | 47 |
| `credit_card` | 26 |
| `api_key` | 21 |

Seeder-backed details:

- Expected prepared split counts: 45 train, 8 validation, 8 test.
- Prepared manifest should preserve output schema and entity types.

Open questions:

- Manifest description says 60 snippets, but `pii_records.csv` contains 61 data
  rows.
- Browser demo must keep two ideas separate: cleaning-time PII redaction and the
  trained PII detector/extractor task.
- Real span generation requires teacher/fallback configuration; do not claim it
  is automatic.

## sentiment-classifier

| Field | Evidence-backed value |
|---|---|
| Sample name | Demo - Sentiment classifier |
| Exact folder path | `backend/data/demo_samples/sentiment-classifier` |
| Files present | `manifest.json`, `reviews.csv`, `gold.jsonl` |
| Data format | CSV source with `text,label`; JSONL gold rows with `expected.label` |
| Apparent task type | `classification` from `manifest.json` |
| Labels/classes | `positive`, `neutral`, `negative` |
| Dataset size | 30 CSV rows; 200 gold JSONL rows |
| Config files | `manifest.json` |
| Expected pipeline use | Seeder maps this task to `classification-label`, forwards labels into prepared manifest, creates raw/gold/prepared datasets |
| Related frontend pages | Dashboard demo tile; pipeline data/gold/dataprep/tokenization/training/evaluation/compression/export tabs; playground/export for final usage |
| Related backend APIs | `GET /api/demo-projects`; `POST /api/demo-projects/sentiment-classifier`; dataset/training/evaluation/compression/export/registry APIs |
| Related services/jobs | `demo_project_service`; classification eval handler; export and deployment target services |
| What can be recorded in UI | Seed tile, raw reviews and labels, label set, dataset prep, classification eval surfaces, export surfaces after artifact exists |
| What requires API/CLI | Direct seed API; training/compression/export runs; final model smoke request |
| Heavy/external dependencies | External/simulated training; ONNX/quantization dependencies; local runtime for final model test |
| Viewer should see | A three-way review sentiment classifier with label-aware evaluation once a model exists |
| Status | partial |

Manifest-backed details:

- `target_profile`: `mobile_cpu`
- `training_preferred_plan_profile`: `fast-iteration`
- `evaluation_preferred_pack_id`: `evalpack.classification.default`
- `dataset_input_field`: `text`
- `dataset_output_field`: `label`

CSV label counts:

| Label | Count |
|---|---:|
| `positive` | 10 |
| `neutral` | 10 |
| `negative` | 10 |

Gold label counts:

| Label | Count |
|---|---:|
| `positive` | 70 |
| `neutral` | 65 |
| `negative` | 65 |

Seeder-backed details:

- Expected prepared split counts: 22 train, 4 validation, 4 test.
- Prepared manifest should preserve labels.

Open questions:

- Manifest prose mentions a 10-row gold set, but the current `gold.jsonl`
  contains 200 rows.
- Manifest says target is mobile CPU and mentions ONNX-INT8 export intent, but a
  successful ONNX/INT8 artifact was not run in this pass.

## Cross-Sample Recording Notes

- All three samples can be selected through the demo tile UI or seeded by API.
- All three seeded projects already include raw rows, gold rows, and prepared
  splits.
- All three can be used to show the shared pipeline tabs.
- Real training/evaluation/compression/export/final usage remain partial until
  a runtime-specific demo path is executed and documented.

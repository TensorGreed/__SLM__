# Pipeline Discovery Checklist

## Sample Data Files Discovered

- [x] `support-faq`
- [x] `pii-detector`
- [x] `sentiment-classifier`

## Schemas/Configs Discovered

- [x] Sample manifests
- [x] PII output schema
- [x] Sentiment label list
- [ ] Prepared manifest after live seeding

## Frontend Routes/Pages Discovered

- [x] Login page
- [x] Project list
- [x] Project pipeline page
- [x] Training config page
- [x] Playground page
- [x] Export/deployment pages
- [ ] Stable selectors for recordings

## API Endpoints Discovered

- [x] Demo projects
- [x] Ingestion
- [x] Cleaning
- [x] Gold
- [x] Synthetic
- [x] Dataset prep
- [x] Tokenization
- [x] Training
- [x] Evaluation
- [x] Compression
- [x] Export
- [x] Registry
- [x] Deployments

## Backend Services Discovered

- [x] Demo project service
- [x] Starter pack service
- [x] Training runtime service
- [x] Synthetic service
- [x] Compression service
- [ ] Full export service behavior
- [ ] Full serve service behavior

## Pipeline Stages Discovered

- [x] ingestion
- [x] cleaning
- [x] gold_set
- [x] synthetic
- [x] dataset_prep
- [x] data_adapter_preview
- [x] tokenization
- [x] training
- [x] evaluation
- [x] compression
- [x] export
- [x] completed

## Async Jobs/Workers Discovered

- [x] Remote import queue
- [x] Cleaning batch async
- [x] Synthetic span async
- [x] Training task polling
- [x] Compression task polling
- [ ] Which demos require Celery/Redis

## Required Environment Variables

- [x] Auth/API key basics
- [x] Synthetic fallback flag
- [x] Training runtime flags
- [x] Compression runtime flags
- [ ] Recording-specific env profile

## Required External Services

- [ ] Teacher model for synthetic generation
- [ ] Judge model for LLM judge evaluation
- [ ] Redis/Celery for queued jobs
- [ ] External training command dependencies
- [ ] Compression toolchain

## Heavy Dependencies

- [ ] GPU or CPU-only training feasibility
- [ ] Tokenizer/model downloads
- [ ] Export/conversion tools

## Sample-Specific Steps

- [x] support-faq source/gold/manifest
- [x] pii-detector source/gold/manifest/schema/scripts
- [x] sentiment-classifier source/gold/manifest/labels
- [ ] One seeded project verification per sample

## Unsupported/Unknown Steps

- [ ] PII masking/removal
- [ ] ONNX-INT8 export proof
- [ ] Final API usage of trained model
- [ ] Deployment target that works credential-free

## Final Model Usage Path

- [ ] Playground proof
- [ ] Export proof
- [ ] Serve plan proof
- [ ] Registry/deploy proof
- [ ] API smoke proof


# Video 04: PII Detector Pipeline Outline

Goal: show the official PII/PCI structured extraction sample.

Evidence:
- `backend/data/demo_samples/pii-detector/manifest.json`
- `backend/data/demo_samples/pii-detector/pii_records.csv`
- `backend/data/demo_samples/pii-detector/gold.jsonl`
- `backend/data/demo_samples/pii-detector/_generate_bundle.py`
- `backend/data/demo_samples/pii-detector/kaggle_pii_to_brewslm.py`

Sections:
1. Seed PII detector.
2. Inspect source text and `entities_json`.
3. Explain entity types and span offsets.
4. Inspect gold rows.
5. Show structured extraction task profile and output schema.
6. Verify span generation/evaluation before recording runtime steps.


# Demo Flows

These are starter flows based on inspected repo evidence. They are not final recording scripts.

## Flow 1: Official Demo Project Quickstart

Status: partial.

Evidence:
- `frontend/src/components/dashboard/DemoProjectTiles.tsx`
- `backend/app/api/demo_projects.py`
- `backend/app/services/demo_project_service.py`

Steps:
1. Start backend and frontend.
2. Login with local credentials if auth is enabled.
3. Open project list.
4. Use one of the three demo project tiles.
5. Confirm project opens at `/project/{id}`.
6. Inspect data, gold, prepared split, and pipeline tabs.

Notes:
- Real selectors and durable Playwright flow are to be added after a UI pass.
- Demo tiles call `GET /api/demo-projects` and `POST /api/demo-projects/{slug}`.

## Flow 2: Support FAQ Pipeline

Status: partial.

Evidence:
- `backend/data/demo_samples/support-faq/manifest.json`
- `backend/data/demo_samples/support-faq/tickets.csv`
- `backend/data/demo_samples/support-faq/gold.jsonl`
- `backend/app/services/demo_project_service.py`

Likely story:
1. Seed the `support-faq` demo.
2. Inspect source Q&A tickets.
3. Inspect locked gold rows.
4. Show prepared train/val/test split seeded by the backend.
5. Configure or inspect training.
6. Run evaluation when runtime prerequisites are ready.
7. Export or test final model only if verified.

## Flow 3: PII Detector Pipeline

Status: partial.

Evidence:
- `backend/data/demo_samples/pii-detector/manifest.json`
- `backend/data/demo_samples/pii-detector/pii_records.csv`
- `backend/data/demo_samples/pii-detector/gold.jsonl`
- `backend/data/demo_samples/pii-detector/_generate_bundle.py`
- `backend/data/demo_samples/pii-detector/kaggle_pii_to_brewslm.py`

Likely story:
1. Seed the `pii-detector` demo.
2. Inspect span JSON in source CSV.
3. Inspect output schema and entity types from manifest.
4. Show prepared manifest forwarding schema/entity types.
5. Use span synthetic generation only after verifying UI/API behavior.

## Flow 4: Sentiment Classifier Pipeline

Status: partial.

Evidence:
- `backend/data/demo_samples/sentiment-classifier/manifest.json`
- `backend/data/demo_samples/sentiment-classifier/reviews.csv`
- `backend/data/demo_samples/sentiment-classifier/gold.jsonl`

Likely story:
1. Seed the `sentiment-classifier` demo.
2. Inspect product review text and labels.
3. Show label set: positive, neutral, negative.
4. Show mobile CPU target and ONNX-INT8 story as manifest-backed intent.
5. Verify export path before claiming ONNX output.


# Quickstart Path

Status: starter plan. Items marked "to verify from repo" need a real run-through before recording.

## Local Setup

1. Backend:
   ```bash
   cd backend
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   cp .env.example .env
   uvicorn app.main:app --reload --port 8000
   ```
2. Frontend:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

## Login

Evidence: `frontend/src/App.tsx`, `frontend/src/pages/SSOLoginPage.tsx`, `backend/app/api/auth.py`, `backend/.env.example`.

Use local login when auth is enabled:
- Username: any value
- Password: `API_KEY` from `backend/.env`, default `sk-mock-admin-key`

To verify from repo:
- Whether current `.env` disables auth or uses local login.

## Project Creation

Use one of the official demo tiles on the project list.

To verify from repo:
- Exact button text and post-click route.
- Whether an already seeded demo returns the existing project.

## Choosing An Official Demo Sample

Official choices:
- `support-faq`
- `pii-detector`
- `sentiment-classifier`

To verify from repo:
- Tile order and visible copy in the current UI.

## Running Or Previewing Pipeline Stages

Evidence: `ProjectPipelinePage.tsx`.

Starter path:
1. Data.
2. Cleaning.
3. Gold Set.
4. Synthetic.
5. Dataset Prep.
6. Tokenization.
7. Training Config.
8. Training Runs.
9. Evaluation.
10. Compression.
11. Export.

To verify from repo:
- Which stages should be run versus inspected for pre-seeded demos.
- Which runtime settings are needed for training, synthetic generation, compression, and export.

## Seeing Outputs

To verify from repo:
- Source documents and samples in Data tab.
- Gold rows in Gold Set tab or workbench.
- Split manifest and preview in Dataset Prep tab.
- Token stats in Tokenization tab.
- Training metrics and logs.
- Evaluation scorecard/gates.
- Export package and registry entry.

## Next Steps

- Map selectors.
- Record prototype manual path.
- Build one real Playwright flow after evidence docs are complete.


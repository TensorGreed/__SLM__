# Quickstart Path

Status: **verified** for all manual setup + login + first-seed steps
(2026-05-19 selector-discovery pass). Items marked "to verify from
repo" have been resolved unless explicitly re-flagged below.

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

Evidence: `frontend/src/App.tsx`, `frontend/src/pages/SSOLoginPage.tsx`, `backend/app/api/auth.py:168-169`, `backend/.env.example:18`.

Use local login when auth is enabled:
- Username: any value (e.g. `demo`)
- Password: `API_KEY` from `backend/.env`, default `sk-mock-admin-key`

**Verified selectors** (2026-05-19):
- Username field: placeholder `Enter your username`.
- Password field: placeholder `API Key or Password`.
- Submit: button with role `Sign in`.

Screenshots: `docs-demo/screenshots/selector-pass-01-login.png`.

## Project Creation

Use one of the official demo tiles on the project list.

**Verified** (2026-05-19):
- Tile container selector: `.demo-project-tiles`.
- Tile button selector: `.demo-project-tile`.
- Tile aria-labels: `Open the Demo · Support FAQ demo project`,
  `Open the Demo · PII / PCI Detector demo project`,
  `Open the Demo · Sentiment classifier demo project` (note the
  centered dot in `·`).
- Click fires `POST /api/demo-projects/<slug>` → 200.
- Already-seeded demo: backend reuses the existing project (idempotent).
- Post-click navigation: `/project/<id>/pipeline/data`.

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

- Map selectors — **done**, see `docs-demo/evidence/11-selector-route-evidence.md`.
- Record prototype manual path — **done** for support-faq, pii-detector, and sentiment-classifier (screenshots under `docs-demo/screenshots/selector-pass-*`).
- Build one real Playwright flow — **next**, see `docs-demo/videos/02-brewslm-quickstart/recording-plan.md` for the first target.
- Decide on `data-testid` additions before recording series 03–07 — see `docs-demo/evidence/11-selector-and-instrumentation-plan.md`.

## Maps onto Video 02

This learning-path file is the conceptual companion to **Video 02 —
BrewSLM Quickstart** (`docs-demo/videos/02-brewslm-quickstart/recording-plan.md`).


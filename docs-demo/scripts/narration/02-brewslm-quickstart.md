# BrewSLM Quickstart Narration Skeleton

Status: repo-specific points are to-verify until a real recording pass.

1. Open with what is verified:
   - BrewSLM is a FastAPI plus React project in this repo. Evidence: `README.md`, `backend/app/main.py`, `frontend/src`.
   - The frontend has pipeline pages and official demo tiles. Evidence: `ProjectPipelinePage.tsx`, `DemoProjectTiles.tsx`.

2. Start services:
   - Backend command: to verify during recording.
   - Frontend command: `npm run dev` from `frontend/package.json`.

3. Login:
   - To verify current local auth state.
   - If auth is enabled, local login uses any username and API key password.

4. Project list:
   - Show official demo tiles.
   - Say only three official samples are being used.

5. Seed one sample:
   - To verify route and visible state.
   - Mention seeder creates project, raw data, gold set, and prepared splits only after showing evidence.

6. Pipeline preview:
   - Walk tabs without claiming every step has run.


# Recording Plan

## Local Services

Backend command verified from `README.md`, `slm-docs/docs/getting-started/quickstart.md`, and `slm-docs/docs/setup/install.md`:

```bash
cd backend
source .venv/bin/activate
uvicorn app.main:app --reload --port 8000
```

Backend setup command from docs:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

In this environment, `python` was not available during inspection, while `python3` was available. If `python -m venv` fails locally, use:

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Frontend command verified from `frontend/package.json`:

```bash
cd frontend
npm run dev
```

Frontend Vite proxy verified from `frontend/vite.config.ts`: `/api` proxies to `http://localhost:8000`.

## Login

Evidence:
- `frontend/src/App.tsx` redirects to `/login` unless `localStorage.slm_token` exists.
- `frontend/src/pages/SSOLoginPage.tsx` posts `/api/auth/local/login`.
- `backend/app/api/auth.py` accepts any username with password equal to `API_KEY`.
- `backend/.env.example` sets `API_KEY="sk-mock-admin-key"`.

Recording setup should use:
- URL: `http://localhost:5173/login`
- Username: any value, for example `demo`
- Password: `sk-mock-admin-key`, unless `backend/.env` changes `API_KEY`

## Playwright

Current prototype spec:
- Opens `baseURL`.
- Waits briefly.
- Saves `docs-demo/screenshots/01-prototype-smoke.png`.
- Does not depend on login selectors.
- Produces Playwright video output.

Do not add real selector flows until:
- Sample files are mapped.
- UI routes are mapped.
- APIs are mapped.
- Real pipeline steps are mapped.
- Runtime prerequisites are known.

## Runtime Caveats

Updated 2026-05-19 after the runtime decisions captured in
`12-runtime-decisions-2026-05-19.md`. The defaults below are still
accurate as repo state; the decisions section is which of those
defaults we'll record against.

Training:
- `backend/app/config.py` defaults `TRAINING_BACKEND="external"` and `ALLOW_SIMULATED_TRAINING=false`.
- **Recording decision**: use the external default with a real Celery
  worker. No simulated training.

Synthetic generation:
- `ALLOW_SYNTHETIC_DEMO_FALLBACK=false` by default in `backend/.env.example`.
- **Recording decision**: use a real Ollama teacher at
  `http://localhost:11434/v1` running
  `qwen2.5:7b-instruct-q4_K_M`. Fallback only if Ollama is
  unreachable at record time.

Compression:
- `COMPRESSION_BACKEND="external"` and `ALLOW_STUB_COMPRESSION=false` by default.
- **Recording decision**: GGUF quantization via llama.cpp's
  `quantize` binary (`backend/scripts/quantize.py`'s GGUF path). No
  stub.

Evaluation:
- LLM judge endpoint: same Ollama at `http://localhost:11434/v1`,
  same `qwen2.5:7b-instruct-q4_K_M` model.

Serving:
- Local serve runtime: Ollama. The GGUF artifact produced by the
  compression step gets loaded directly.

Required local services before recording Videos 04 / 09 / 10 / 11 /
12 (full list with copy-paste commands lives in
`12-runtime-decisions-2026-05-19.md`):

1. Redis (`redis-server --daemonize yes`).
2. Celery worker (`celery -A app.worker.celery_app worker
   --loglevel=INFO --pool=threads --concurrency=2`).
3. Ollama (`ollama serve` + `ollama pull qwen2.5:7b-instruct-q4_K_M`).
4. Four `backend/.env` additions (TEACHER_MODEL_* and JUDGE_MODEL_*).

Modules 02, 03, 05, 06, 07 (inspect-only) do not need any of these.

## Per-video Recording Plans (2026-05-19 expansion)

Each module's full recording plan lives under
`docs-demo/videos/<module>/recording-plan.md`. Index:

| Module | Title | Recording plan | Status |
|---|---|---|---|
| 01 | SLM 101 | `docs-demo/videos/01-slm-101/recording-plan.md` (slide-only) | conceptual |
| **02** | **BrewSLM Quickstart** | `docs-demo/videos/02-brewslm-quickstart/recording-plan.md` | **ready** — first real recording target |
| 03 | Support FAQ Pipeline | `docs-demo/videos/03-support-faq-pipeline/recording-plan.md` | partial |
| 04 | PII Detector Pipeline | `docs-demo/videos/04-pii-detector-pipeline/recording-plan.md` | partial |
| 05 | Sentiment Classifier Pipeline | `docs-demo/videos/05-sentiment-classifier-pipeline/recording-plan.md` | partial |
| 06 | Custom Outside Samples | `docs-demo/videos/06-custom-outside-samples-pipeline/recording-plan.md` (not yet fleshed out) | partial |
| 07 | Final Model Usage | `docs-demo/videos/07-final-model-usage/recording-plan.md` (not yet fleshed out) | partial |

Recommended recording order (mirrors
`03-video-series-plan.md`):

1. **02 quickstart** — first, every selector verified.
2. **03 support-faq** — second, smallest runtime risk of the three samples.
3. **04 pii-detector** + **05 sentiment-classifier** — independent, can record in parallel sessions.
4. Modules 08–14 unlock as the open runtime questions (Q16–Q25) resolve.

## Selector Source of Truth

`docs-demo/evidence/11-selector-route-evidence.md` is the canonical
record of which selectors work for which routes, observed during the
2026-05-19 disposable-UI passes. Any new recording plan should cite
that file, not re-discover selectors.

For proposed product-level instrumentation (e.g. adding
`data-testid="demo-tile-<slug>"` to the demo tiles) see
`docs-demo/evidence/11-selector-and-instrumentation-plan.md`.


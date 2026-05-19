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

Training:
- `backend/app/config.py` defaults `TRAINING_BACKEND="external"` and `ALLOW_SIMULATED_TRAINING=false`.
- `backend/app/services/training_runtime_service.py` documents enabling simulated training for demos.
- Real training may require heavy dependencies and prepared data.

Synthetic generation:
- `ALLOW_SYNTHETIC_DEMO_FALLBACK=false` by default in `backend/.env.example`.
- Teacher model config may be required for non-fallback generation.

Compression:
- `COMPRESSION_BACKEND="external"` and `ALLOW_STUB_COMPRESSION=false` by default.
- Real compression can require external tools such as quantization scripts or llama.cpp.


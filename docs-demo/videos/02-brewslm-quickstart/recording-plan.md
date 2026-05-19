# Video 02 — BrewSLM Quickstart · Recording Plan

Status: **ready** — every selector, route, and screenshot below has
already been verified by the 2026-05-19 selector-discovery pass.
This is the recommended **first real Playwright recording target**.

## Goal

Take a viewer from a fresh repo clone to a seeded demo project,
landed on the Data tab with 20 raw rows visible, in under five
minutes of screen time.

## Audience

Beginner. Assumes nothing about ML pipelines.

## Expected video length

5–7 minutes.

## Prerequisites (manual; not part of recording)

| Step | Command | Evidence |
|---|---|---|
| 1. Backend deps | `cd backend && python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && cp .env.example .env` | `README.md`, `slm-docs/docs/getting-started/quickstart.md` |
| 2. Backend boot | `uvicorn app.main:app --reload --port 8000` from `backend/` (venv active) | `04-recording-plan.md`, `README.md` |
| 3. Frontend boot | `npm run dev` from `frontend/` | `frontend/package.json` |
| 4. Vite proxy | none needed; `frontend/vite.config.ts` already proxies `/api` to `http://localhost:8000` | `frontend/vite.config.ts` |
| 5. Login credentials | username = anything (e.g. `demo`); password = `sk-mock-admin-key` (default `API_KEY` in `backend/.env.example:18`) | `backend/app/api/auth.py:168-169`, `backend/.env.example:18` |
| 6. Optional clean state | If a previous `support-faq` project exists in the DB, either delete it via the project list or use a disposable data dir as Codex did in the selector pass (`/tmp/slm-selector-pass.*`) | `docs-demo/evidence/11-selector-route-evidence.md` "Disposable Run Setup" |

**Confirm app is ready** (manual smoke test before pressing Record):

1. Open `http://localhost:5173/login` in a browser. You should see
   the BrewSLM login form (placeholder text `Enter your username`
   and `API Key or Password`).
2. `curl http://localhost:8000/api/health` should return
   `{"status":"ok"}`.
3. `curl http://localhost:8000/api/demo-projects` should return a
   JSON array containing the three official demo entries with slugs
   `pii-detector`, `sentiment-classifier`, `support-faq`.

If any of those three fail, **do not start recording** — fix and
re-verify.

## Recording arc (5 sections, target ~60 sec each)

### Section 1 — Login (45–60 sec)

| Step | Action | What viewer sees | Narration checkpoint |
|---|---|---|---|
| 1.1 | Navigate to `http://localhost:5173/login` | BrewSLM login form | "This is BrewSLM running locally. We log in with a username and the local API key." |
| 1.2 | Type `demo` in username | Username field fills | (silence) |
| 1.3 | Type `sk-mock-admin-key` in password | Password field fills | "The default password is the `API_KEY` value in `backend/.env`." |
| 1.4 | Click **Sign in** | Browser navigates to `/` (project list) | "And we're in." |

**Selectors verified** (`docs-demo/evidence/11-selector-route-evidence.md`):
- Username: `getByPlaceholder("Enter your username")`
- Password: `getByPlaceholder("API Key or Password")`
- Submit: `getByRole("button", { name: /^Sign in$/ })`

**Screenshot to capture**: `selector-pass-01-login.png` already
exists; re-shoot at recording resolution if needed.

### Section 2 — Project list + demo catalog (45–60 sec)

| Step | Action | What viewer sees | Narration checkpoint |
|---|---|---|---|
| 2.1 | (already there from login) | Project list page; demo tile strip near the top | "On the project list we can either bring our own dataset, or pick one of three official demo samples." |
| 2.2 | Pause on the demo tile strip | Three tiles: `Demo · Support FAQ`, `Demo · PII / PCI Detector`, `Demo · Sentiment classifier` | "These three samples are evidence-backed and live under `backend/data/demo_samples/`. They're the only official starting points." |
| 2.3 | Hover the first tile (no click yet) | Tile aria-label tooltip: `Open the Demo · Support FAQ demo project` | (silence — let the viewer read) |

**Selectors verified**:
- Container: `.demo-project-tiles`
- Tile: `.demo-project-tile`
- Filter by text: `Support FAQ`, `PII / PCI Detector`, `Sentiment classifier`
- Aria-label form: `Open the Demo · <name> demo project`

**Screenshot**: `selector-pass-02-demo-tiles.png` already exists.

**Pause for viewer comprehension**: 1.5 sec after step 2.2 so the
viewer can read all three tile names.

### Section 3 — Seed support-faq (60–75 sec)

| Step | Action | What viewer sees | Narration checkpoint |
|---|---|---|---|
| 3.1 | Click the **Support FAQ** tile | Brief loading state, then browser navigates to `/project/<new-id>/pipeline/data` | "Clicking the tile fires a `POST /api/demo-projects/support-faq`. The backend copies the sample CSV into the project's raw data, creates 20 raw documents, imports 200 gold rows from `gold.jsonl`, and pre-builds train/val/test splits at 16/2/2." |
| 3.2 | Wait for pipeline page to settle | Pipeline tabs render at the top: Data, Cleaning, Gold Set, Synthetic, Dataset Prep, Tokenization, Training, Evaluation, Compression, Export. Bottom shows 20 raw documents in the Data tab. | "Notice the seed has *already* pre-loaded data and prepared splits. That's why the Pipeline Status badge shows 'training' stage at 60%." |

**API calls fired** (verify in network panel during recording):
- `POST /api/demo-projects/support-faq` → 200
- `GET /api/projects/<id>` → 200
- `GET /api/projects/<id>/pipeline/status` → 200
- `GET /api/projects/<id>/ingestion/documents` → 200 (returns 20 docs)
- `GET /api/projects/<id>/ingestion/eda` → 200

**Screenshot**: `selector-pass-03-support-faq-data-tab.png` already
exists.

### Section 4 — Quick tour of pipeline tabs (90–120 sec)

Do **NOT** run anything. Just click tabs and let them render.

| Step | Action | What viewer sees | Narration checkpoint |
|---|---|---|---|
| 4.1 | Click **Cleaning** tab | "Cleaning Configuration" heading + cleaning options | "Cleaning is where we'd chunk text, redact regex-detected PII, and score quality. Doesn't run automatically." |
| 4.2 | Click **Gold Set** tab | "Gold Evaluation Dataset" + "Entries 200" badge | "The seeded gold set has 200 hand-labelled rows. Locked. Used for honest eval, not training." |
| 4.3 | Click **Dataset Prep** tab | "Dataset Preview", "Schema Profile", "Semantic Intelligence" panels | "Splits already exist (16 / 2 / 2). We can rebuild them here, but for the demo we'll trust the seed." |
| 4.4 | Click **Training** tab | "No experiments yet" empty state | "Training tab. We'd kick off a run here, but that needs Module 9." |
| 4.5 | Click back to **Data** | Data tab returns | "We're done with the tour. The viewer is now oriented." |

**Selectors verified**: All pipeline tabs are
`button.tab` elements with `title` attribute matching the visible
label (`Data`, `Cleaning`, `Gold Set`, etc).

**Screenshots covered by**: `selector-pass-05-cleaning-tab.png`
through `selector-pass-13-export-tab.png`.

**Pause for viewer comprehension**: 1 sec on each tab after it
loads.

### Section 5 — Wrap (30–45 sec)

| Step | Action | What viewer sees | Narration checkpoint |
|---|---|---|---|
| 5.1 | Expand one raw document row | `[data-testid="expand-doc-20"]` (or whichever doc id is first) opens; shows raw `question`/`answer` | "Each raw row is one ticket: the question and its agent answer. This is the data the SFT loop will fine-tune the model against." |
| 5.2 | (silence) | (viewer reads) | (pause 2 sec) |
| 5.3 | End on the Data tab | (no action) | "Next video: we'll walk through the full lifecycle — from this raw data, through cleaning, gold, synthetic, prep, all the way to a trained model." |

**Selector**: `[data-testid^="expand-doc-"]`. Concrete id observed
in the selector pass: `expand-doc-20` for support-faq.

**Screenshot**: `selector-pass-04-support-faq-expanded-raw-row.png`.

## What Playwright should automate

Goal of the script: deterministically reproduce the 5 sections above
under `npx playwright test` so the video render matches the docs.

**Yes, automate**:
- Navigation to `/login`, fill inputs, click Sign in.
- Wait for `/` (project list) URL.
- Click the Support FAQ tile by accessible name
  (`Open the Demo · Support FAQ demo project`).
- Wait for `/project/<id>/pipeline/data` URL pattern.
- Click each pipeline tab in sequence with 1-sec pauses.
- Click the first `expand-doc-*` element.

**No, do not automate** (manual / out of scope for this first cut):
- Killing existing Support FAQ projects before recording. Do that
  manually before each take.
- Asserting specific UI text beyond the section headings. The first
  recording is for visual capture, not regression testing.
- Window/cursor positioning effects. Use OBS or the Playwright video
  recorder's defaults.

## Selectors needed and current status

| UI action | Selector | Status |
|---|---|---|
| Login username | `getByPlaceholder("Enter your username")` | verified |
| Login password | `getByPlaceholder("API Key or Password")` | verified |
| Login submit | `getByRole("button", { name: /^Sign in$/ })` | verified |
| Demo tile (Support FAQ) | `[aria-label="Open the Demo · Support FAQ demo project"]` | verified |
| Pipeline tab (any) | `button.tab[title="Data"]`, etc | verified (10 tabs total) |
| Raw row expander | `[data-testid^="expand-doc-"]` | verified |

**No new data-testid additions required for this recording.** The
demo-tile selector currently relies on the centered-dot in `Demo · …`
form, which is a real character in the manifest names. If we ever
want a more brittle-resistant selector, `data-testid="demo-tile-<slug>"`
on `.demo-project-tile` would be ideal — but that is **not blocking**
this recording and is captured as a proposal in
`11-selector-and-instrumentation-plan.md`.

## Screenshots to capture during the real recording

| # | Filename target | Trigger |
|---|---|---|
| 1 | `docs-demo/screenshots/v02-login.png` | After step 1.1 |
| 2 | `docs-demo/screenshots/v02-tiles.png` | After step 2.2 |
| 3 | `docs-demo/screenshots/v02-data-tab.png` | After step 3.2 |
| 4 | `docs-demo/screenshots/v02-cleaning-tab.png` | After step 4.1 |
| 5 | `docs-demo/screenshots/v02-goldset-tab.png` | After step 4.2 |
| 6 | `docs-demo/screenshots/v02-training-tab-empty.png` | After step 4.4 |
| 7 | `docs-demo/screenshots/v02-expanded-row.png` | After step 5.1 |

(Reusing the existing selector-pass screenshots is fine if rerecording at the same resolution.)

## Failure modes to anticipate

| Failure | Symptom | Mitigation |
|---|---|---|
| Backend not running | Login submit fails with network error | Manual smoke test (3 commands above) before pressing Record |
| Wrong API_KEY | Login submit returns 401 | Re-read `backend/.env`; default is `sk-mock-admin-key` |
| Existing `Demo · Support FAQ` project conflicts | Tile click → opens existing project at different id; raw docs may not be exactly 20 | Delete the old project from project-list or use a disposable data dir |
| Vite HMR causes re-render mid-recording | Tabs blink | Restart `npm run dev` and let it idle for 30 sec before recording |
| Browser zoom not 100% | Selectors fail to find elements | Hit `Ctrl/Cmd-0` before recording starts |

## What to mark in the narration

| Phrase | Reason |
|---|---|
| "Pre-loaded data and prepared splits" | The seed jumps the project to `training` stage; if the narrator says "we're starting from scratch," that's inaccurate. |
| "Doesn't run automatically" (for cleaning, training, etc) | Codex confirmed all pipeline tabs render but no runs fire on seed. |
| "200 hand-labelled gold rows" | Verified count, but flag that the manifest text says "6" — manifest prose is stale; the file is the source of truth (`backend/data/demo_samples/support-faq/gold.jsonl`). |
| "Local API key" | Don't call it "no auth"; auth is enabled by default. |

## Open questions before scaling to videos 03–07

- Should the screenshot capture mode in Playwright run alongside the
  video, or in a separate non-recording pass? (Decision: separate
  pass to keep the video clean.)
- Should we add `data-testid="demo-tile-<slug>"` before the second
  recording series? (Proposal: yes, low-risk; see
  `11-selector-and-instrumentation-plan.md`.)

## Manual steps the recording will NOT cover

These are documented separately in the deck (Section C) but not
shown in the browser portion of this video:

- Worker startup (`celery -A app.worker.celery_app worker …`) — only
  needed for queued imports + external training; not for the seed
  flow this video covers.
- Redis startup — same.
- Setting `TEACHER_MODEL_API_KEY` / `TEACHER_MODEL_API_URL` for
  synthetic generation — that's Video 04's problem.

# Selector And Route Evidence Pass

Discovery date: 2026-05-19.

These passes used disposable local app instances and the real browser UI. They
did not create final Playwright recording specs, did not run
training/evaluation/compression/export, and did not change product behavior.

## Disposable Run Setup

| Item | Evidence |
|---|---|
| Backend | `http://127.0.0.1:8010` |
| Frontend | `http://127.0.0.1:5174` |
| Disposable data dir | `/tmp/slm-selector-pass.5AGEKQ/data` |
| Disposable database | `/tmp/slm-selector-pass.5AGEKQ/slm_platform.db` |
| Auth path used | Browser login form at `/login` |
| Login selectors used | placeholder `Enter your username`, placeholder `API Key or Password`, button role name `Sign in` |
| Username used | `admin` |
| Selected official demo sample | `support-faq` |
| Seed route observed | `POST /api/demo-projects/support-faq` returned 200 |
| Seeded project id | `1` |
| Final browser route | `/project/1/training-config` |

Notes:

- Ports `8000` and `5173` were already occupied, so this pass used alternate
  ports.
- A short-lived Vite config was used to point the frontend dev server at the
  disposable backend. It was runtime scaffolding only and should not be kept as
  product code.

## Screenshots

These files were created under `docs-demo/screenshots/`.

| Screenshot | What it proves |
|---|---|
| `selector-pass-01-login.png` | Local login form exists and is recordable. |
| `selector-pass-02-demo-tiles.png` | Demo catalog renders the three official demo tiles. |
| `selector-pass-03-support-faq-data-tab.png` | Seeded Support FAQ project opens on the data tab with pipeline progress and ingested documents. |
| `selector-pass-04-support-faq-expanded-raw-row.png` | Raw document expansion works through a `data-testid` selector. |
| `selector-pass-05-cleaning-tab.png` | Cleaning route renders. |
| `selector-pass-06-goldset-tab.png` | Gold set route renders and shows 200 entries. |
| `selector-pass-07-synthetic-tab.png` | Synthetic route renders. |
| `selector-pass-08-dataprep-tab.png` | Dataset prep route renders with preview/profile/semantic areas. |
| `selector-pass-09-tokenization-tab.png` | Tokenization route renders. |
| `selector-pass-10-training-tab.png` | Training route renders. |
| `selector-pass-11-eval-tab.png` | Evaluation route renders. |
| `selector-pass-12-compression-tab.png` | Compression route renders. |
| `selector-pass-13-export-tab.png` | Export and registry route renders. |
| `selector-pass-14-training-config.png` | Dedicated Training Config route renders. |

## Observed Browser Routes

| Route | Status | Screenshot | UI evidence |
|---|---|---|---|
| `/login` | verified | `selector-pass-01-login.png` | BrewSLM local login form. |
| `/` | verified | `selector-pass-02-demo-tiles.png` | Project list with official demo tiles. |
| `/project/1` | verified | none separate | Browser navigated here immediately after seeding. |
| `/project/1/pipeline/data` | verified | `selector-pass-03-support-faq-data-tab.png`, `selector-pass-04-support-faq-expanded-raw-row.png` | Data tab, system readiness, import controls, EDA, ingested documents. |
| `/project/1/pipeline/cleaning` | verified | `selector-pass-05-cleaning-tab.png` | Heading `Cleaning Configuration`. |
| `/project/1/pipeline/goldset` | verified | `selector-pass-06-goldset-tab.png` | Heading `Gold Evaluation Dataset`; visible `Entries 200`. |
| `/project/1/pipeline/synthetic` | verified | `selector-pass-07-synthetic-tab.png` | Heading `Synthetic Data Generation`. |
| `/project/1/pipeline/dataprep` | verified | `selector-pass-08-dataprep-tab.png` | Headings `Dataset Preview`, `Schema Profile`, `Semantic Intelligence`. |
| `/project/1/pipeline/tokenization` | verified | `selector-pass-09-tokenization-tab.png` | Tokenization tab loaded. |
| `/project/1/pipeline/training` | verified | `selector-pass-10-training-tab.png` | Training tab loaded. |
| `/project/1/pipeline/eval` | verified | `selector-pass-11-eval-tab.png` | Evaluation tab loaded. |
| `/project/1/pipeline/compression` | verified | `selector-pass-12-compression-tab.png` | Heading `Compression Engine`. |
| `/project/1/pipeline/export` | verified | `selector-pass-13-export-tab.png` | Heading `Export and Registry`. |
| `/project/1/training-config` | verified | `selector-pass-14-training-config.png` | Headings `Training Configurations`, `Create and Configure Experiment`. |

## Observed Selectors

These selectors worked in the disposable UI pass.

| UI area | Selector or locator | Status | Notes |
|---|---|---|---|
| Login username | `getByPlaceholder("Enter your username")` | verified | Filled with `admin`. |
| Login password | `getByPlaceholder("API Key or Password")` | verified | Filled with local API key. |
| Login submit | `getByRole("button", { name: /^Sign in$/ })` | verified | Navigated to `/`. |
| Demo tiles container | `.demo-project-tiles` | verified | Appears after catalog load. |
| Demo tile buttons | `.demo-project-tile` | verified | All three official samples rendered. |
| Support FAQ tile | `.demo-project-tile` filtered by visible text `Support FAQ` | verified | More robust than assuming exact punctuation. |
| Support FAQ tile aria label | `Open the Demo · Support FAQ demo project` | verified | Exact accessible name includes the manifest's centered dot. |
| PII tile aria label | `Open the Demo · PII / PCI Detector demo project` | verified | Official tile present. |
| Sentiment tile aria label | `Open the Demo · Sentiment classifier demo project` | verified | Official tile present. |
| Pipeline tab buttons | `button.tab` with `title` values such as `Data`, `Cleaning`, `Gold Set` | verified | All pipeline tabs were unlocked in the seeded project. |
| Raw document expander | `[data-testid^="expand-doc-"]` | verified | Observed concrete id `expand-doc-20`. |
| Progress chip | `[data-testid="progress-chip"]` | verified | Present in top bar. |
| Import wizard button | `[data-testid="open-import-wizard-btn"]` | verified from code and page surface | Visible on data tab as import dataset action. |

Selector caution:

- Final recording specs should not rely on the hyphenated text `Demo - Support
  FAQ`. The rendered UI uses the manifest name with a centered dot.
- Demo tiles currently have useful aria labels but no stable `data-testid`.
  Adding test ids would make long-term recordings less brittle, but that should
  be approved before product markup changes.

## Observed API Calls

All listed calls returned HTTP 200 during the UI pass.

| Method | Path | What triggered it |
|---|---|---|
| GET | `/api/auth/config` | Login page auth configuration. |
| POST | `/api/auth/local/login` | Local login form submit. |
| GET | `/api/demo-projects` | Demo tile catalog. |
| POST | `/api/demo-projects/support-faq` | Support FAQ tile click. |
| GET | `/api/projects` | Project list. |
| GET | `/api/projects/1` | Project workspace load. |
| GET | `/api/projects/1/pipeline/status` | Pipeline progress/status. |
| GET | `/api/projects/1/gamification` | Progress chip/workspace shell. |
| GET | `/api/projects/1/runtime/readiness` | Data/training readiness panels. |
| GET | `/api/projects/1/ingestion/documents` | Data tab ingested documents. |
| GET | `/api/projects/1/ingestion/eda` | Data health dashboard. |
| GET | `/api/projects/1/ingestion/documents/20/sample` | Expanded raw row. |
| GET | `/api/projects/1/gold/entries?dataset_type=gold_dev` | Gold set tab. |
| GET | `/api/projects/1/prepared-manifest` | Synthetic/data prep context. |
| GET | `/api/projects/1/pipeline/graph/contract` | Pipeline/data prep/training context. |
| GET | `/api/projects/1/dataset/adapters/catalog` | Data import/data prep context. |
| GET | `/api/projects/1/dataset/adapter-preference` | Dataset prep adapter state. |
| POST | `/api/projects/1/dataset/split/effective-config` | Dataset prep effective split config. |
| GET | `/api/projects/1/training/runtimes` | Training route/config. |
| GET | `/api/projects/1/training/recipes` | Training route/config. |
| GET | `/api/projects/1/training/preferences` | Training route/config. |
| GET | `/api/projects/1/training/experiments` | Training/eval/export context. |
| POST | `/api/projects/1/training/experiments/effective-config` | Training Config page. |
| GET | `/api/projects/1/evaluation/packs` | Evaluation tab. |
| GET | `/api/projects/1/evaluation/pack-preference` | Evaluation tab. |
| GET | `/api/projects/1/export/list` | Export tab. |
| GET | `/api/projects/1/export/deployment-targets?export_format=gguf` | Export target catalog. |
| GET | `/api/projects/1/registry/models` | Export/registry area. |

## Seeded Support FAQ Evidence

| Item | Observed value | Source |
|---|---|---|
| Project name | `Demo · Support FAQ` | `GET /api/projects/1` |
| Project pipeline stage | `training` | `GET /api/projects/1` |
| Pipeline current stage | `training` | `GET /api/projects/1/pipeline/status` |
| Pipeline progress | `60` percent | `GET /api/projects/1/pipeline/status` |
| Stage count | `10` | `GET /api/projects/1/pipeline/status` |
| Raw document count | `20` | `GET /api/projects/1/ingestion/documents` |
| First returned document | `tickets.csv#row-20` | `GET /api/projects/1/ingestion/documents` |
| Expanded raw row | document id `20` | UI selector `[data-testid="expand-doc-20"]` |
| Gold entries | `200` | Gold set UI screenshot and API route |
| Prepared total entries | `20` | `GET /api/projects/1/prepared-manifest` |
| Prepared split | train `16`, val `2`, test `2` | `GET /api/projects/1/prepared-manifest` |
| Prepared adapter | `qa-pair` | `GET /api/projects/1/prepared-manifest` |
| Prepared task profile | `instruction_sft` | `GET /api/projects/1/prepared-manifest` |
| Prepared field mapping | input `question`, output `answer` | `GET /api/projects/1/prepared-manifest` |

## Remaining Demos Disposable Run Setup

| Item | Evidence |
|---|---|
| Backend | `http://127.0.0.1:8010` |
| Frontend | `http://127.0.0.1:5174` |
| Disposable data dir | `/tmp/slm-selector-pass-remaining.8c6MCJ/data` |
| Disposable database | `/tmp/slm-selector-pass-remaining.8c6MCJ/slm_platform.db` |
| Auth path used | Browser login form at `/login` |
| Samples seeded through UI | `pii-detector`, then `sentiment-classifier` |
| PII seed route observed | `POST /api/demo-projects/pii-detector` returned 200 |
| Sentiment seed route observed | `POST /api/demo-projects/sentiment-classifier` returned 200 |

## Remaining Demo Screenshots

These files were created under `docs-demo/screenshots/`.

| Sample | Screenshot | What it proves |
|---|---|---|
| PII | `selector-pass-pii-01-demo-tile.png` | Demo tile screen before selecting PII. |
| PII | `selector-pass-pii-02-data-tab.png` | Seeded PII project data tab with 61 ingested documents. |
| PII | `selector-pass-pii-03-expanded-raw-row.png` | Raw PII row expansion via `expand-doc-61`. |
| PII | `selector-pass-pii-04-cleaning-tab.png` | Cleaning route renders. |
| PII | `selector-pass-pii-05-goldset-tab.png` | Gold set route renders and shows 200 entries. |
| PII | `selector-pass-pii-06-synthetic-tab.png` | Synthetic route renders. |
| PII | `selector-pass-pii-07-dataprep-tab.png` | Dataset prep route renders. |
| PII | `selector-pass-pii-08-tokenization-tab.png` | Tokenization route renders. |
| PII | `selector-pass-pii-09-training-tab.png` | Training route renders. |
| PII | `selector-pass-pii-10-eval-tab.png` | Evaluation route renders. |
| PII | `selector-pass-pii-11-compression-tab.png` | Compression route renders. |
| PII | `selector-pass-pii-12-export-tab.png` | Export/registry route renders. |
| PII | `selector-pass-pii-13-training-config.png` | Dedicated Training Config route renders. |
| Sentiment | `selector-pass-sentiment-01-demo-tile.png` | Demo tile screen before selecting sentiment. |
| Sentiment | `selector-pass-sentiment-02-data-tab.png` | Seeded sentiment project data tab with 30 ingested documents. |
| Sentiment | `selector-pass-sentiment-03-expanded-raw-row.png` | Raw sentiment row expansion via `expand-doc-91`. |
| Sentiment | `selector-pass-sentiment-04-cleaning-tab.png` | Cleaning route renders. |
| Sentiment | `selector-pass-sentiment-05-goldset-tab.png` | Gold set route renders and shows 200 entries. |
| Sentiment | `selector-pass-sentiment-06-synthetic-tab.png` | Synthetic route renders. |
| Sentiment | `selector-pass-sentiment-07-dataprep-tab.png` | Dataset prep route renders. |
| Sentiment | `selector-pass-sentiment-08-tokenization-tab.png` | Tokenization route renders. |
| Sentiment | `selector-pass-sentiment-09-training-tab.png` | Training route renders. |
| Sentiment | `selector-pass-sentiment-10-eval-tab.png` | Evaluation route renders. |
| Sentiment | `selector-pass-sentiment-11-compression-tab.png` | Compression route renders. |
| Sentiment | `selector-pass-sentiment-12-export-tab.png` | Export/registry route renders. |
| Sentiment | `selector-pass-sentiment-13-training-config.png` | Dedicated Training Config route renders. |

## Seeded PII Detector Evidence

| Item | Observed value | Source |
|---|---|---|
| Project id | `1` | Browser URL and `GET /api/projects/1` |
| Project name | `Demo · PII / PCI Detector` | `GET /api/projects/1` |
| Project pipeline stage | `training` | `GET /api/projects/1` |
| Pipeline current stage | `training` | `GET /api/projects/1/pipeline/status` |
| Pipeline progress | `60` percent | `GET /api/projects/1/pipeline/status` |
| Browser routes visited | `/`, `/project/1`, `/project/1/pipeline/data`, `/project/1/pipeline/cleaning`, `/project/1/pipeline/goldset`, `/project/1/pipeline/synthetic`, `/project/1/pipeline/dataprep`, `/project/1/pipeline/tokenization`, `/project/1/pipeline/training`, `/project/1/pipeline/eval`, `/project/1/pipeline/compression`, `/project/1/pipeline/export`, `/project/1/training-config` | Browser route capture |
| Raw document count | `61` | Data tab UI and `GET /api/projects/1/ingestion/documents` |
| Expanded raw row | document id `61`, selector `[data-testid="expand-doc-61"]` | Data tab UI |
| Expanded raw row shape | `text`, `entities_json` | `GET /api/projects/1/ingestion/documents/61/sample` |
| Gold entries | `200` | Gold set UI and `GET /api/projects/1/gold/entries?dataset_type=gold_dev` |
| Prepared split | train `45`, val `8`, test `8` | `GET /api/projects/1/prepared-manifest` |
| Prepared adapter | `structured-extraction` | `GET /api/projects/1/prepared-manifest` |
| Prepared task profile | `structured_extraction` | Demo tile badge and prepared manifest API |
| Prepared field mapping | input `text`, output `entities_json` | `GET /api/projects/1/prepared-manifest` |
| Schema surfaced | `output_schema.scoring_mode=span_set`, required `entities` array | `GET /api/projects/1/prepared-manifest` |
| Entity types surfaced | `email`, `phone`, `ssn`, `credit_card`, `person_name`, `street_address`, `date_of_birth`, `ip_address`, `api_key`, `bank_account` | `GET /api/projects/1/prepared-manifest` |
| Runtime warnings shown | `WARN`; missing `TEACHER_MODEL_API_KEY`; synthetic tab says no text loaded yet; training tab says no experiments yet; eval tab says no experiments to evaluate | UI text capture and screenshots |

PII selectors that worked:

- Login: placeholder `Enter your username`, placeholder `API Key or Password`,
  button role name `Sign in`.
- Demo tiles: `.demo-project-tiles`, `.demo-project-tile`, filtered by visible
  text `PII / PCI Detector`.
- Raw row expansion: `[data-testid^="expand-doc-"]`, concrete id
  `expand-doc-61`.
- Pipeline tabs: `button.tab` with title values such as `Data`, `Cleaning`,
  `Gold Set`, `Synthetic`, `Dataset Prep`, `Tokenization`, `Training`,
  `Evaluation`, `Compression`, `Export`.

PII API calls observed:

- `GET /api/demo-projects`
- `POST /api/demo-projects/pii-detector`
- `GET /api/projects/1`
- `GET /api/projects/1/pipeline/status`
- `GET /api/projects/1/ingestion/documents`
- `GET /api/projects/1/ingestion/eda`
- `GET /api/projects/1/ingestion/documents/61/sample`
- `GET /api/projects/1/gold/entries?dataset_type=gold_dev`
- `GET /api/projects/1/prepared-manifest`
- `POST /api/projects/1/dataset/split/effective-config`
- `GET /api/projects/1/training/runtimes`
- `GET /api/projects/1/training/experiments`
- `GET /api/projects/1/evaluation/packs`
- `GET /api/projects/1/export/list`
- `GET /api/projects/1/export/deployment-targets?export_format=gguf`
- `GET /api/projects/1/registry/models`
- `POST /api/projects/1/training/experiments/effective-config`

## Seeded Sentiment Classifier Evidence

| Item | Observed value | Source |
|---|---|---|
| Project id | `2` | Browser URL and `GET /api/projects/2` |
| Project name | `Demo · Sentiment classifier` | `GET /api/projects/2` |
| Project pipeline stage | `training` | `GET /api/projects/2` |
| Pipeline current stage | `training` | `GET /api/projects/2/pipeline/status` |
| Pipeline progress | `60` percent | `GET /api/projects/2/pipeline/status` |
| Browser routes visited | `/`, `/project/2`, `/project/2/pipeline/data`, `/project/2/pipeline/cleaning`, `/project/2/pipeline/goldset`, `/project/2/pipeline/synthetic`, `/project/2/pipeline/dataprep`, `/project/2/pipeline/tokenization`, `/project/2/pipeline/training`, `/project/2/pipeline/eval`, `/project/2/pipeline/compression`, `/project/2/pipeline/export`, `/project/2/training-config` | Browser route capture |
| Raw document count | `30` | Data tab UI and `GET /api/projects/2/ingestion/documents` |
| Expanded raw row | document id `91`, selector `[data-testid="expand-doc-91"]` | Data tab UI |
| Expanded raw row shape | `text`, `label` | `GET /api/projects/2/ingestion/documents/91/sample` |
| Gold entries | `200` | Gold set UI and `GET /api/projects/2/gold/entries?dataset_type=gold_dev` |
| Prepared split | train `22`, val `4`, test `4` | `GET /api/projects/2/prepared-manifest` |
| Prepared adapter | `classification-label` | `GET /api/projects/2/prepared-manifest` |
| Prepared task profile | `classification` | Demo tile badge and prepared manifest API |
| Prepared field mapping | input `text`, output `label` | `GET /api/projects/2/prepared-manifest` |
| Labels surfaced | `positive`, `neutral`, `negative` | `GET /api/projects/2/prepared-manifest` |
| Runtime warnings shown | `WARN`; missing `TEACHER_MODEL_API_KEY`; synthetic tab says no text loaded yet; training tab says no experiments yet; eval tab says no experiments to evaluate | UI text capture and screenshots |

Sentiment selectors that worked:

- Login: placeholder `Enter your username`, placeholder `API Key or Password`,
  button role name `Sign in`.
- Demo tiles: `.demo-project-tiles`, `.demo-project-tile`, filtered by visible
  text `Sentiment classifier`.
- Raw row expansion: `[data-testid^="expand-doc-"]`, concrete id
  `expand-doc-91`.
- Pipeline tabs: `button.tab` with title values such as `Data`, `Cleaning`,
  `Gold Set`, `Synthetic`, `Dataset Prep`, `Tokenization`, `Training`,
  `Evaluation`, `Compression`, `Export`.

Sentiment API calls observed:

- `GET /api/demo-projects`
- `POST /api/demo-projects/sentiment-classifier`
- `GET /api/projects/2`
- `GET /api/projects/2/pipeline/status`
- `GET /api/projects/2/ingestion/documents`
- `GET /api/projects/2/ingestion/eda`
- `GET /api/projects/2/ingestion/documents/91/sample`
- `GET /api/projects/2/gold/entries?dataset_type=gold_dev`
- `GET /api/projects/2/prepared-manifest`
- `POST /api/projects/2/dataset/split/effective-config`
- `GET /api/projects/2/training/runtimes`
- `GET /api/projects/2/training/experiments`
- `GET /api/projects/2/evaluation/packs`
- `GET /api/projects/2/export/list`
- `GET /api/projects/2/export/deployment-targets?export_format=gguf`
- `GET /api/projects/2/registry/models`
- `POST /api/projects/2/training/experiments/effective-config`

## Recording Implications

- A prototype selector pass can use real UI login and real tile seeding.
- The seeded project starts at the `training` stage and the browser shows 60
  percent progress, so narration should explain that the official demo seed
  preloads raw data, gold data, and prepared splits.
- All pipeline tabs were unlocked in all three seeded official demo projects
  during selector passes.
- The data tab shows system readiness warnings, including missing teacher-model
  secrets. This is useful evidence for runtime prerequisites, not a failure.
- The data health dashboard includes estimated values such as estimated rows and
  duplicate ratio. Those should be narrated as estimates, not source-file facts.
- Gold set UI confirmed 200 entries visually for support, PII, and sentiment.
- Training, evaluation, compression, export, registry, and final model usage
  routes rendered, but no run/artifact was produced in this pass.

## Gaps Before Real Recording Specs

- Confirm whether to add `data-testid` values to demo tiles and pipeline tabs.
- Decide whether recordings should use UI login or pre-authenticated storage.
- Pick a canonical sample for the first full recording. Support FAQ is viable
  for route coverage, but runtime-heavy steps remain unresolved.
- Run a controlled runtime pass before claiming training, evaluation,
  compression, export, registry promotion, or final model API/UI usage.
- Decide whether PII and sentiment recordings should include cleaning before the
  sample-specific pipeline tabs, because cleaning is visible but not required by
  the seeded prepared manifest.

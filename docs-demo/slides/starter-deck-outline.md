# Starter Deck Outline

Every slide entry includes title, audience level, visual idea, talking notes summary, evidence needed, and video module mapping.

## Section A: SLM 101

| Slide title | Audience level | Visual idea | Talking notes summary | Evidence needed | Video module mapping |
|---|---|---|---|---|---|
| What Is An SLM? | beginner | Simple model-size spectrum | Define SLM conceptually without product claims. | None; conceptual. | 01 |
| Why Smaller Models Matter | beginner | Cost/latency/private deployment triangle | Smaller models can be practical for narrow tasks. | None; conceptual. | 01 |
| Key Terms | beginner | Vocabulary cards | LLM, SLM, fine-tuned model, adapter, dataset, gold set, synthetic data. | None; conceptual. | 01 |
| SLM Lifecycle | beginner | Linear lifecycle diagram | Data to evaluation to export/usage. | None; conceptual. | 01 |

## Section B: BrewSLM 101

| Slide title | Audience level | Visual idea | Talking notes summary | Evidence needed | Video module mapping |
|---|---|---|---|---|---|
| BrewSLM Workspace | beginner | Screenshot placeholder: project list | This repo appears to provide a project-based SLM lifecycle app. | `README.md`, `ProjectPipelinePage.tsx`; screenshot needed. | 02 |
| Product Workflow | beginner | Pipeline tab strip | Data, cleaning, gold set, synthetic, prep, tokenization, training, evaluation, compression, export. | `ProjectPipelinePage.tsx`; screenshot needed. | 02 |
| High-Level Architecture | intermediate | Browser -> Vite -> FastAPI -> services/data | React frontend and FastAPI backend. | `frontend/vite.config.ts`, `backend/app/main.py`. | 02 |
| What Is Still To Verify | beginner | Checklist | Runtime-specific parts need real runs. | Evidence docs. | 02 |

## Section C: Quickstart

| Slide title | Audience level | Visual idea | Talking notes summary | Evidence needed | Video module mapping |
|---|---|---|---|---|---|
| Local Services | beginner | Terminal commands | Start backend and frontend. | `README.md`, `frontend/package.json`. | 02 |
| Login | beginner | Login screenshot placeholder | Local login uses any username and API key password when auth is enabled. | `SSOLoginPage.tsx`, `auth.py`, `.env.example`. | 02 |
| Pick A Demo | beginner | Demo tiles screenshot placeholder | Use only the three official samples. | `DemoProjectTiles.tsx`, sample folders. | 02 |
| First Pipeline Walkthrough | beginner | Project pipeline screenshot placeholder | Inspect tabs before running heavy jobs. | `ProjectPipelinePage.tsx`. | 02 |

## Section D: Dataset Lifecycle

| Slide title | Audience level | Visual idea | Talking notes summary | Evidence needed | Video module mapping |
|---|---|---|---|---|---|
| Ingest | beginner | CSV -> raw documents | Demo seeder creates raw dataset/docs; import UI exists. | `demo_project_service.py`, `IngestionPanel.tsx`. | 03-06 |
| Inspect Raw Data | beginner | Document sample panel | Show row samples before transformations. | `DocumentSampleAccordion.tsx`; screenshot needed. | 03-06 |
| Clean | beginner | Raw -> cleaned chunks | Cleaning batch UI/API exists. | `CleaningPanel.tsx`, `cleaning.py`; output needed. | 03-06 |
| Normalize | intermediate | Field mapping diagram | Seeder canonicalizes fields; adapter preview supports mappings. | `demo_project_service.py`, `DatasetPrepPanel.tsx`. | 03-06 |
| Deduplicate If Supported | intermediate | Duplicate rows crossed out | Gold workbench mentions dedup; broader support needs verification. | `GoldSetWorkbenchPanel.tsx`; output needed. | 03-06 |
| Gold Set | beginner | Trusted examples table | Official samples include gold JSONL. | sample `gold.jsonl`, `gold.py`. | 03-06 |
| Synthetic Data | intermediate | Teacher/fallback path | Synthetic APIs exist; runtime must be configured. | `SyntheticPanel.tsx`, `synthetic.py`. | 03-06 |
| Dataset Prep | intermediate | Train/val/test split | Seeder and UI support splits. | `demo_project_service.py`, `dataset.py`. | 03-06 |
| Quality Checks | intermediate | Gate indicator | Adapter preview and evaluation gates. | `dataset.py`, `evaluation.py`. | 03-06 |

## Section E: Official Sample Demos

| Slide title | Audience level | Visual idea | Talking notes summary | Evidence needed | Video module mapping |
|---|---|---|---|---|---|
| support-faq | beginner | Q&A table | Instruction SFT support assistant. | `support-faq/manifest.json`, `tickets.csv`. | 03 |
| pii-detector | intermediate | Span offset annotation | Structured extraction with entity spans. | `pii-detector/manifest.json`, `pii_records.csv`. | 04 |
| sentiment-classifier | beginner | Label distribution | Three-way review classifier. | `sentiment-classifier/manifest.json`, `reviews.csv`. | 05 |
| Where Samples Live | beginner | Folder tree | Official templates are exactly three folders. | `backend/data/demo_samples/*`. | 03-05 |
| Real Steps Per Sample | intermediate | Evidence table | Show verified, partial, unknown per sample. | `06-official-demo-samples-map.md`, `07-pipeline-step-evidence.md`. | 03-05 |

## Section F: Beyond Official Samples

| Slide title | Audience level | Visual idea | Talking notes summary | Evidence needed | Video module mapping |
|---|---|---|---|---|---|
| Bring Your Own Dataset | intermediate | Upload/import UI screenshot placeholder | Custom demos start from repo import paths. | `DatasetImportWizard.tsx`; screenshot needed. | 06 |
| Custom Pipeline | intermediate | Branching pipeline diagram | Mark unsupported or unknown steps. | `08-outside-template-demo-ideas.md`. | 06 |
| Limits | intermediate | Caveat callouts | Runtime and external dependencies matter. | `04-recording-plan.md`. | 06 |

## Section G: Evaluation, Compression, Export

| Slide title | Audience level | Visual idea | Talking notes summary | Evidence needed | Video module mapping |
|---|---|---|---|---|---|
| Evaluation | intermediate | Scorecard screenshot placeholder | Held-out eval, scorecards, gates exist. | `EvalPanel.tsx`, `evaluation.py`; output needed. | 03-07 |
| Compression | advanced | Quantize/merge/benchmark controls | Runtime-dependent compression paths exist. | `CompressionPanel.tsx`, `compression.py`; output needed. | 07 |
| Export | advanced | Export history screenshot placeholder | Export and serve-plan APIs exist. | `ExportPanel.tsx`, `export.py`; output needed. | 07 |

## Section H: Final Model Usage

| Slide title | Audience level | Visual idea | Talking notes summary | Evidence needed | Video module mapping |
|---|---|---|---|---|---|
| What Happens After Training? | beginner | Decision tree | Test, compare, export, deploy, monitor. | `09-final-model-usage-plan.md`; run evidence needed. | 07 |
| Playground | intermediate | Playground screenshot placeholder | Possible final testing UI. | `ProjectPlaygroundPage.tsx`; output needed. | 07 |
| Registry And Deployment | advanced | Promotion stages | Registry and deployment APIs exist. | `registry.py`, `deployments.py`; output needed. | 07 |

## Section I: Advanced Topics

| Slide title | Audience level | Visual idea | Talking notes summary | Evidence needed | Video module mapping |
|---|---|---|---|---|---|
| Governance | advanced | Gate policy flow | Quality gates and audit matter. | `projects.py`, `evaluation.py`. | 10 |
| Registry | advanced | Model stages | Register/promote/deploy. | `registry.py`. | 10 |
| Secrets | advanced | Key management UI placeholder | Secrets support runtime integrations. | `secrets.py`, UI needed. | 10 |
| Recipes | advanced | Workflow graph placeholder | Pipeline recipes and graph APIs exist. | `pipeline.py`. | 10 |
| Automation | advanced | CLI/API diagram | Use APIs/CLI after paths are verified. | CLI docs, API docs. | 10 |
| Architecture | advanced | Service map | React/Vite/FastAPI/services/data. | `backend/app/main.py`, `frontend/vite.config.ts`. | 10 |


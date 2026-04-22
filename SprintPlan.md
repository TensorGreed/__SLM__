# Sprint Plan: Production-Grade SLM Builder for Newbie ML Engineers

## Goal
Turn this repository into a production-grade platform that lets a new machine learning engineer build, evaluate, export, and operate a domain-specific SLM from almost any domain brief, base model, and data structure without needing to hand-author Python plugins for the common path.

## Repo Evaluation

### Current strengths
- The repo already has a serious end-to-end foundation: FastAPI backend, React frontend, documented CLI, workflow runner, export/compression paths, domain packs/profiles, starter packs, training runtimes, evaluation packs, and project secrets.
- The architecture is already extensible: there are plugin templates for data adapters and training runtimes, typed schemas, workflow contracts, and a large backend test surface.
- The docs are beginner-aware, and the product already attempts guided flows with Autopilot, starter packs, project wizard screens, and training preflight.

### Current gaps blocking the “newbie can build any SLM” goal
- The beginner experience still exposes too many advanced concepts too early. A new user must mentally model domain packs, domain profiles, adapters, runtimes, recipes, targets, and evaluation packs before they are confident.
- The current autopilot logic is still heuristic-heavy and intent-preset driven. That is not enough for “any domain/base model/data structure”.
- The model story is flexible but not yet unified around a first-class, normalized model registry that explains family compatibility, tokenizer/chat-template requirements, training constraints, license concerns, and deployment tradeoffs in one place.
- The data adapter flow is powerful, but it still assumes a row-mapping mindset and some plugin authoring literacy. A newbie needs a schema explorer, mapping studio, and adapter generation flow that works for tabular, nested JSON, relational, document, and preference data.
- The CLI is useful, but it is not yet a full pipeline-as-code interface that an agent or CI system can use to reproduce the entire project lifecycle from a manifest.
- Production operations are present in pieces, but the platform still needs stronger observability, support bundles, smoke-test automation, rollback posture, and end-to-end docs/sample projects.

### Grounded observations from this repo
- The project already includes documented CLI commands, but they are still described as a lightweight wrapper.
- The plugin system already has templates for data adapters and training runtimes, which is a strong base for future scaffolding.
- The backend test surface is substantial, but the frontend coverage is still comparatively narrow and there is no obvious full browser-level end-to-end suite.
- The current newbie autopilot service uses static preset-style intent logic, which is a good bootstrap but not a universal planner.

## Product Outcome
After this plan is complete, a new ML engineer should be able to:

1. Describe a domain and target behavior in plain English.
2. Point the system at local or remote data in common formats.
3. Import or choose a base model with clear compatibility guidance.
4. Let the system infer the right task, schema, adapter, recipe, and evaluation plan.
5. Review a transparent dry-run with safe defaults and one-click repairs.
6. Train, evaluate, export, deploy, and reproduce the workflow from either UI or CLI.
7. Extend the system for uncommon domains/models/data structures with generated scaffolds instead of hand-written boilerplate.

## Global Delivery Standard
Every story below should be implemented to the same production-grade bar.

- Preserve existing behavior unless the story explicitly replaces it.
- Add or update database migrations for any persisted schema changes.
- Use typed backend schemas, typed frontend client calls, and stable machine-readable error codes.
- Protect new write APIs with the same authz/audit expectations used elsewhere in the platform.
- Every long-running action must support status, cancel, retry, and log visibility.
- Every CLI command must support non-interactive mode, meaningful exit codes, and `--json` output.
- Every UI flow must handle loading, empty, success, warning, and error states.
- Every auto-decision must expose provenance, confidence, and a human-readable explanation.
- Every feature must update docs and include realistic example fixtures.
- Every feature must ship with backend unit tests, API integration tests, frontend tests, CLI tests, and at least one end-to-end workflow test where appropriate.

## Release Exit Criteria
- A first-time user can complete a zero-code project from domain brief to exported artifact in under 30 minutes using docs and built-in guidance alone.
- The platform supports at least these input structure families: tabular, nested JSON, chat transcripts, pairwise preference data, structured extraction data, and document/chunk corpora.
- The platform supports at least these base model families with explicit compatibility handling: instruction-tuned causal LM, seq2seq, encoder/classifier, and LoRA/QLoRA-compatible adapter training targets.
- Every major workflow is reproducible from a manifest and debuggable from a support bundle.

## Status Snapshot — 22 April 2026

Legend: ✅ done · ⚠️ partial · ❌ not started

| # | Story | Status |
|---|-------|--------|
| 1 | Beginner Mode and Domain Brief Wizard | ✅ **DONE** |
| 2 | Universal Base Model Registry and Compatibility Engine | ✅ **DONE** |
| 3 | Dataset Structure Explorer and Adapter Studio | ✅ **DONE** |
| 4 | Autopilot v3 Planner with Explainability and One-Click Repair | ⚠️ **PARTIAL** — v2 orchestration shipped; missing decision-log persistence, rollback, repair preview, and `autopilot` CLI |
| 5 | Evaluation Pack Generator and Gold Set Workbench | ⚠️ **PARTIAL** — eval packs + gates + remediation shipped; gold set is add/import/lock only; missing annotation queue, rubric versioning, failure-cluster UI, and `eval` CLI |
| 6 | Training Runtime Planner and Reproducibility Manifests | ⚠️ **PARTIAL** — planner + preflight + recipes shipped; missing full reproducibility manifest contract, rerun/clone, and `train rerun / clone / repro` CLI |
| 7 | Pipeline-as-Code and Full CLI Parity | ❌ **NOT STARTED** — pipeline recipes exist but no `brewslm.yaml` round-trip or `manifest` CLI |
| 8 | Extension Scaffolder and Plugin Contract Validator | ❌ **NOT STARTED** — adapter-only scaffold export exists; no runtime/domain/eval scaffolds, no contract validator, no Extension Studio |
| 9 | Deployment Assistant, Target Smoke Tests, and Rollback | ⚠️ **PARTIAL** — deployment target catalog + smoke tests + deploy-as-api shipped; missing deployment rollback, promote/reject flow, and `deploy` CLI |
| 10 | Observability, Failure Analysis, and Support Bundles | ❌ **NOT STARTED** — audit log list exists; no support-bundle service, no failure taxonomy, no unified timeline, no `support-bundle / logs tail / doctor --deep` CLI |
| 11 | Quality Bar, Sample Projects, and Guided Learning Mode | ❌ **NOT STARTED** — starter packs + 6 sample projects on disk; no Guided Learning UI, no `demo / tutorial` CLI, no Playwright/Cypress E2E suite |

Per-story verification detail is recorded in a **Status Notes (22 Apr 2026)** block at the end of each story below.

## Sprint 1: Universal Onboarding Foundation

### Story 1: Beginner Mode and Domain Brief Wizard ✅ DONE

User story: As a newbie ML engineer, I want to describe my use case, data examples, desired outputs, and deployment constraints in plain English so that the platform can scaffold the right SLM project without requiring me to know platform jargon.

Priority: P0
Dependencies: None

Prompt for coding agent:

```text
Implement a production-grade Beginner Mode and Domain Brief Wizard across the FastAPI backend, React frontend, and brewslm CLI.

Build a new first-class "Domain Blueprint" concept that captures:
- domain name
- problem statement
- target user persona
- task family
- input modality
- expected output schema or output examples
- safety/compliance notes
- deployment target constraints
- success metrics
- glossary and jargon explanations
- confidence and unresolved assumptions

Backend work:
- Add persisted domain blueprint models, schemas, migrations, services, and artifact lineage.
- Create deterministic parsing/normalization of user-provided briefs and examples.
- Support optional LLM-assisted enrichment only when configured, with deterministic fallback when not configured.
- Add server-side validators that reject contradictory blueprints and return actionable errors.
- Version blueprint changes and attach them to project history.

API work:
- Add endpoints to analyze a raw brief, save a draft blueprint, apply a blueprint to a project, compare blueprint revisions, and fetch glossary/help text.
- Return machine-readable guidance, unresolved questions, and recommended next actions.

Frontend work:
- Add a Beginner Mode toggle in project creation and workspace onboarding.
- Build a multi-step wizard that asks for plain-English intent, sample inputs, sample outputs, risk constraints, and deployment target.
- Show a live "What the system understood" panel with task type, output contract, and assumptions.
- Add a jargon translator panel so advanced concepts are translated into beginner-friendly language.

CLI work:
- Add `brewslm project bootstrap` with flags for brief text, brief file, sample inputs, sample outputs, target, and `--create-project`.
- Add `brewslm project blueprint show` and `brewslm project blueprint diff`.

Docs work:
- Update quickstart and first-project docs around Beginner Mode.
- Add at least two sample briefs and their resulting blueprints.

Acceptance criteria:
- A user can create a project from a plain-English brief without manually selecting domain packs, adapters, and task profiles on the first screen.
- The saved blueprint is versioned, inspectable, and reusable.
- The system explains every inferred field and clearly marks low-confidence assumptions.
- The feature works without external LLM dependencies.

Required tests:
- Backend unit tests for brief parsing, contradiction detection, glossary generation, and revision diffing.
- API tests for create/read/update/apply blueprint flows.
- Frontend tests for wizard happy path, ambiguous brief flow, and error states.
- CLI tests for bootstrap, show, and diff commands with `--json`.
- End-to-end test that creates a new project from a brief and verifies persisted blueprint state.
```

#### Status Notes — 22 Apr 2026  ✅ DONE
- Backend: ✅ `DomainBlueprintRevision` model + migration `20260311_0011_domain_blueprints.py`; `domain_blueprint_service.py` with deterministic parsing and LLM-optional enrichment.
- API: ✅ analyze / save draft / list revisions / diff / apply-to-project / glossary-help in [backend/app/api/domain_blueprints.py](backend/app/api/domain_blueprints.py).
- Frontend: ✅ wizard in [frontend/src/pages/ProjectWizardPage.tsx](frontend/src/pages/ProjectWizardPage.tsx); glossary rendered in [frontend/src/pages/ProjectListPage.tsx](frontend/src/pages/ProjectListPage.tsx:584); `beginner_mode` toggle in [frontend/src/pages/ProjectGuidePage.tsx](frontend/src/pages/ProjectGuidePage.tsx:84).
- CLI: ✅ `project bootstrap`, `project blueprint show`, `project blueprint diff` in [backend/scripts/brewslm.py](backend/scripts/brewslm.py:1324).
- Docs: ✅ [slm-docs/docs/getting-started/quickstart.md](slm-docs/docs/getting-started/quickstart.md), [first-project.md](slm-docs/docs/getting-started/first-project.md).
- Tests: ✅ [backend/tests/test_phase48_domain_blueprints.py](backend/tests/test_phase48_domain_blueprints.py), [test_phase48_domain_blueprint_cli.py](backend/tests/test_phase48_domain_blueprint_cli.py), [frontend/src/pages/ProjectWizardPage.test.tsx](frontend/src/pages/ProjectWizardPage.test.tsx).
- Minor polish only: a contextual "jargon translator" overlay (concept alias + popover everywhere in the UI) is not implemented — only the wizard-stage glossary is. Treat as nice-to-have for a later polish pass, not as blocking Story 1's acceptance criteria.

### Story 2: Universal Base Model Registry and Compatibility Engine ✅ DONE
User story: As a newbie ML engineer, I want the system to explain which base models are safe and compatible for my task, data, hardware, and deployment target so that I do not choose a model that will fail later.

Priority: P0
Dependencies: Story 1

Prompt for coding agent:

```text
Implement a production-grade Universal Base Model Registry and Compatibility Engine.

Create a normalized model registry that can ingest models from:
- Hugging Face model ids
- local filesystem paths
- pre-registered internal catalogs

Backend work:
- Add model registry persistence for model family, architecture, tokenizer, chat template, context length, parameter count, license, modalities, quantization support, PEFT support, full fine-tune support, supported task families, estimated hardware needs, and deployment target compatibility.
- Extend existing model introspection services to normalize these fields into a stable contract.
- Add compatibility scoring between model + domain blueprint + dataset adapter + runtime + deployment target.
- Add cache refresh and provenance metadata for imported model records.

API work:
- Add endpoints to import model metadata, refresh it, validate a model against a project, list compatible models, and explain incompatibilities.
- Return explicit reason codes and unblock actions when a model is not suitable.

Frontend work:
- Add a model registry page and a project-scoped model chooser with filters for family, size, license, hardware fit, context length, and training mode.
- Show "Why recommended" and "Why risky" cards for each model.
- Show tokenizer/chat-template warnings before training starts.

CLI work:
- Add `brewslm models import`, `brewslm models refresh`, `brewslm models list`, `brewslm models recommend`, and `brewslm models validate`.
- Ensure `--json` output exposes the same compatibility reasons as the API.

Docs work:
- Add a beginner guide that explains causal LM vs seq2seq vs encoder/classifier choices.
- Add a model-family compatibility matrix for the platform.

Acceptance criteria:
- A user can register or inspect a base model before training and understand whether it fits the project.
- The system can explain compatibility across task, tokenizer/chat template, hardware, and deployment target.
- Model recommendations are grounded in explicit capability contracts, not only hard-coded name heuristics.

Required tests:
- Backend unit tests for model normalization across multiple families.
- API tests for import, refresh, validate, and recommend flows.
- Frontend tests for filter/search behavior, recommendation explanations, and incompatibility warnings.
- CLI tests for import/list/validate with both success and failure cases.
- Integration tests covering at least one causal LM, one seq2seq model, and one classifier family.
```

#### Status Notes — 22 Apr 2026  ✅ DONE
- Backend: ✅ `BaseModelRegistry` model + migration `20260312_0012_universal_base_model_registry.py`; `base_model_registry_service.py` normalizes family / tokenizer / chat template / context / params / license / modalities / PEFT / full-FT / task families / hardware / deployment compatibility; `evaluate_project_model_compatibility()` scoring.
- API: ✅ import / refresh / list / get / validate / recommend / explain-incompatibility in [backend/app/api/models.py](backend/app/api/models.py).
- Frontend: ✅ [frontend/src/pages/ProjectModelsPage.tsx](frontend/src/pages/ProjectModelsPage.tsx) with filter and recommendation surfaces.
- CLI: ✅ `models import/refresh/list/recommend/validate` with `--json` in [backend/scripts/brewslm.py](backend/scripts/brewslm.py:1488).
- Docs: ✅ [slm-docs/docs/getting-started/model-family-guide.md](slm-docs/docs/getting-started/model-family-guide.md), [slm-docs/docs/reference/model-compatibility-matrix.md](slm-docs/docs/reference/model-compatibility-matrix.md).
- Tests: ✅ [test_phase49_universal_model_registry.py](backend/tests/test_phase49_universal_model_registry.py), [test_phase49_universal_model_registry_cli.py](backend/tests/test_phase49_universal_model_registry_cli.py), [ProjectModelsPage.test.tsx](frontend/src/pages/ProjectModelsPage.test.tsx).

### Story 3: Dataset Structure Explorer and Adapter Studio ✅ DONE
User story: As a newbie ML engineer, I want to bring in data from many formats and have the system help me map it into a training-ready structure so that I do not need to write a Python adapter for common cases.

Priority: P0
Dependencies: Story 1

Prompt for coding agent:

```text
Implement a production-grade Dataset Structure Explorer and Adapter Studio.

Support common source types:
- CSV/TSV
- JSON/JSONL
- Parquet
- Hugging Face datasets
- SQL query result snapshots
- document/chunk corpora
- chat transcripts
- pairwise preference datasets

Backend work:
- Add a schema profiling service that infers field types, nested structures, null rates, label distributions, sequence lengths, document lengths, and potential PII/sensitive columns.
- Introduce a stored adapter definition format that can be authored without Python for common mappings.
- Add adapter inference, preview, validation, and versioning services.
- Allow adapter export to plugin/template form for advanced extension.

API work:
- Add endpoints to profile a dataset, infer an adapter, preview transformed rows, save adapter versions, validate adapter coverage, and export adapter scaffolds.

Frontend work:
- Build a visual Adapter Studio with a schema explorer, field mapping canvas, task/output contract alignment panel, transformed row preview, drop/error analysis, and auto-fix suggestions.

CLI work:
- Add `brewslm dataset profile`, `brewslm adapter infer`, `brewslm adapter validate`, `brewslm adapter preview`, and `brewslm adapter export`.

Docs work:
- Add format-specific examples for tabular QA, extraction JSON, chat SFT, and preference data.

Acceptance criteria:
- A user can transform at least one dataset from each supported source family into a validated training-ready contract without writing Python.
- The system highlights dropped rows, unmapped fields, type conflicts, and suggested fixes before training.
- Adapter definitions are versioned and reusable across projects.

Required tests:
- Backend unit tests for schema profiling and adapter inference across tabular, nested JSON, chat, preference, and extraction fixtures.
- API tests for profile/infer/preview/save/export flows.
- Frontend tests for mapping UI, preview UI, and error summaries.
- CLI tests for profile/infer/validate/export.
- End-to-end test that imports a messy dataset, accepts an inferred adapter, and produces a valid prepared split.
```

#### Status Notes — 22 Apr 2026  ✅ DONE
- Backend: ✅ `DatasetAdapterDefinition` model + migration `20260313_0013_dataset_adapter_definitions.py`; `adapter_studio_service.py` covers profiling, inference, preview, validation, versioning, and `export_adapter_scaffold()` for Python-plugin export.
- API: ✅ profile / infer / preview / validate / save / export in [backend/app/api/adapter_studio.py](backend/app/api/adapter_studio.py).
- Frontend: ✅ [frontend/src/pages/ProjectAdapterStudioPage.tsx](frontend/src/pages/ProjectAdapterStudioPage.tsx) with schema explorer, mapping canvas, preview, and validation surfaces.
- CLI: ✅ `dataset profile`, `adapter infer / preview / validate / export` in [backend/scripts/brewslm.py](backend/scripts/brewslm.py:1412).
- Docs: ✅ [slm-docs/docs/getting-started/adapter-studio-examples.md](slm-docs/docs/getting-started/adapter-studio-examples.md).
- Tests: ✅ [test_phase50_adapter_studio.py](backend/tests/test_phase50_adapter_studio.py), [test_phase50_adapter_studio_cli.py](backend/tests/test_phase50_adapter_studio_cli.py), [ProjectAdapterStudioPage.test.tsx](frontend/src/pages/ProjectAdapterStudioPage.test.tsx).

## Sprint 2: Safe Automation and Evaluation

### Story 4: Autopilot v3 Planner with Explainability and One-Click Repair ⚠️ PARTIAL
User story: As a newbie ML engineer, I want the system to produce a safe, transparent end-to-end plan and repair common blockers automatically so that I can move forward without guessing.

Priority: P0
Dependencies: Stories 1, 2, and 3

Prompt for coding agent:

```text
Implement Autopilot v3 as a production-grade planning and repair system that replaces intent-only heuristics with capability-driven planning.

Planner inputs must include:
- domain blueprint
- dataset profile and adapter quality
- model registry capabilities
- hardware/runtime availability
- budget/time preference
- deployment target
- evaluation requirements

Backend work:
- Add a planner service that produces an explicit execution plan with recommended task profile, adapter, base model, training recipe, evaluation pack, export target, confidence scores, risk codes, and repair actions.
- Persist full decision logs with provenance for each recommendation.
- Add safe auto-repair actions for common issues such as missing adapter mappings, target incompatibility, oversized model choice, missing validation split, or weak evaluation coverage.
- Add rollback support so users can revert autopilot-applied changes.

API work:
- Add endpoints for plan, run, repair preview, repair apply, rollback, and decision-log retrieval.
- Return reason codes, confidence, what changed, and why each repair is considered safe.

Frontend work:
- Add an Autopilot Planner page with dry-run vs run, side-by-side before/after config diff, risk cards, and one-click fixes.
- Add a persistent decision log drawer with filterable statuses.

CLI work:
- Add `brewslm autopilot plan`, `brewslm autopilot run`, `brewslm autopilot repair`, and `brewslm autopilot rollback`.
- Add machine-readable output for CI/agent workflows.

Docs work:
- Add an Autopilot v3 decision log guide with examples of safe repairs and strict-mode stops.

Acceptance criteria:
- Every plan decision is explainable and inspectable.
- Every repair action is previewable before apply.
- Strict mode never hides risky fallbacks.
- Rollback can restore the prior project state for autopilot-applied changes.

Required tests:
- Backend unit tests for planning, scoring, repair selection, and rollback.
- API tests for plan/run/repair/rollback endpoints.
- Frontend tests for dry-run UI, repair preview UI, and decision log UI.
- CLI tests for plan/run/repair/rollback commands.
- End-to-end test covering an initial blocked project that becomes trainable through autopilot repair.
```

#### Status Notes — 22 Apr 2026  ⚠️ PARTIAL

**Done**
- Capability-driven planner: ✅ [backend/app/services/newbie_autopilot_service.py](backend/app/services/newbie_autopilot_service.py) + capability-contract inputs from [backend/app/services/capability_contract_service.py](backend/app/services/capability_contract_service.py).
- Plan / run / orchestrate endpoints: ✅ `/autopilot/plan-v2`, `/autopilot/one-click-run`, `/autopilot/v2/orchestrate`, `/autopilot/v2/orchestrate/run`, `/autopilot/estimate`, `/autopilot/intent-resolve` in [backend/app/api/training.py](backend/app/api/training.py:2176).
- Decision log **payload**: ✅ emitted in every plan/run response via `_append_autopilot_decision(...)` ([training.py:1059](backend/app/api/training.py#L1059)).
- Safe auto-repair for common blockers: ✅ repair logic in `newbie_autopilot_service` handles missing adapter, target incompat, oversized model, etc.
- Strict mode blocker: ✅ `StrictExecutionError` in `backend/app/exceptions.py`; test coverage in [test_sprint1.py](backend/tests/test_sprint1.py).
- Pipeline-run rollback: ✅ `POST /pipeline/rollback` ([backend/app/api/pipeline.py:1595](backend/app/api/pipeline.py#L1595)) — but this rolls back a pipeline run, not a project-state snapshot from autopilot.

**Missing**
- ❌ Persistent decision-log table/model. Decision logs are returned inline with responses; there is no `DecisionLog` entity, no `/autopilot/decisions` query endpoint, and no historical retrieval.
- ❌ Autopilot-state rollback: no snapshot-before-change, no `POST /autopilot/rollback/{decision_id}`, no project-state restore.
- ❌ Repair preview endpoint (diff before apply) and explicit repair-apply endpoint are not separated from the orchestrate flow.
- ❌ `autopilot` CLI subparser entirely absent from [backend/scripts/brewslm.py](backend/scripts/brewslm.py) — no `autopilot plan / run / repair / rollback` commands.
- ❌ Autopilot Planner page: no dedicated frontend page with before/after diff viewer and risk cards. Autopilot is invoked from inside [frontend/src/pages/ProjectTrainingConfigPage.tsx](frontend/src/pages/ProjectTrainingConfigPage.tsx) only.
- ❌ Persistent decision-log drawer UI.
- ⚠️ Docs page [slm-docs/docs/workflows/newbie-autopilot.md](slm-docs/docs/workflows/newbie-autopilot.md) exists but v3-specific content (decision log, rollback, strict mode) is not covered.

**To close Story 4**
1. Add `autopilot_decisions` and `autopilot_snapshots` tables with migration.
2. Add `/autopilot/decisions`, `/autopilot/repair-preview`, `/autopilot/repair-apply`, `/autopilot/rollback/{decision_id}`.
3. Add `autopilot` subparser in `brewslm.py` with plan / run / repair / rollback / decisions (all `--json`).
4. Add `ProjectAutopilotPage.tsx` with dry-run-vs-run diff, risk cards, and persistent decision-log drawer component.
5. Expand the docs page to cover v3 flow, strict mode, and rollback examples.

### Story 5: Evaluation Pack Generator and Gold Set Workbench ⚠️ PARTIAL
User story: As a newbie ML engineer, I want evaluation to be generated from my task and output contract so that I can measure quality before deployment instead of relying on intuition.

Priority: P0
Dependencies: Stories 1, 3, and 4

Prompt for coding agent:

```text
Implement a production-grade Evaluation Pack Generator and Gold Set Workbench.

Backend work:
- Add services to generate starter evaluation packs from the domain blueprint, output contract, and dataset profile.
- Support task-aware metric selection for classification, extraction, QA, summarization, chat, and pairwise preference use cases.
- Add gold-set sampling, annotation task creation, rubric versioning, and failure taxonomy clustering.
- Add remediation suggestion generation tied to evaluation failures.

API work:
- Add endpoints to scaffold an evaluation pack, edit it, version it, create annotation batches, submit labels, compare experiments against the same pack, and fetch remediation suggestions.

Frontend work:
- Add an Evaluation Workbench page with generated metric/rubric suggestions, a gold-set builder, reviewer queue, score breakdowns, failure cluster explorer, and remediation recommendation panel.

CLI work:
- Add `brewslm eval scaffold`, `brewslm eval label`, `brewslm eval run`, `brewslm eval compare`, and `brewslm eval remediate`.

Docs work:
- Add beginner examples showing how evaluation differs for extraction, classification, and QA tasks.

Acceptance criteria:
- A new project can generate a usable evaluation starter pack before the first training run.
- Evaluation packs are versioned and reusable.
- The platform can explain failed gates in beginner-friendly language and suggest concrete next steps.

Required tests:
- Backend unit tests for metric/rubric generation, annotation batching, and remediation logic.
- API tests for scaffold/edit/version/run/compare flows.
- Frontend tests for workbench editing, gold-set queue, and failure-cluster UI.
- CLI tests for scaffold/run/compare/remediate commands.
- End-to-end test that creates an eval pack, runs evaluation, and retrieves remediation suggestions.
```

#### Status Notes — 22 Apr 2026  ⚠️ PARTIAL

**Done**
- Task-aware evaluation packs: ✅ catalog + `/evaluation/packs` endpoint; pack preference per project.
- LLM-judge runs: ✅ `/evaluation/llm-judge`, safety scorecard, gates per experiment.
- Remediation plans: ✅ `evaluation_remediation_service.py` + `/evaluation/remediation-plans/generate` and list/get endpoints.
- Gold set basic I/O: ✅ [backend/app/api/gold.py](backend/app/api/gold.py) — `/gold/add`, `/gold/import`, `/gold/entries`, `/gold/lock`; [backend/app/services/gold_service.py](backend/app/services/gold_service.py) manages gold QA-pair dataset per project.
- Frontend Eval panel: ✅ [frontend/src/components/evaluation/EvalPanel.test.tsx](frontend/src/components/evaluation/EvalPanel.test.tsx) covers run / scorecard / remediation surfaces.
- Docs: ✅ [slm-docs/docs/workflows/evaluation-and-remediation.md](slm-docs/docs/workflows/evaluation-and-remediation.md).

**Missing**
- ❌ Auto-generate pack from (blueprint + dataset profile + output contract). Packs today are selected from a static catalog; there is no `packs/generate` endpoint that derives a pack from the project's blueprint.
- ❌ Pack versioning / edit endpoints.
- ❌ Annotation task creation, reviewer queue, rubric versioning — gold set is append-only pairs, not a workbench.
- ❌ Failure taxonomy clustering — remediation plans exist but there is no cluster explorer over recent eval failures.
- ❌ `eval` CLI subparser: no `eval scaffold / label / run / compare / remediate` in [brewslm.py](backend/scripts/brewslm.py).
- ❌ Evaluation Workbench page with gold-set builder, reviewer queue, failure-cluster explorer.

**To close Story 5**
1. Add `pack_generation_service` deriving a pack from the blueprint + dataset profile.
2. Add versioning + edit endpoints for packs and for gold sets.
3. Add `annotation_task`, `annotation_assignment`, and `failure_cluster` models + migration.
4. Add CLI subparser `eval` with scaffold / label / run / compare / remediate (`--json`).
5. Build an Evaluation Workbench page (gold-set builder, reviewer queue, cluster explorer).

### Story 6: Training Runtime Planner and Reproducibility Manifests ⚠️ PARTIAL
User story: As a newbie ML engineer, I want the system to choose a safe training strategy for my model and hardware and give me a reproducible manifest so that I can rerun or debug the experiment later.

Priority: P0
Dependencies: Stories 2, 3, 4, and 5

Prompt for coding agent:

```text
Implement a production-grade Training Runtime Planner and Reproducibility Manifest system.

Backend work:
- Extend training planning to choose among SFT, LoRA, QLoRA, seq2seq fine-tuning, classification fine-tuning, and alignment modes based on project capabilities and hardware constraints.
- Capture an immutable reproducibility manifest for every run, including dataset snapshot ids, adapter version, domain blueprint revision, model registry revision, training recipe, tokenization settings, runtime id, environment metadata, dependency versions, random seed, and export lineage.
- Add resume/retry/clone-from-manifest flows.
- Add stronger hardware-aware preflight and cost/time estimation with provenance labels.

API work:
- Add endpoints to plan training, fetch a reproducibility manifest, rerun from manifest, clone an experiment, and compare planned vs actual runtime outcomes.

Frontend work:
- Add a Training Planner panel that explains strategy choice, memory tradeoffs, projected runtime/cost, and reproducibility metadata.
- Add "Rerun this exact experiment" and "Clone with safe changes" actions.

CLI work:
- Add `brewslm train plan`, `brewslm train run`, `brewslm train rerun`, `brewslm train clone`, and `brewslm repro manifest`.

Docs work:
- Add a reproducibility guide and examples of rerun/clone workflows.

Acceptance criteria:
- Every training run produces a manifest that can be used to rerun the same experiment.
- The planner explains why it chose a given training mode.
- The platform can compare estimated vs actual time/cost/resource consumption.

Required tests:
- Backend unit tests for strategy selection, manifest generation, rerun, and clone logic.
- API tests for train plan/run/rerun/clone/manifest flows.
- Frontend tests for planner UI and manifest actions.
- CLI tests for plan/run/rerun/manifest commands.
- Integration tests covering CPU-only fallback, GPU-compatible path, and at least one PEFT flow.
```

#### Status Notes — 22 Apr 2026  ⚠️ PARTIAL

**Done**
- Strategy selection (SFT / LoRA / QLoRA / seq2seq / classifier / alignment): ✅ [backend/app/services/training_recipe_service.py](backend/app/services/training_recipe_service.py) with `/training/recipes` and `/training/recipes/resolve`.
- Hardware-aware preflight: ✅ [backend/app/services/training_preflight_service.py](backend/app/services/training_preflight_service.py).
- Experiment config snapshot captured on run: ✅ `experiment_config_snapshot` in [training_service.py:431](backend/app/services/training_service.py#L431).
- Vibe-check / telemetry snapshots captured: ✅ `capture_vibe_check_snapshot()` and telemetry wired into experiment metadata.
- Cost / time estimation: ⚠️ partial via `autopilot/estimate` endpoint and model-selection sweep; not unified as measured-vs-estimated per-run report.
- CLI `train` + `preflight`: ✅ in [brewslm.py](backend/scripts/brewslm.py:1568).
- Docs: ✅ [slm-docs/docs/reliability/measured-vs-estimated.md](slm-docs/docs/reliability/measured-vs-estimated.md).

**Missing**
- ❌ Immutable reproducibility manifest as a first-class object. No single record capturing { dataset snapshot ids, adapter version, blueprint revision, model registry revision, recipe, tokenizer config, runtime id, env digest, pip freeze, seed, git sha, artifact lineage }.
- ❌ `/training/runs/{id}/manifest`, `/training/runs/{id}/rerun-from-manifest`, `/training/runs/{id}/clone` endpoints.
- ❌ CLI `train rerun`, `train clone`, `repro manifest`.
- ❌ "Rerun this exact experiment" and "Clone with safe changes" UI actions.
- ❌ Planned-vs-actual compare view per run.

**To close Story 6**
1. Add `training_manifests` table and backfill serializer covering every field listed above.
2. Add the three endpoints and wire CLI + UI actions.
3. Add measured-vs-estimated per-run report with provenance badges.

## Sprint 3: Reproducibility and Extensibility

### Story 7: Pipeline-as-Code and Full CLI Parity ❌ NOT STARTED
User story: As a newbie ML engineer working with an agent or CI system, I want the full project workflow represented as a declarative manifest and runnable from CLI so that my work is reproducible and automation-friendly.

Priority: P1
Dependencies: Stories 1 through 6

Prompt for coding agent:

```text
Implement Pipeline-as-Code with full CLI parity for the beginner workflow.

Create a declarative project manifest format, for example `brewslm.yaml`, that can describe:
- domain blueprint
- data sources
- adapter definitions
- model choice
- training plan
- evaluation pack
- export/deployment targets
- workflow execution options

Backend work:
- Add manifest validation, import, export, diff, and apply services.
- Ensure manifests are versioned and linked to project history.
- Add compatibility checks that validate a manifest without executing it.

API work:
- Add endpoints to export a project manifest, import a manifest, validate a manifest, diff manifests, and apply selected sections.

Frontend work:
- Add import/export manifest actions in the project workspace.
- Add a manifest diff viewer and validation summary.

CLI work:
- Add `brewslm manifest export`, `brewslm manifest validate`, `brewslm manifest apply`, `brewslm manifest diff`, and `brewslm pipeline run`.
- Ensure every major UI action has an equivalent CLI path.

Docs work:
- Add a manifest reference and CI examples.

Acceptance criteria:
- A complete project can be exported, validated, imported, and rerun from a manifest.
- CLI can execute the full beginner flow without depending on hidden UI-only state.
- Manifest diffs are clear enough to review in code review.

Required tests:
- Backend unit tests for manifest serialization, validation, and diffing.
- API tests for import/export/apply flows.
- Frontend tests for manifest import/export UI.
- CLI tests for export/validate/apply/diff/pipeline-run.
- End-to-end test that exports a project, imports it into a fresh project, and reruns the pipeline successfully.
```

#### Status Notes — 22 Apr 2026  ❌ NOT STARTED

**Adjacent work that exists (not a substitute)**
- Pipeline recipes with compile / run / cancel / retry / resume: ✅ [backend/app/api/pipeline.py:827–1135](backend/app/api/pipeline.py). These are server-side recipe definitions, not declarative project manifests.
- Export-manifest serialization for export runs: ✅ [test_phase4_export_manifest.py](backend/tests/test_phase4_export_manifest.py). Scoped to exports, not to whole-project state.

**Missing (all acceptance criteria unmet)**
- ❌ `brewslm.yaml` (or `.json`) schema definition covering blueprint + data sources + adapters + model + training plan + eval + export/deployment + workflow options.
- ❌ Endpoints: manifest export / validate / diff / apply (with plan-only).
- ❌ CLI: `manifest export / validate / apply / diff / pipeline run` — no `manifest` subparser in [brewslm.py](backend/scripts/brewslm.py).
- ❌ Frontend: import / export manifest UI; diff viewer.
- ❌ Docs: manifest reference, CI examples.
- ❌ Round-trip end-to-end test.

**To close Story 7**
1. Define and version the `brewslm.yaml` schema (JSON Schema + Pydantic).
2. Implement serializer, deserializer, validator, diff, apply services.
3. Wire endpoints, CLI subparser, and frontend import/export UI.
4. Publish manifest reference and CI workflow example.

### Story 8: Extension Scaffolder and Plugin Contract Validator ❌ NOT STARTED
User story: As a user who outgrows the beginner path, I want generated scaffolds for adapters, runtimes, domain packs, and evaluation packs so that I can extend the system for unusual domains, base models, and data structures without starting from scratch.

Priority: P1
Dependencies: Stories 2, 3, and 7

Prompt for coding agent:

```text
Implement a production-grade Extension Scaffolder and Plugin Contract Validator.

Backend work:
- Add a contract validator for data adapter plugins, training runtime plugins, domain packs, and evaluation packs.
- Add packaging metadata, compatibility validation, and safe reload checks.
- Add artifact lineage so generated scaffolds are linked back to the originating project or manifest.

API work:
- Add endpoints to scaffold extensions, validate extension contracts, list installed extensions, inspect compatibility warnings, and reload validated extensions.

Frontend work:
- Add an Extension Studio page for admins/engineers that can generate scaffolds from templates or from an existing project.
- Show compatibility warnings, required dependencies, and versioning guidance.

CLI work:
- Add `brewslm scaffold adapter`, `brewslm scaffold runtime`, `brewslm scaffold domain-pack`, `brewslm scaffold eval-pack`, and `brewslm extensions validate`.

Docs work:
- Replace copy-manual-template guidance with scaffold-first guidance.
- Add extension authoring tutorials and sample generated outputs.

Acceptance criteria:
- A user can generate a scaffold for a custom adapter/runtime/domain/eval extension from CLI or UI.
- Generated extensions include tests, docs stubs, version metadata, and validation hooks.
- The platform can validate an extension before load and explain failures clearly.

Required tests:
- Backend unit tests for contract validation and scaffold generation.
- API tests for scaffold/validate/reload flows.
- Frontend tests for Extension Studio generate/validate views.
- CLI tests for scaffold and validate commands.
- Integration test that generates a sample adapter scaffold, validates it, and loads it successfully.
```

#### Status Notes — 22 Apr 2026  ❌ NOT STARTED

**Adjacent work that exists**
- Adapter-only plugin scaffold export: ✅ `export_adapter_scaffold()` in [backend/app/services/adapter_studio_service.py:1152](backend/app/services/adapter_studio_service.py#L1152); emitted as Python source via `_plugin_scaffold_code()`. CLI exposes it as `adapter export`.
- Plugin system exists for data adapters, training runtimes, and domain hooks under [backend/app/plugins/](backend/app/plugins/) — templates in place but no generator UX.
- Alignment scaffolding service (internal): `backend/app/services/alignment_service.py` (preference alignment; not a plugin scaffold generator).
- Starter-pack reload + plugin status endpoints on [backend/app/api/starter_packs.py](backend/app/api/starter_packs.py) and [backend/app/api/training.py](backend/app/api/training.py:2154) — close to a registry but no validator.

**Missing**
- ❌ Contract validator service covering data adapter / training runtime / domain pack / evaluation pack plugins.
- ❌ Scaffold generator for runtime, domain pack, evaluation pack (only adapter is scaffolded today).
- ❌ Endpoints: `/extensions/scaffold`, `/extensions/validate`, `/extensions/reload`, `/extensions/list`.
- ❌ CLI `scaffold adapter / runtime / domain-pack / eval-pack` and `extensions validate`.
- ❌ Extension Studio page.
- ❌ Scaffold-first authoring docs.

**To close Story 8**
1. Formalise the plugin contract interfaces (ABC / Protocol) per kind with schema-level tests.
2. Build a generic `scaffold_service` templating layer; extend `adapter` generator to the other three kinds.
3. Add `/extensions/*` endpoints and the `scaffold` CLI subparser.
4. Build the Extension Studio page and replace copy-template docs with scaffold-first guides.

### Story 9: Deployment Assistant, Target Smoke Tests, and Rollback ⚠️ PARTIAL
User story: As a newbie ML engineer, I want the platform to validate deployment readiness and run target-specific smoke tests so that I can trust an exported artifact before I share it.

Priority: P1
Dependencies: Stories 2, 5, 6, and 7

Prompt for coding agent:

```text
Implement a production-grade Deployment Assistant with target-specific smoke tests and rollback support.

Backend work:
- Extend export/deployment services to create deployment plans tied to target capabilities, artifact profiles, benchmark results, and smoke-test suites.
- Add deployment run records, smoke-test outputs, and rollback metadata.
- Add deployability scoring with measured vs estimated provenance.

API work:
- Add endpoints to plan deployment, run smoke tests, fetch readiness reports, promote or reject a deployment candidate, and rollback to a previous deployment.

Frontend work:
- Add a Deployment Assistant page with target recommendations, benchmark summaries, smoke-test reports, and rollback actions.
- Show beginner-friendly warnings when a target is technically possible but risky.

CLI work:
- Add `brewslm deploy plan`, `brewslm deploy smoke-test`, `brewslm deploy promote`, and `brewslm deploy rollback`.

Docs work:
- Add target-specific deployment guides for local runner, Ollama, vLLM, and lightweight edge export examples.

Acceptance criteria:
- Every exported deployment candidate can be checked against a target-specific smoke-test suite before promotion.
- Rollback is recorded and auditable.
- The system distinguishes measured readiness from estimated readiness.

Required tests:
- Backend unit tests for deployment planning, smoke-test orchestration, and rollback.
- API tests for plan/smoke/promote/rollback flows.
- Frontend tests for Deployment Assistant readiness and smoke-test UI.
- CLI tests for deploy plan/smoke/promote/rollback.
- End-to-end test that exports a model, runs smoke tests, promotes it, and rolls back.
```

#### Status Notes — 22 Apr 2026  ⚠️ PARTIAL

**Done**
- Deployment target catalog with per-target `smoke_supported` flag: ✅ [backend/app/services/deployment_target_service.py](backend/app/services/deployment_target_service.py).
- Smoke tests for vLLM / TGI / Ollama / HF Endpoint / local runner: ✅ `_run_smoke_command()` and target-specific checks in `deployment_target_service.py` (lines 441–520+); orchestrated via `run_smoke_tests=True` parameter.
- Deployment validation endpoint: ✅ `POST /export/{id}/deployment-validate` in [backend/app/api/export.py:162](backend/app/api/export.py#L162).
- Deploy-as-API planning + execution: ✅ `POST /export/{id}/deploy-as-api` and `/execute`, plus `serve-runs` start / status / stop endpoints (`/export/serve-runs/{run_id}` family).
- Target catalog + compatibility + estimate endpoints: ✅ [backend/app/api/targets.py](backend/app/api/targets.py).
- Deployment reports + serve-plan snippets: ✅ in export API.
- Docs: ✅ [slm-docs/docs/workflows/export-and-deployment.md](slm-docs/docs/workflows/export-and-deployment.md), [reliability/measured-vs-estimated.md](slm-docs/docs/reliability/measured-vs-estimated.md).

**Missing**
- ❌ Deployment rollback: no endpoint, no model to track "previous served version".
- ❌ Promote / reject flow for a deployment candidate (today serve-runs just start/stop).
- ❌ Deployability score synthesising measured smoke-test outcomes vs estimated compatibility with provenance labels.
- ❌ Deployment Assistant frontend page bringing readiness, smoke, promote, rollback together; today these are surfaces on [ExportPanel.tsx](frontend/src/components/export/ExportPanel.test.tsx).
- ❌ `deploy` CLI subparser — no `deploy plan / smoke-test / promote / rollback` commands.
- ❌ Post-deploy observability (latency / errors / drift) — out of scope for Story 9 as written but worth noting.

**To close Story 9**
1. Add `deployment_versions` and `deployment_rollbacks` models.
2. Add `/deployments/{id}/promote`, `/reject`, `/rollback`, `/readiness-report`.
3. Add `deploy` CLI subparser with plan / smoke-test / promote / rollback (`--json`).
4. Build a Deployment Assistant page with the full readiness → promote → rollback flow.

## Sprint 4: Production Hardening and Learning Assets

### Story 10: Observability, Failure Analysis, and Support Bundles ❌ NOT STARTED
User story: As a newbie ML engineer, I want one place to understand what failed, why it failed, and what to send for help so that I can debug issues without digging through raw logs.

Priority: P1
Dependencies: Stories 4, 6, 7, and 9

Prompt for coding agent:

```text
Implement a production-grade observability layer with failure analysis and support bundle generation.

Backend work:
- Correlate project, workflow, job, experiment, evaluation, export, and deployment events into a unified run timeline.
- Add structured error taxonomy, failure clustering, and support bundle packaging.
- Include decision logs, environment metadata, config snapshots, recent logs, and relevant artifacts in support bundles with secret redaction.

API work:
- Add endpoints to fetch timeline views, failure clusters, recent warnings, and downloadable support bundles.

Frontend work:
- Add a Run Timeline and Failure Analysis dashboard with filters by stage, severity, reason code, and time.
- Add one-click support bundle generation and download.

CLI work:
- Add `brewslm doctor --deep`, `brewslm logs tail`, and `brewslm support-bundle`.

Docs work:
- Add troubleshooting guides tied to reason codes and support bundle contents.

Acceptance criteria:
- A user can identify the failing stage, reason code, and suggested fix from one page or one CLI command.
- Support bundles are redacted, reproducible, and useful for debugging.
- Failure clustering groups similar issues instead of surfacing only raw stack traces.

Required tests:
- Backend unit tests for failure taxonomy, support bundle generation, and redaction.
- API tests for timeline/failure/support-bundle endpoints.
- Frontend tests for timeline filters, failure clustering UI, and support bundle actions.
- CLI tests for doctor/logs/support-bundle.
- Integration test that triggers a controlled failure and verifies the support bundle contents.
```

#### Status Notes — 22 Apr 2026  ❌ NOT STARTED

**Adjacent work that exists**
- Audit log list endpoint: ✅ `GET /audit/logs` in [backend/app/api/audit.py](backend/app/api/audit.py) — but this is a flat list, not a correlated timeline.
- Per-service telemetry: ✅ `/training/observability/telemetry`, `/training/model-selection/telemetry`, dataset telemetry service. These emit stage-local metrics; they are not joined into one timeline.
- CLI `doctor`: ✅ exists in [brewslm.py](backend/scripts/brewslm.py:1646), but it is a single-shot readiness check — no `--deep` timeline view.
- Workflow runner state tracking: ✅ [backend/app/services/workflow_runner_service.py](backend/app/services/workflow_runner_service.py).

**Missing (all Story 10 acceptance criteria unmet)**
- ❌ Canonical `RunEvent` schema and correlated timeline across project / workflow / job / experiment / eval / export / deployment. (Verified absent: `grep -rn "support_bundle\|failure_cluster\|failure_taxonomy\|run_timeline" backend/app/` returns zero matches.)
- ❌ Structured error taxonomy / reason-code enum applied platform-wide.
- ❌ Failure clustering service.
- ❌ Support bundle service (packaging + secret redaction). Verified absent.
- ❌ Endpoints: `/projects/{id}/timeline`, `/projects/{id}/failure-clusters`, `/projects/{id}/support-bundle`.
- ❌ CLI: `doctor --deep`, `logs tail`, `support-bundle`.
- ❌ Run Timeline + Failure Analysis frontend dashboards.

**To close Story 10**
1. Introduce the `RunEvent` schema and emit from every stage service.
2. Build `timeline_service`, `failure_taxonomy_service`, `support_bundle_service` (with redaction).
3. Add endpoints, CLI commands, and frontend Run Timeline / Failure Analysis pages.

### Story 11: Quality Bar, Sample Projects, and Guided Learning Mode ❌ NOT STARTED
User story: As a newbie ML engineer, I want built-in examples, guided learning paths, and strong end-to-end test coverage so that I can learn the system safely and trust that the features work together.

Priority: P1
Dependencies: Stories 1 through 10

Prompt for coding agent:

```text
Implement the final production hardening layer: sample projects, guided learning mode, and a stronger QA matrix.

Backend work:
- Add sample project fixtures and reproducible demo assets for at least support QA, structured extraction, summarization, classification, and preference tuning.
- Add seed data loaders and fixture lifecycle utilities for tests and local demos.

API work:
- Add endpoints to create a sample project, reset a sample workspace, and list installed tutorial/sample assets.

Frontend work:
- Add a Guided Learning Mode with contextual tips, walkthrough checkpoints, and links into docs for first-time users.
- Add a sample project gallery from the dashboard or project wizard.

CLI work:
- Add `brewslm demo create`, `brewslm demo reset`, and `brewslm tutorial list`.

Docs work:
- Add step-by-step tutorials for each sample project.
- Standardize product naming and terminology across README, docs, UI labels, and CLI help text.

Quality work:
- Add an end-to-end browser test suite for the main beginner paths.
- Add CI gates for backend, frontend, CLI, end-to-end, and fixture validation.
- Add regression tests that cover the complete "brief -> data -> train -> eval -> export -> deploy" path.

Acceptance criteria:
- A first-time user can launch a sample project and follow a guided path through the platform.
- CI blocks regressions in the main beginner workflow.
- Terminology is consistent across the product and docs.

Required tests:
- Backend tests for sample fixture creation/reset.
- API tests for sample project endpoints.
- Frontend tests for guided learning flows and sample gallery.
- CLI tests for demo and tutorial commands.
- End-to-end tests for at least two sample projects from creation to export.
```

#### Status Notes — 22 Apr 2026  ❌ NOT STARTED

**Adjacent work that exists**
- Starter packs registry + catalog endpoints: ✅ [backend/app/api/starter_packs.py](backend/app/api/starter_packs.py) + [starter_pack_service.py](backend/app/services/starter_pack_service.py) with reload.
- Six sample projects on disk: ✅ [data/projects/](data/projects/) directories 1–6.
- Quickstart template aliases (`general`, `support`, `legal`): ✅ in `brewslm.py`.
- Per-task docs: ✅ quickstart + first-project in [slm-docs/docs/getting-started/](slm-docs/docs/getting-started/).
- Glossary: ✅ [slm-docs/docs/reference/glossary.md](slm-docs/docs/reference/glossary.md).
- CI configuration: ✅ [.github/workflows/](.github/workflows/).

**Missing (all Story 11 acceptance criteria unmet)**
- ❌ Sample project lifecycle endpoints (`/samples/create`, `/samples/{id}/reset`, `/samples/list`).
- ❌ Guided Learning Mode UI: no contextual tips, no walkthrough checkpoints, no sample gallery on the dashboard.
- ❌ CLI `demo create / reset`, `tutorial list`.
- ❌ Per-sample tutorials in docs (support-QA / extraction / summarization / classification / preference) showing end-to-end flow.
- ❌ End-to-end browser suite. Verified absent: no `playwright.config*` or `cypress.config*` anywhere in the repo. Only unit/integration tests (10 frontend `.test.tsx` files).
- ❌ Regression matrix gating the "brief → data → train → eval → export → deploy" path.
- ❌ Terminology audit: README / docs / UI / CLI strings have not been reconciled.

**To close Story 11**
1. Add sample project service with create/reset/list endpoints; seed fixtures for the five task families.
2. Add `demo` and `tutorial` CLI subparsers.
3. Build Guided Learning Mode (checkpoint sidebar + contextual tips) and a Sample Gallery on the project wizard.
4. Adopt Playwright and ship at least two golden-path E2E tests.
5. Run a terminology pass across README, docs, UI labels, and CLI help, and add a lint script to keep them in sync.

## Recommended Delivery Order
If only a subset can be built first, implement in this order:

1. Story 1
2. Story 3
3. Story 2
4. Story 4
5. Story 5
6. Story 6
7. Story 7
8. Story 9
9. Story 10
10. Story 8
11. Story 11

## Practical Definition of Success
The platform is ready for the target audience when a beginner can:

1. Start with a plain-English brief instead of platform-specific jargon.
2. Import messy real-world data without custom Python for the common path.
3. Get trustworthy model and training recommendations with explanations.
4. Produce a measurable evaluation plan before deployment.
5. Reproduce the full workflow from CLI or manifest.
6. Debug failures using decision logs and support bundles.
7. Extend the platform with generated scaffolds when they hit an uncommon case.

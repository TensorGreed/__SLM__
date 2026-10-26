# Changelog

### Domain packs as data + your own checks (October 26, 2026)

- A **domain pack** can now carry content as well as its overlay and hooks: a zip with `pack.yaml` plus a label taxonomy, probe packs (your own checks), rubrics, schemas, a tool catalog, seed documents (licence + usage each), answer-key dev / test rows and a project template (recipe, base model, target and plan profile, training defaults, retrieval settings). Domain content ships as data, never as code; no pack ships in core.
- **Import** (`POST /api/domain-packs/import/preview`, `/import`): validated before anything is written (missing or unsafe paths, zip-slip, size limits, corrupt zips, duplicate ids, bad regexes, unknown recipe, invalid taxonomy / schema / contract; `INVALID_DOMAIN_PACK` with `metadata.errors`, grouped by code in the UI). A re-import shows a per-section diff; identical content is `unchanged`, otherwise a version that isn't higher is patch-bumped. Each version's files are kept at `data/domain_packs/<pack_id>/<version>/`; system packs can't be imported over. No migration.
- **Create a project from a pack** (`POST /api/domain-packs/{pack_id}/instantiate`): copies every artifact, all-or-nothing, stamped with `pack_id@version` (documents get a `domain_pack` provenance source). **Export a project as a pack** (`GET /api/projects/{id}/domain-pack/export`); the zip re-validates and round-trips. Preview, import and instantiate need the admin or engineer role. Training defaults prefill the Train form; they are not applied to API or autopilot starts.
- **Your own checks** (user probe packs, `GET` / `PUT` / `DELETE /api/projects/{id}/probe-pack/user`, `/validate`, `/restore`): any existing probe property plus `must_match` / `must_not_match` (regex, 0.5 s timeout per search) and `json_pointer_not_in` (JSON pointer with `*` wildcards, every JSON value in the output is checked), with `severity` (weight) and `critical`. They run beside the built-in checks in the same evaluation; `probe_pass_rate` stays built-in only, your checks report `user_probe_pass_rate` and `by_source`. **Any critical check that fails blocks the gate** (`probe_critical_failures`, required, always on even when the probe gate is off; it also fails when the checks didn't run).
- UI: Settings > Domain packs gains import with preview and diff, the pack list with "Create project from this pack" and "Export this project as a pack"; the checks panel lists your checks (source, severity, critical), critical failures and per-source pass rates, with a paste-YAML editor (merge by pack id, restore an earlier version); the scorecard groups the critical gate under "Independent ruler".

### Code-findings recipe + finding-set scoring (October 26, 2026)

- Added the **Code findings** recipe (`code-findings`, structured extraction, scoring mode `finding_set`): a chunk of code in, `{"findings": [{label, start_line, end_line, evidence, attrs}]}` out, with absolute line numbers; suggested base `Qwen/Qwen2.5-Coder-7B-Instruct`. One shared prompt builder serves training rows, held-out eval and probes (byte-identical, chat template, every line numbered, taxonomy labels listed); rows are detected from the G3 importer shape. The structured adapter only uses it when the project's scoring mode is `finding_set`; a findings-less row is rejected as `findings_json_missing`. The prepared manifest stamps `finding_match`, `labels`, `label_parents` and `prompt_labels`.
- **`finding_set` metrics**: strict label + line-range IoU matching (default 0.5, one-to-one), micro and macro precision / recall / F1, a per-label table, `clean_chunk_fp_rate`, `malformed_rate`, `json_validity_rate`, attribute accuracy and an optional hierarchical score (parent-label credit). Scaffolded pass/fail rules use `lte` for the `max_` rates. Empty-on-empty scores 1.0 for macro as for micro.
- Three **synthetic playbooks** (`seeded_defect`, `hard_negatives`, `cluster_targeted`) write findings rows and check the teacher's claim against the code it returns (changed-line anchoring, real evidence, lines inside the chunk). Seeds come from the answer key's dev set and the project's code documents only (never the test set, never evaluation-only / retrieval-only sources) and rotate across batches; a context dataset cannot be an answer key (`context_dataset_not_allowed`). Every playbook run now reports `rejected_count`, `reject_summary` and `rejected_examples`, kept across resumed chunks. `params.label_id` is a free label when the project has no taxonomy.
- **Structured-output hint**: applying the recipe stores a JSON Schema (label enum from the taxonomy) at `GET /api/projects/{id}/guided-json`, refreshed when the taxonomy changes and carried into the export manifest.
- A recipe-keyed **probe pack** (empty / comment-only code reports nothing; labels stay put when a reassuring or instruction-like comment is added or variables are renamed), run with the full JSON generation budget.
- UI: finding-set metrics panel and row view in Evaluate, a seeded-defect form (label, attributes, data-carrying context with a dataset and column picker) and a reject summary on the synthetic playbook panel, a guided-JSON card on Ship > Package, rejected counts in the notification bell. New `GET /api/projects/{id}/datasets` and `GET /api/projects/{id}/datasets/{dataset_id}/columns`.
- Training preflight no longer reads a code row's `path` as a media file; the synthetic dedup keys findings rows on code plus findings.

### Label taxonomy + code-dataset mappers (October 26, 2026)

- **Label taxonomy** per project (`runtime_config["label_taxonomy"]`, no migration): entries with id, title, parent, aliases, attrs and a crosswalk to reference ids in other namespaces; YAML / JSON in several layouts, `crosswalk_from` for `NAMESPACE:ref` fields, all-or-nothing validation (`INVALID_TAXONOMY`, per-entry `metadata.errors`), tiered resolution by id / alias / title (exact first; a label that matches several entries is ambiguous and never guessed), YAML aliases refused, a content-hash version and a 20-version history with restore (full snapshots for the newest 5, older entries listed but not restorable). A `crosswalk_from` value without `:` names a whole namespace. New `GET` / `PUT` / `DELETE /api/projects/{id}/label-taxonomy`, `/validate`, `/restore`, `/resolve`.
- **Code-dataset mappers** writing canonical findings rows (`row_kind: "findings"`: code, language, path, absolute finding lines, taxonomy-resolved labels, `taxonomy_version`): `code_label_to_findings` (labelled code), `fix_pair` (before / after pairs to findings, patch or preference rows; finding lines from a line diff) and `contrast_functions` (files with `bad…` / `good…` functions, via the code chunker). `fix_pair` finding lines are per changed hunk by default; `contrast_functions` anonymises `bad…` / `good…` names and blanks revealing comments by default so a model cannot learn the names. Unknown / ambiguous labels reject as `unknown_label` / `ambiguous_label` and can be bulk-dropped; an odd row is rejected as `parse_error` instead of stopping the import; without a taxonomy labels stay as written with a `no_taxonomy` warning. Dataset-import preview / run results gain `taxonomy: {version, entry_count}`; identical per-row warnings are listed once.
- The dataset-import introspector recognises code columns and proposes the three mappers (and ranks plain text classification below them for code data).

### Command-line teacher backend + hard API budget (October 26, 2026)

- Added the **command (local CLI) synth backend** (`command` / `command:<model>`): runs an allow-listed command-line tool (for example a headless vendor CLI on a subscription) as the teacher. Explicit pin only (never auto-picked, not even as the judge); disabled until `SYNTH_COMMAND_ALLOWED_BINARIES` is set. No shell, minimal environment (API-key variables never forwarded), a temporary working directory per call, killed on timeout / cancel; argv, binary, environment and prompt mode are install-wide, never per project. Requests are packed into JSON-lines batches (`SYNTH_COMMAND_BATCH_SIZE`) with per-item retries and reasons; the documents-to-Q&A flow, teacher answers and relabel use the packed path, and the flow's **Batch** option now accepts it.
- A usage-limit failure **pauses and resumes**: the Job message reads "waiting for usage window - resumes ~HH:MM UTC" (a "paused" chip in the bell), the same call is retried, and the Job result records `paused_seconds`. Synthetic playbooks run **chunked and checkpointed** on this backend (`SYNTH_COMMAND_PLAYBOOK_ROWS_PER_CALL`): a re-run reuses finished chunks, failed chunks are listed in `failed_chunks`.
- **Hard API budget** for metered cloud calls: the per-project monthly budget is now a hard stop, and `LLM_MONTHLY_BUDGET_USD` adds an account-wide cap across all projects (editable in runtime settings). A job pinned to a cloud model starts as `402 llm_budget_exceeded` at the cap; a job that reaches it mid-run fails with "API budget reached (...)". Local models and subscription CLIs are never counted or blocked. `GET /projects/{id}/llm-budget` gains `global` and `hard_stop`.
- Spend: subscription CLI usage is recorded as `subscription:<binary>`, priced $0 and shown separately (`billing` per model row; `subscription_calls` / `subscription_tokens` in the totals).
- Hardening: a `command:<model>` pin may not start with `-`; usage-limit pauses have a 60 s floor and one call gives up after 20 consecutive hits; a bare allow-list name only matches a bare `argv[0]`; cancelling a Job kills the in-flight CLI and sibling chunks; concurrency is one process-wide limit (a project override can only lower it); Message-Batch and unpriced-model spend count toward the hard budget; the playground, probe judge, plan refinement and teacher capture stop at a reached budget; playbook checkpoints are keyed on the prompt and deleted only after the rows are committed.
- New `GET` / `PUT /api/projects/{id}/eval-judge` (the judge model picker, with billing per option); `GET /projects/{id}/synthetic/backends` entries gain `billing` and `detail`.

### Licence & provenance ledger (October 26, 2026)

- Added a **licence & provenance ledger**: one source per upload, website crawl, git repository, Hugging Face / Kaggle / URL import, dataset-import run, predefined split, demo and external benchmark (`GET` / `PATCH /api/projects/{id}/provenance/sources`; migration `20261026_0050`). Each records where the data came from (URL / dataset id / repo, commit or revision), when, and its licence (read from the hub / repository where published; unknown otherwise, with warnings for unknown and non-commercial licences; a licence can be marked verified).
- Added per-source **usage** (`any`, `train`, `eval`, `rag`): `eval` rows are held out of train / validation and go to the test split (duplicates of them - near-identical rows too when dedup is on - are removed from train and validation; not added to a pinned test), `rag` rows are in no split and never seed synthetic data (their documents stay indexed for passages). The prepared manifest gains `provenance` (rows per source per split, exclusions, licence warnings) and training runs record it (`data_provenance`); projects with no eval / rag source prepare exactly as before.
- Exports carry `provenance` (sources, rows per licence, warnings) in `manifest.json` and a "Data sources & licences" section in `RELEASE_NOTES.md`. Uploads accept `usage` and `license_info` (batch too); the website crawl accepts `licence`; `DocumentResponse` gains `data_source_id`, `usage`, `licence`, `licence_flags`, `source_ref`, `retrieved_at`; the `Imported from ...` placeholder licence is no longer written.
- A duplicated project remaps `source_document_id` in the copied cleaned / synthetic / prepared rows, so eval / retrieval-only marks keep working in the copy; rows generated from cleaned chunks are stamped with their documents; answer-key (gold) building is grounded in eval-only documents too; sources left with no documents are removed; more non-commercial wordings are flagged (`none` / `n/a` count as unknown).

### Document sources + code-aware chunking (October 26, 2026)

- Added **Documents from website / git repository**: `POST /api/projects/{id}/document-sources/web-crawl` and `/git-repo` (background Jobs `document_source_web_crawl` / `document_source_git_repo`). The crawler honours robots.txt, stays inside allowed prefixes, rate-limits, caches by ETag / Last-Modified and strips navigation; the git importer pins the commit, detects the licence and skips binaries, large files and vendored directories. The website importer follows RFC 9309 (longest match, `*` / `$`, percent-encoding; an unreachable robots.txt or a 5xx skips the host as `robots_unreachable`, a 4xx allows it) and caps robots.txt / sitemap downloads; navigation stripping never drops `html` / `body` / `main` / `article` / `role="main"` and falls back to tag-only stripping when a class name would empty the page. Git `exclude_globs` are added to the built-in defaults (`default_excludes: false` turns those off; they match at any depth), URLs with embedded credentials and loopback ssh targets are refused, and imports stop promptly when the Job is cancelled. Repo source / config files are never read as training rows or used for task-shape detection. Rejected pages / files are reported by stable reason code; each document records where it came from (`source_url`, `path`, `commit`, licence).
- Added a **code chunker**: source and config files (Java, C#, Python, Go, JS/TS, C/C++, shell, Terraform, YAML, JSON, XML, SQL, INI, properties, Dockerfile) are cut at function / method / class boundaries with exact line ranges and symbols; code is cleaned verbatim (counted, not redacted) and passages are cited as `path:start-end (symbol)`. Source files can now also be uploaded.
- BM25 retrieval splits `camelCase` / `snake_case` identifiers into words (and keeps the whole identifier); all-caps plurals (`IDs`, `APIs`) and `5G` stay whole; indexes are stamped `BM25_TOKENIZER_VERSION = 3`, older ones are rebuilt automatically and `retrieve()` refuses a stale one instead of mis-scoring it.
- Dataset-import sources `web:` and `git:`; `DocumentResponse` now includes `license_info`, `source_url`, `path`, `commit`, `content_kind`, `language`; new setting `DOCUMENT_SOURCES_ALLOW_LOCAL_GIT` (default off).

### Latest Updates (March 5, 2026)

- Added **Model Registry** lifecycle with readiness snapshots, promotion gates, and deploy tracking.
- Added **Project Secrets** APIs with masked listing and encrypted-at-rest values.
- Added **Domain Profiles** with a typed contract and project-level assignment.
- Added **Domain Packs** (pluggable overlays + default profile pointer) with project-level assignment.
- Added **Domain Pack Hooks** for custom normalizers, validators, and evaluators.
- Added **Domain Profile Manager UI** (list/create/edit/assign) in project detail and profile selection during project creation.
- Added **Domain Pack Manager UI** (list/create/edit/assign) in project detail and pack selection during project creation.
- Added **Workflow Graph Preview** (read-only visual stage graph + step contracts) in project detail.
- Added **Workflow Contract Runtime (Phase 2)** APIs: graph validate, dry-run, run active step, and run history.
- Added **Visual Pipeline Editor (Phase 3)** with project-scoped graph contract save/load/reset and compile diagnostics.
- Added **Step Contract SDK v1** runtime fields (`runtime_requirements`) with compile/dry-run/run-step enforcement.
- Added **Typed Artifact Registry (Phase 4)** with versioned artifacts and pipeline run-step artifact publication.
- Added **Workflow Runner v1 (Phase 5)** with persisted DAG runs, node retries, dependency tracking, and backend adapters (`local`, `celery`, `external`).
- Added **Workflow Templates + Run Monitor UI (Phase 6)** on the Workflow page.
- Refactored project UI into a cleaner workspace IA:
  - `Pipeline` page
  - `Training Config` page
  - `Workflow Graph` page
  - `Domain Packs` page
  - `Domain Profiles` page
  - `Domain Contracts` legacy combined page (backward-compatible route)
  - `Pipeline Recipes` page
  - Training run monitoring remains in the `Pipeline -> Training` stage
  - left sidebar navigation with pipeline stages as submenu
- Added runtime transparency in split/training responses (applied profile + resolved defaults).
- Added **Duplicate-as-new-version** flow for domain profiles from the project UI.
- Added dedicated **Resolved Defaults** panels in Dataset Prep and Training tabs.
- Added queued **remote ingestion** jobs with task status/cancel APIs and WebSocket log streaming.
- Added queued **compression** jobs (quantize/merge/benchmark) with task status/cancel APIs and WebSocket logs.
- Added real GGUF/ONNX compression runtime paths in `backend/scripts/quantize.py`.
- Added **experiment comparison** API/UI for side-by-side metrics and loss-history visualization.
- Added auth UX updates for **SSO + local login** flow.
- Added training runtime generalization:
  - trainer backend abstraction (`auto`, `hf_trainer`, `trl_sft`)
  - task/data adapter contract (`causal_lm`, `seq2seq`, `classification`)
  - CUDA OOM auto-retry planner with progressive memory downscaling
- Added **Training Capability Matrix + Preflight**:
  - explicit config preflight endpoint before experiment creation/start
  - model/task/trainer compatibility checks
  - dependency/runtime/data-file preflight checks
  - start-time server guard that blocks incompatible training runs
- Added **Training Runtime Plugin SDK v1**:
  - runtime registry + catalog endpoint (`/training/runtimes`)
  - pluggable runtime selection via `training_runtime_id`
  - built-in runtimes: `builtin.simulate`, `builtin.external_celery`
  - legacy `TRAINING_BACKEND` remains default fallback for backward compatibility
- Added **Training Recipe System v1**:
  - built-in domain-agnostic recipe catalog (safe/balanced SFT, LoRA-fast, classification, seq2seq)
  - recipe resolve endpoint with runtime/profile default merge + optional preflight
  - Training UI recipe starter picker with one-click apply
- Added **Evaluation Packs + Auto Gates v1**:
  - built-in gate packs (`general`, `strict`, `fast-iteration`) + domain-profile-derived dynamic pack
  - project-level evaluation pack preference with fallback to defaults
  - experiment gate evaluation endpoint and pipeline status auto-gate summary
- Added **Pipeline Recipes / Blueprints v2 (Recipe Compiler)**:
  - built-in end-to-end recipes (`recipe.pipeline.sft_default`, `recipe.pipeline.lora_fast`, `recipe.pipeline.eval_gate`, `recipe.pipeline.program.sft_sweep`)
  - resolve/apply APIs that wire domain pack/profile, dataset adapter preset, training recipe, and evaluation pack
  - workflow resolution now supports `workflow.template_id` and executable `workflow.program` contracts
  - Recipe Compiler v2 supports conditional nodes/edges, failure-type retry policies, and loop/sweep node execution plans
  - task-profile-aware recipe recommendation context (`recommended_recipe_id`) for faster defaults on the Recipes page
  - persisted active recipe state + versioned reproducibility manifest artifact (`manifest.pipeline_recipe`)
  - dedicated `Pipeline Recipes` page in project workspace navigation (separate from Pipeline and Workflow pages)
- Added top-right **User Menu** in workspace:
  - profile identity (username/role)
  - logout action
  - runtime **Settings** modal for editable system config fields (persisted overrides)
- Added **Preflight Plan Suggestions** (`safe`, `balanced`, `max_quality`) with one-click config apply in Training UI.
- Added project-persisted training plan preference (`preferred_plan_profile`) so recommended profile choice is remembered per project.
- Added project/domain-pack adapter preset resolution for dataset split defaults (`adapter_id`, `adapter_config`, `field_mapping`).
- Added **Alignment Workbench baseline** for DPO/ORPO:
  - preference dataset import/filter APIs with apply-to-train support
  - judge quality keep-ratio guardrails in preflight
  - training-mode controls + alignment auto-filter settings in Training Config UI
  - native TRL pairwise objective runtime path (`DPOTrainer` / `ORPOTrainer`) in external training script
- Added **Universal Task/Dataset Adapter v3**:
  - typed adapter contracts (`task_profiles`, `preferred_training_tasks`, `output_contract`)
  - expanded built-in adapters: `rag-grounded`, `tool-call-json`, `structured-extraction` (plus existing chat/preference adapters)
  - richer preview diagnostics: `conformance_report`, `auto_fix_suggestions`, `inferred_task_profiles`, `raw_field_frequency`
  - `task_profile` wiring across adapter preview, project adapter preference, split manifests, workflow adapter-preview node, and remote import
- Added **Pipeline Recipe Executable Runs (Phase 18)**:
  - `POST /pipeline/recipes/run` to apply + execute recipe-backed workflow DAG (sync or async)
  - persisted recipe-run lineage records (`recipe_run_id -> workflow_run_id -> manifest snapshot`)
  - `GET /pipeline/recipes/runs` and `GET /pipeline/recipes/runs/{recipe_run_id}` for status/history
  - Recipe page now includes one-click run controls and recipe-run monitor
- Added **Recipe Run Control + Recovery (Phase 19)**:
  - `POST /pipeline/recipes/runs/{recipe_run_id}/cancel`
  - `POST /pipeline/recipes/runs/{recipe_run_id}/retry`
  - `POST /pipeline/recipes/runs/{recipe_run_id}/resume`
  - resume reuses completed nodes from source workflow run and can start from explicit node id
  - Recipes run inspector now exposes Cancel/Retry/Resume actions
- Added strict dataset contract checks in training preflight (task-shape coverage gate with actionable fix hints).
- Expanded Training experiment form controls for memory/runtime tuning (`gradient_accumulation_steps`, `max_seq_length`, `save_steps`, `eval_steps`, `fp16`/`bf16`, `flash_attention`, `sequence_packing`).
- Added split backend dependency profiles for CPU/GPU installs (`requirements-base.txt`, `requirements.txt`, `requirements-gpu.txt`, `requirements-gpu-cu128.txt`).
- Added live external-training telemetry streaming (`epoch`, `step`, loss metrics) from Celery worker to Training dashboard via WebSocket.
- Added **Deployment Target SDK v1**:
  - deployment target catalog for exporters/runners (`HF`, `GGUF`, `ONNX`, `TensorRT`, `vLLM`, `TGI`, `Ollama`)
  - export-time deployability validation (artifact profile checks)
  - optional local runner smoke tests and explicit deployment reports in export manifests
- Added **Conformance + Scale Test Matrix (Phase 20)**:
  - automated backend matrix for `adapters x runtimes x model families x task types`
  - scale checks for adapter mapping quality at larger sample sizes
  - explicit guardrail assertions for model-task incompatibility and dataset-contract incompatibility
- Added Alembic revisions:
  - `20260305_0002` for registry + secrets tables
  - `20260305_0003` for domain profiles + project binding
  - `20260305_0004` for domain packs + project binding
  - `20260305_0005` for artifact registry table
  - `20260306_0007` for project training preference persistence
  - `20260306_0008` for project dataset adapter preset persistence
  - `20260307_0009` for project evaluation pack preference persistence

---

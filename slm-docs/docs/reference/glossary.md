---
sidebar_position: 4
title: Glossary
---

# Glossary

Every term you'll encounter, alphabetised. Use Cmd-F.

## Adapter

The mapping layer between your raw row shape and BrewSLM's canonical record (`{"text": ..., "messages": [...], ...}`). Adapters can be built-in (`qa-pair`, `chat-messages`, `structured-extraction`, `preference-pair`, `default-canonical`, `auto`) or plugin-defined. See [Adapter examples](../getting-started/adapter-studio-examples.md).

## Artifact

Any persisted output a downstream stage might depend on — a cleaned dataset, a checkpoint, an export bundle. Tracked in `artifact_records` with a `schema_ref` like `slm.checkpoint/v1`. See [Projects + artifacts](../concepts/projects-and-artifacts.md).

## Autopilot

The planner that turns a plain-language brief into a full pipeline plan (adapter, model, recipe, eval pack, target). v3 persists a decision log, snapshots state for rollback, and supports strict mode. See [Newbie Autopilot](../workflows/newbie-autopilot.md).

## Base model

The pretrained model you fine-tune from. Registered in the Universal Base Model Registry (one row per HF id), with per-project compatibility scoring.

## Beginner mode

Per-project flag that hides advanced UI surfaces (Adapter Studio, Extension Studio, Workflow Builder, Recipes, Pipeline-as-Code, Domain Packs, Domain Profiles). Backend behaviour unchanged. See [Beginner mode](../concepts/beginner-mode.md).

## Checkpoint

A saved training-loop state. Includes weights + optimizer state. Used by promote, resume-from, and rollback. See [Training → Checkpoint browser](../workflows/training.md#checkpoint-browser).

## Compression

Stage 10. Quantises / prunes a trained checkpoint. Methods: GGUF-Q4, GGUF-Q8, ONNX-INT8, pruning. See [Export + deployment → Compression](../workflows/export-and-deployment.md#compression).

## Confidence band

`high` / `medium` / `low` rating attached to cost estimates. `high` ≥ 0.8, `medium` ≥ 0.6, `low` otherwise.

## Confidence threshold (dataset import)

The 0.8 floor the [Schema introspection](#schema-introspection) pipeline applies before `--auto` will pick a mapping for you. Below the threshold, the CLI refuses to proceed unless you pass `--force`; the UI surfaces a warning banner asking for explicit confirmation. Encodes the "no silent auto-mapping" rule from the dataset-import plan — the proposal can be wrong, so the user always confirms before rows land in a project.

## Contract version

A version pin on a plugin contract, e.g. `slm.data_adapter/v3`. A plugin declaring an older version is rejected by the loader. See [Plugin contracts](../extensions/contracts.md).

## Decision log

Persistent record of every autopilot planning + repair action, with provenance per component. See [Newbie Autopilot → Decision log](../workflows/newbie-autopilot.md#decision-log).

## Deployability score

A 0–1 composite of smoke pass rate, telemetry health, drift delta, target compatibility, and historical reliability. Verdicts: `ready` / `caution` / `block`. See [Rollback + score](../deployment/rollback-and-score.md).

## Domain pack

A reusable overlay for a domain — dataset splits, training defaults, registry gates, data-quality rules, normalisation, tools, evaluation, audit. Adopts a default `domain profile` for safe starters. See [Pipeline overview](../workflows/pipeline-overview.md).

## Domain profile

A typed contract describing task family + compliance + runtime preferences. Attached to a project; consumed by domain packs.

## Drift check

Re-runs the gold-set eval against the **live deployment** endpoint and compares to the baseline pass rate. Three verdicts: `passing`, `borderline`, `drift_detected`. See [Drift checks](../deployment/drift-checks.md).

## Eval pack

Bundle of task-aware metric schemas + gate policies. Built-in packs cover general / QA / classification / summarisation / preference; custom packs via [scaffolding](../extensions/scaffold.md). Contract version `slm.evaluation-pack/v2`.

## Failure cluster

A group of error events folded on `(stage, reason_code, signature)`. The signature normalises away timestamps / ids / digit runs so similar errors collapse into one row. Two flavours:

- **Eval-stage clusters** — per-prediction failures inside one eval result.
- **Cross-stage clusters** — project-wide event clusters in the Observability page.

See [Failure clusters](../observability/failure-clusters.md).

## Gate policy

Rules on eval metrics that determine whether a model passes a stage or is allowed to promote. `required` gates block; `optional` gates allow degradation within tolerance.

## Gold set

The ground-truth labelled set you trust for evaluation. Two parallel surfaces:

- **Gold workbench** (Pipeline → Gold workbench) — sampling + reviewer-queue
  flow with versioned `draft → locked` state machine. DB-backed
  (`GoldSetRow` / `GoldSetVersion`). Best for hand-labelling at scale.
- **Gold set tab** (Pipeline → Gold set) — direct row CRUD + LLM-assisted
  generation. JSONL-backed on disk at `data/projects/{id}/gold/gold_dev.jsonl`.
  Best for bootstrapping a small set fast.

Both surfaces share the same on-disk JSONL — workbench rows materialize
into it, gold-set-tab rows append to it. The evaluator reads either path.
See [Evaluation + remediation](../workflows/evaluation-and-remediation.md).

## Train ↔ gold leakage

When rows in your gold set (the ruler that decides "does the model work?")
also appear in the training data — identical or near-identical copies. A
leaked gold set makes the eval pass-rate a lie: the model can recite the
answers from memory, so a green gate no longer means the model generalises.
The **Data Health Report** scans for it (`leakage.gold_train_overlap`),
catching both exact duplicates (the common bad-split / copy-paste case) and
near-duplicates (synthetic paraphrases of gold rows bleeding into train,
detected via token-set Jaccard ≥ 0.9). Any overlap warns; ≥ 10% of gold rows
blocks. A `GOLD_TEST` leak — the held-out final grade — is called out
separately as the worst kind. There is no one-click auto-fix (deciding which
copy is canonical is a judgement call); the Coach nudge routes you to re-split
so the gold set is held out. See
[Evaluation + remediation](../workflows/evaluation-and-remediation.md).

The same scanner also checks the **prepared splits** (`leakage.split_overlap`):
train↔validation, train↔test, and val↔test must be disjoint. A validation row
that's also in train makes your validation metric optimistic, so early-stopping
and checkpoint selection pick the wrong model; a test row in train (or in val)
inflates the final grade. Expanding a leakage signal opens a **drill-down** of
exactly which rows leaked, from which split, and the source row they matched.

## Hallucination trap

A gold row whose reference answer is *"I don't know"* / *"that's not in the
source"* — designed to test whether the model refuses fabrication. Tagged
via `is_hallucination_trap: true` on the row. The LLM-gen panel accepts an
explicit count of traps in its row-mix distribution (qa-sft only). See
[Step 1b — LLM-assisted gold](../workflows/evaluation-and-remediation.md#step-1b--or-build-the-gold-set-with-a-cloud-llm).

## Row mix (gold set)

The difficulty + hallucination-trap distribution across a qa-sft gold set.
The panel surfaces it as `N entries: X easy / Y medium / Z hard · W traps`
with a filter dropdown to scan a single bucket. The LLM-gen path accepts
an explicit `{easy, medium, hard, hallucination_traps}` distribution and
asks the LLM to craft + tag rows accordingly.

## Knowledge distillation (KD)

Training a small *student* model to match a strong *teacher*'s output
distribution, not just the hard labels — the single biggest quality lever an SLM
has. BrewSLM does **offline** KD: first capture the teacher's top-k token
log-probabilities for your dataset (`POST .../distillation/capture`), then train
with `training_mode="distillation"` so the loss mixes cross-entropy on the gold
label with KL on the teacher's soft targets:
`α·CE + (1−α)·T²·KL(student/T ‖ teacher/T)` (defaults α=0.5, temperature T=2.0).
No teacher model is loaded at train time — it reads the captured artifact. After
training, **quality retained** (`student/teacher` per metric + slice) shows how
much of the teacher you kept. Recipes: `recipe.kd.classification` / `recipe.kd.qa`
/ `recipe.kd.span_extraction`. Offline alignment is exact when teacher and
student share a tokenizer. See [Distillation workflow](../workflows/distillation.md).

## Label drift

When a project's gold set accumulates `positive` / `Positive` / `Positive (with sentiment)`
as separate labels — silently fragmenting eval metrics. The classification add-form
surfaces a soft amber-border warning on the Label input when you type a value not
in the existing vocabulary (case-insensitive). Same soft-warning pattern on the
span-extraction Type input.

## Manifest (training)

Immutable JSON snapshot of every input that produced a training run. Replay re-creates an identical experiment. See [Training → Reproducibility](../workflows/training.md#reproducibility--the-manifest).

## Manifest (project) / brewslm.yaml

Human-readable YAML representation of the whole project — datasets, adapters, recipes, eval packs, target profile. Round-trips via `brewslm manifest export/apply`. See [Pipeline-as-Code](../workflows/pipeline-overview.md).

## Measured

Provenance label meaning the value came from real observation. See [Measured vs estimated](../reliability/measured-vs-estimated.md).

## Estimated

Provenance label meaning the value came from a heuristic. Same page.

## Plugin contract

Formal Protocol interface for one of four plugin kinds (data_adapter, training_runtime, domain_pack, eval_pack). Validators check module interface, schema compliance, version metadata, and safe-reload support. See [Plugin contracts](../extensions/contracts.md).

## Provenance

The `measured` / `estimated` / `mixed` label attached to a metric. Every numeric output in BrewSLM has one. See [Measured vs estimated](../reliability/measured-vs-estimated.md).

## Reason code

Stable enum from `app/models/reason_codes.py`. Required on `error` / `critical` severity RunEvents. Pivots failure clustering + support bundles + the audit log.

### The 27 canonical reason codes

| Stage | Reason code |
|---|---|
| ingestion | `ingest_unsupported_format` `ingest_io_error` `ingest_validation_failed` |
| cleaning | `cleaning_outlier_threshold_exceeded` `cleaning_pii_block` |
| adapter | `adapter_schema_mismatch` `adapter_field_resolution_failed` |
| training | `training_dispatch_error` `training_runtime_error` `training_oom` `training_timeout` `training_cancelled` |
| eval | `eval_runtime_error` `eval_dataset_missing` `eval_judge_unavailable` |
| export | `export_run_failed` `export_artifact_missing` `export_quantization_failed` |
| deployment | `deployment_smoke_failed` `deployment_promote_blocked` `deployment_rollback_no_predecessor` `deployment_drift_detected` |
| autopilot | `autopilot_repair_blocked` `autopilot_strict_mode_refused` `autopilot_no_safe_plan` |
| system | `system_db_error` `system_config_invalid` `extension_load_failed` `extension_contract_invalid` |

## Recipe

A named training-config template (e.g. `safe-balanced-sft`, `lora-fast`, `classification`). Carries default learning rate, batch size, epochs, scheduler. Resolved at training start with provenance per field.

## Remediation suggestion

A concrete action surfaced by the eval failure cluster card — "add 20 rows", "lower LR", "switch model". Per cluster.

## Run event

Single observability row written by every pipeline stage. Schema: `(run_id, parent_run_id, stage, severity, reason_code, actor, summary, payload, ts)`. See [Run events](../observability/run-events.md).

## LLM-assisted mapping

Optional teacher-model fallback for the dataset-import introspector (Phase H). When the column-content sniffer can't form a high-confidence hypothesis on its own, the introspector can also ask the project's teacher model "given these column names + sample values, which mapper fits?" and merge the LLM's JSON response into the ranked hypothesis list. The LLM proposal is treated exactly like a deterministic one — same [Confidence threshold](#confidence-threshold-dataset-import) gate, same `--auto` / `--force` flow — and any mapper id the model hallucinates is rejected at the registry boundary. Tagged with `proposal-source: llm-assist` in the warnings list so the UI / CLI can highlight LLM entries distinctly. Disabled by default behind `DATASET_IMPORT_LLM_ASSIST_ENABLED`; requires `TEACHER_MODEL_API_URL` to be set. Opt-in per call via `--llm-assist` on the CLI or `llm_assist: true` on the `/dataset-import/introspect` request body. Implementation: [`backend/app/services/dataset_import/llm_assist.py`](https://github.com/anugram/__SLM__/blob/main/backend/app/services/dataset_import/llm_assist.py).

## Mapper plugin

A user-supplied Python module that registers extra [target mappers](#target-mapper-dataset-import) alongside the built-ins (Phase H). Two registration shapes are accepted:
- `register_dataset_mappers(register)` hook — preferred. `register(mapper_id, factory)` adds one mapper to the registry; the factory is a zero-arg callable returning an object that satisfies the `TargetMapper` protocol.
- Top-level `DATASET_MAPPERS: dict[str, factory]` constant — declarative form.

Modules are listed in `settings.DATASET_MAPPER_PLUGIN_MODULES` and imported at app boot (parallel to `DATA_ADAPTER_PLUGIN_MODULES`, `TRAINING_RUNTIME_PLUGIN_MODULES`, etc). A misconfigured plugin module doesn't block the rest of the list — the loader records the error per module and continues.

## Saved mapping (dataset import)

A persisted `(locator, mapper_id, field_map, drop_reasons)` tuple under a user-chosen name. Created via the "Save this mapping" card on the import wizard's Preview step; re-runs go through the **Saved mappings** panel on the Data tab. Re-runs use the same `run_import` code path as a fresh import, so they emit the same audit [run event](#run-event) (`stage=ingestion`, `reason_code=dataset_import_run`) plus a `config_id` link in the payload. `last_run_at` and `last_run_accepted` columns on the config row track the most recent re-run's timestamp and row count for at-a-glance status. Backed by the `dataset_import_configs` table; API: `POST /api/projects/{id}/dataset-import/configs` (create) → `POST /api/projects/{id}/dataset-import/configs/{cfg_id}/run` (re-run).

## Scoring mode

Sub-mode of a task handler that picks the metric shape without changing which handler runs. Today's modes live inside `StructuredExtractionHandler`: `field_match` (per-field EM/F1 — invoice / form extraction) and `span_set` (entity-level P/R/F1 — PII / NER / span extraction). Declared as `output_schema.scoring_mode` on the prepared manifest. Same general handler, internal dispatch — BrewSLM doesn't add a new handler per domain. See [PII detector demo](../demos/pii-detector.md#how-scoring-works).

## Schema introspection

The dataset-import pipeline's column sniffer + shape detector + mapping proposer. Reads ~20 sample rows from any registered [source connector](#source-connector), classifies each column by content (`text_like`, `categorical`, `bio_tag_list`, `entity_list_json`, `chat_messages`, `tokens_list`, `numeric`, `boolean`, `path_like`), and proposes the best-fit [target mapper](#target-mapper) + field map with a confidence score and rationale. Drives the `--auto` flag on the dataset-import CLI and the import-wizard UI (Pipeline → Data → "Import dataset (auto-mapping)"); never silently auto-picks — the user confirms either by passing `--auto` (when confidence ≥ the [Confidence threshold](#confidence-threshold-dataset-import)), by passing `--force` under it, or by accepting the proposal in the wizard's Map step. CLI: `python -m app.cli.dataset_import introspect --locator <prefix:rest>`.

Optional [LLM-assisted mapping](#llm-assisted-mapping) mode is gated behind `DATASET_IMPORT_LLM_ASSIST_ENABLED`; pass `--llm-assist` on the CLI / `llm_assist: true` in the API body to opt in per-call.

## SLM

Small Language Model. The 1B–10B parameter range that BrewSLM is optimised for. Small enough to fine-tune on a single GPU; useful enough to ship in production.

## Span-set scoring

Entity-level precision/recall/F1 for tasks whose output is a list of typed spans (`[{type, start, end, text}, ...]`). True positives require identical `(type, start, end)`; off-by-one boundaries count as miss + hallucination. Produces per-class P/R/F1 alongside micro and macro aggregates — the metric shape compliance teams need ("99% credit_card recall before ship"). Triggered by `output_schema.scoring_mode: "span_set"`. See [PII detector demo](../demos/pii-detector.md).

## Snapshot

Pre-run autopilot state capture. Rollback restores a snapshot. See [Newbie Autopilot → Rollback](../workflows/newbie-autopilot.md#rollback).

## Source connector (dataset import)

Pluggable loader for an external dataset system — one file per source, registered into the dataset-import registry at module-load time. Connectors are addressed via locator prefix and implement `load(locator, *, limit)` (lazy streaming of raw row dicts) and `describe(locator)` (sample rows + column names + approximate row count — the introspector's input). Catalog today:

| Source id | Locator format | Notes |
|---|---|---|
| `jsonl` | `jsonl:/path/to/file.jsonl` | One JSON object per line. Unparseable lines surface as `__parse_error__` sentinel rows. |
| `csv` | `csv:/path/to/file.csv` | First row is the header. Every cell is a string; mappers handle type coercion. |
| `hf` | `hf:<dataset_id>[:<split>[:<revision>]]` | Wraps `datasets.load_dataset` with `streaming=True`. Auth via `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN`. Multi-split DatasetDict picks first key when split isn't pinned. |
| `kaggle` | `kaggle:competition:<slug>` / `kaggle:dataset:<owner/slug>` (optional `?file=<path>`) | Downloads + extracts via the Kaggle Python API. Cache under `$BREWSLM_KAGGLE_CACHE` (default `~/.cache/brewslm/kaggle/`). Auth via `KAGGLE_USERNAME` + `KAGGLE_KEY` or `~/.kaggle/kaggle.json`. |

Plugin sources via Phase H.

## Starter pack

Novice-oriented bootstrap that pre-fills model family, adapter, gates, target, and safety reminders for a domain. Picked at project creation time.

## Stage

One of the 11 canonical pipeline stages (ingestion → export) + three virtual stages (deployment, autopilot, system). See [Pipeline stages](../concepts/pipeline-stages.md).

## Strict mode

Autopilot mode that refuses to take any fallback path; surfaces every blocker verbatim. Reach for it when reproducibility matters more than convenience.

## Support bundle

Redacted zip export of recent RunEvents + failure clusters + deployment state + experiments + autopilot decisions. For hand-off to support / oncall. See [Support bundles](../observability/support-bundles.md).

## Target mapper (dataset import)

Pluggable transform that turns raw source rows into one of BrewSLM's canonical task shapes. Domain-agnostic by design — a single `bio_to_spans` mapper serves PII, medical, legal, and financial NER; the per-row [Schema introspection](#schema-introspection) result picks the right one and fills the field map. Every mapper declares its target task profile so the orchestrator validates project/mapper compatibility before a row lands. Catalog today (one file per mapper, registered at import time):

| Mapper id | Target task profile | Use when the row carries… |
|---|---|---|
| `bio_to_spans` | `structured_extraction` (span_set) | BIO-tagged tokens + labels (NER / PII) |
| `label_to_classification` | `classification` | `{text, label}` |
| `text_only` | `language_modeling` | a single text column, no labels |
| `qa_pair_passthrough` | `qa` | `{question, answer}` |
| `chat_messages_passthrough` | `chat_sft` | a `messages` list of `{role, content}` dicts |
| `preference_pair` | `dpo` | `{prompt, chosen, rejected}` (RLHF / DPO / ORPO) |
| `rag_passthrough` | `rag_qa` | `{question, context, answer}` (grounded QA) |
| `kv_to_structured` | `structured_extraction` (field_match) | flat key-value extractions (invoices / forms) |

The introspector's `--auto` flow detects every shape above *except* `kv_to_structured` — that one needs an explicit `fields` config the introspector can't infer. Plugin mappers via Phase H.

## Target profile

Deployment-environment shape: `mobile_cpu`, `browser_webgpu`, `edge_gpu`, `vllm_server`. Determines weight budget, compression preference, runtime constraints. Pluggable via the target-profile plugin kind.

## Telemetry sample

One inference observation pushed to BrewSLM by your serving runtime (status, latency_ms, prompt_tokens, completion_tokens, TTFT). Rolls up into the deployment telemetry window. See [Post-deploy telemetry](../deployment/telemetry.md).

## Trainability forecast

Pre-training "will this clear gates?" prediction shown above the Preflight button on the Training Config page. Combines recipe-agnostic signals (row count, gold-set diversity, gate-pass probability heuristic) with per-recipe signals dispatched by `task_profile`. Classification adds class-imbalance, per-class minimums, label-vocab fragmentation, and single-class dominance. Span-extraction adds span-offset validity, entity-type coverage, and negative-example presence. Summarization adds summary/document length-ratio outlier detection. Advisory only — never blocks training; the Train button shifts to "Train anyway" when the verdict is amber/red. A snapshot of every cache-miss compute is persisted to `training_forecast_snapshots` and surfaced as a sparkline + verdict-delta strip above the signal list so the user can see whether their last edit moved the needle (60-day retention). See [Training workflow](../workflows/training.md#trainability-forecast).

## Training mode

What kind of fine-tuning the trainer runs: `sft` (instruction), `dpo` (direct preference), `orpo` (odds-ratio preference), `classification`, `seq2seq`, `distillation` (offline KD against captured teacher logits — see [Knowledge distillation](#knowledge-distillation-kd)). Filtered by the chosen base model's capabilities.

## Hyperparameter sweep · cost axis (`cost_kind`)

Which axis the Pareto scatter uses to score "cost" in the hyperparameter bake-off. Three supported values, picked via the radio above the scatter (and as `?cost_kind=` on `GET .../training/sweeps/{sweep_id}`):

- `wall_clock_seconds` — measured `completed_at - started_at` per cell. The honest default: real training time captured by the platform. Cells without timestamps surface as `cost_source="pending"` and sit out the scatter rather than being fabricated.
- `lora_r` — adapter footprint proxy. Cheap and immediately available, but a fiction when `base_model` is also a swept axis (rank-16 on a 135M base does not cost what rank-16 on a 3B base costs).
- `base_params_m` — base-model parameter count in millions. The right axis when the sweep varies `base_model` and you're reading the Pareto as a model-size trade-off. Models outside the platform catalog surface as `cost_source="unknown_base_model"`.

Unsupported values return 400 from the API rather than silently falling back. See [Hyperparameter bake-off](../workflows/training.md#model--hyperparameter-bake-off-pareto).

## Sweep pre-flight budget · `basis`

Estimator returned by `POST .../training/sweeps/preflight-budget` for a planned grid: `{cell_count, seconds_per_cell, estimated_seconds, basis, sample_size}`. The `basis` field labels how the seconds-per-cell median was derived, in order of tightness:

- `same_base_and_recipe` — median of prior cells in this project on the same base model AND recipe. Tightest signal.
- `same_base_model` — median of prior cells on the same base model (any recipe). Used when no same-recipe history exists.
- `project_default` — median across any prior sweep cells in the project. Loose, but better than guessing.
- `no_history` — no prior cells at all; falls back to a conservative default (`DEFAULT_NO_HISTORY_SECONDS_PER_CELL`, ~2 min). UI labels this as "rough estimate, no prior runs".

Dollars are deliberately not reported — GPU cost depends on the runtime backend (local GB10 = $0, cloud-burst = variable). Wall-clock is the honest unit we always have.

## Sweep quality target · stop-when-met

Threshold the orchestrator uses to decide "winner found — cancel the rest of the sweep". Stored on each cell's `_sweep.quality_target` at dispatch. The watcher is lazy-on-fetch: the panel polls `get_sweep_pareto` every 4s while cells train, and that endpoint checks "did any completed cell clear the target?" — if yes and other cells are still running, it fires `cancel_training` against them and annotates each row with `cancelled_by_target=true`. The launcher accepts either decimal (`0.85`) or percent (`85`) form; the percent form is coerced. Blank target = run the full grid to completion (legacy behaviour).

## Sweep verdict · promote / inconclusive / pending

Three-state outcome on a hyperparameter sweep, surfaced by `get_sweep_pareto` after running each completed cell through `evaluate_experiment_auto_gates`:

- `promote` — at least one cell cleared the project's evaluation pack gate. The UI offers promote-to-base backed by a real signal.
- `inconclusive` — every completed cell has eval results but none cleared the gate. The UI surfaces "nobody cleared `<gate_id>`" and links to the failure-cluster panel rather than letting the user quietly promote a sub-gate winner.
- `pending` — cells are still running, or completed cells don't have eval results yet (`gate_passed=null`).

Per-cell, `gate_passed ∈ {true, false, null}` and `gate_failed_ids` lists the specific gate IDs the cell missed. A pack with zero gates is treated as `not measurable` (gate_passed=null) — passing trivially when no gates exist is exactly the vanity behaviour the honesty pass prevents. See [Winner-vs-gate verdict](../workflows/training.md#winner-vs-gate-verdict).

## See also

- [Reason-code taxonomy](../observability/failure-clusters.md#reason-code-taxonomy) — same table with more context.
- [Pipeline stages](../concepts/pipeline-stages.md) — the 11-stage map.

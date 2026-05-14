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

The ground-truth labelled set you trust for evaluation. Versioned (draft → locked) and immutable once locked. Sampled stratified / random / targeted from the cleaned dataset. See [Evaluation + remediation](../workflows/evaluation-and-remediation.md).

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

## Scoring mode

Sub-mode of a task handler that picks the metric shape without changing which handler runs. Today's modes live inside `StructuredExtractionHandler`: `field_match` (per-field EM/F1 — invoice / form extraction) and `span_set` (entity-level P/R/F1 — PII / NER / span extraction). Declared as `output_schema.scoring_mode` on the prepared manifest. Same general handler, internal dispatch — BrewSLM doesn't add a new handler per domain. See [PII detector demo](../demos/pii-detector.md#how-scoring-works).

## Schema introspection

The dataset-import pipeline's column sniffer + shape detector + mapping proposer. Reads ~20 sample rows from any registered [source connector](#source-connector), classifies each column by content (`text_like`, `categorical`, `bio_tag_list`, `entity_list_json`, `chat_messages`, `tokens_list`, `numeric`, `boolean`, `path_like`), and proposes the best-fit [target mapper](#target-mapper) + field map with a confidence score and rationale. Drives the `--auto` flag on the dataset-import CLI and the import-wizard UI; never silently auto-picks — the user confirms either by passing `--auto` (when confidence ≥ the [Confidence threshold](#confidence-threshold-dataset-import)), by passing `--force` under it, or by accepting the proposal in the UI wizard. CLI: `python -m app.cli.dataset_import introspect --locator <prefix:rest>`.

## SLM

Small Language Model. The 1B–10B parameter range that BrewSLM is optimised for. Small enough to fine-tune on a single GPU; useful enough to ship in production.

## Span-set scoring

Entity-level precision/recall/F1 for tasks whose output is a list of typed spans (`[{type, start, end, text}, ...]`). True positives require identical `(type, start, end)`; off-by-one boundaries count as miss + hallucination. Produces per-class P/R/F1 alongside micro and macro aggregates — the metric shape compliance teams need ("99% credit_card recall before ship"). Triggered by `output_schema.scoring_mode: "span_set"`. See [PII detector demo](../demos/pii-detector.md).

## Snapshot

Pre-run autopilot state capture. Rollback restores a snapshot. See [Newbie Autopilot → Rollback](../workflows/newbie-autopilot.md#rollback).

## Source connector (dataset import)

Pluggable loader for an external dataset system — one file per source, registered into the dataset-import registry at module-load time. Built-ins ship for `jsonl` and `csv`; planned connectors cover `hf` (HuggingFace), `kaggle`, and `parquet`. Connectors are addressed via locator prefix: `jsonl:/path/to/file`, `hf:org/dataset:split`, etc. Every connector implements `load(locator, *, limit)` (lazy streaming of raw row dicts) and `describe(locator)` (sample rows + column names + approximate row count — the introspector's input).

## Starter pack

Novice-oriented bootstrap that pre-fills model family, adapter, gates, target, and safety reminders for a domain. Picked at project creation time.

## Stage

One of the 11 canonical pipeline stages (ingestion → export) + three virtual stages (deployment, autopilot, system). See [Pipeline stages](../concepts/pipeline-stages.md).

## Strict mode

Autopilot mode that refuses to take any fallback path; surfaces every blocker verbatim. Reach for it when reproducibility matters more than convenience.

## Support bundle

Redacted zip export of recent RunEvents + failure clusters + deployment state + experiments + autopilot decisions. For hand-off to support / oncall. See [Support bundles](../observability/support-bundles.md).

## Target mapper (dataset import)

Pluggable transform that turns raw source rows into one of BrewSLM's canonical task shapes. Domain-agnostic by design — a single `bio_to_spans` mapper serves PII, medical, legal, and financial NER; the per-row [Schema introspection](#schema-introspection) result picks the right one and fills the field map. Built-ins ship for `bio_to_spans` (→ structured_extraction) and `label_to_classification` (→ classification); the planned mapper catalog adds `preference_pair`, `rag_passthrough`, `qa_pair_passthrough`, `chat_messages_passthrough`, `kv_to_structured`, and `text_only`. Every mapper declares its target task profile so the orchestrator validates project/mapper compatibility before a row lands.

## Target profile

Deployment-environment shape: `mobile_cpu`, `browser_webgpu`, `edge_gpu`, `vllm_server`. Determines weight budget, compression preference, runtime constraints. Pluggable via the target-profile plugin kind.

## Telemetry sample

One inference observation pushed to BrewSLM by your serving runtime (status, latency_ms, prompt_tokens, completion_tokens, TTFT). Rolls up into the deployment telemetry window. See [Post-deploy telemetry](../deployment/telemetry.md).

## Training mode

What kind of fine-tuning the trainer runs: `sft` (instruction), `dpo` (direct preference), `orpo` (odds-ratio preference), `classification`, `seq2seq`. Filtered by the chosen base model's capabilities.

## See also

- [Reason-code taxonomy](../observability/failure-clusters.md#reason-code-taxonomy) — same table with more context.
- [Pipeline stages](../concepts/pipeline-stages.md) — the 11-stage map.

---
sidebar_position: 2
title: API surface
---

# API surface

A practical, hand-curated index of the most-used HTTP endpoints. For exhaustive request/response schemas use the live Swagger UI at **`http://localhost:8000/api/docs`** — every endpoint here is documented there with full types.

Every route is prefixed by `/api`. Most are project-scoped (`/api/projects/{id}/...`). Auth modes are covered in [Setup → Auth + SSO](../setup/auth-and-sso.md).

## Authentication

```
POST   /api/auth/local/login            Local username+password → JWT
POST   /api/auth/sso/start              Begin OIDC SSO flow
POST   /api/auth/sso/callback           Finish OIDC SSO flow
GET    /api/auth/me                     Current principal (user_id, role)
POST   /api/auth/logout                 Invalidate the current session
```

## Projects + pipeline

```
GET    /api/projects                    List projects
POST   /api/projects                    Create a project
GET    /api/projects/{id}               Fetch one
PUT    /api/projects/{id}               Update (beginner_mode, name, etc.)
DELETE /api/projects/{id}               Archive

GET    /api/projects/{id}/pipeline/status      Per-stage status
GET    /api/projects/{id}/runtime/readiness    Deep readiness (used by `doctor`)
```

## Domain contracts

```
GET    /api/domain-packs
POST   /api/domain-packs                Create a pack
GET    /api/domain-profiles
POST   /api/domain-blueprints/analyze   Analyse a plain-language brief
POST   /api/projects/{id}/domain-blueprints
POST   /api/projects/{id}/domain-blueprints/{version}/apply
GET    /api/projects/{id}/domain-runtime
```

## Datasets + ingestion

```
POST   /api/projects/{id}/ingest                Pull from HF / URL / Kaggle (queued)
POST   /api/projects/{id}/datasets/upload       Upload local file (multipart)
POST   /api/projects/{id}/datasets/profile      Inspect schema, sample rows
POST   /api/projects/{id}/datasets/clean        Run cleaning stage
POST   /api/projects/{id}/datasets/prepare      Run adapter + split
POST   /api/projects/{id}/datasets/tokenize     Tokenize with a base model
GET    /api/projects/{id}/datasets              List dataset versions

POST   /api/projects/{id}/synthetic/generate    Synthetic Q&A augmentation
```

## Data adapter (Adapter Studio)

```
POST   /api/projects/{id}/data-adapter/profile       Profile a source file
POST   /api/projects/{id}/data-adapter/infer         Rank adapter candidates
POST   /api/projects/{id}/data-adapter/preview       Map sample rows
POST   /api/projects/{id}/data-adapter/validate      Full coverage check
GET    /api/projects/{id}/adapter-studio/adapters    List project adapters
POST   /api/projects/{id}/adapter-studio/adapters/{name}/versions/{version}/export
```

## Base Model Registry

```
GET    /api/models                              Browse catalog
POST   /api/models/import                       Import from HF
POST   /api/models/refresh                      Re-pull metadata
GET    /api/projects/{id}/models/recommend      Project-aware recommendation
POST   /api/projects/{id}/models/{model_id}/validate
GET    /api/projects/{id}/models/compatible
POST   /api/projects/{id}/models/explain        Why was this scored?
PUT    /api/projects/{id}/models/default        Pin a project default
```

## Training

```
POST   /api/projects/{id}/training/preflight    Validate a config (no run)
POST   /api/projects/{id}/experiments           Start training
POST   /api/projects/{id}/experiments/rerun     Replay an existing manifest
POST   /api/projects/{id}/experiments/clone     Replay + overrides
GET    /api/projects/{id}/experiments           List
GET    /api/projects/{id}/experiments/{eid}     Get one (status, metrics)
POST   /api/projects/{id}/experiments/{eid}/pause
POST   /api/projects/{id}/experiments/{eid}/resume
POST   /api/projects/{id}/experiments/{eid}/cancel
GET    /api/projects/{id}/experiments/{eid}/checkpoints
POST   /api/projects/{id}/experiments/{eid}/checkpoints/{step}/promote
POST   /api/projects/{id}/experiments/{eid}/checkpoints/{step}/resume

POST   /api/projects/{id}/training/plan/estimate-cost   Cost + provenance
GET    /api/projects/{id}/training/recipes              Recipe catalog
POST   /api/projects/{id}/training/recipes/resolve      Recipe + defaults preview
GET    /api/projects/{id}/training/runtimes             Runtime catalog
```

## Distillation (knowledge distillation)

```
POST   /api/projects/{id}/distillation/capture           Capture teacher top-k
                                                         logprobs for a dataset
                                                         → 202 + task envelope
GET    /api/projects/{id}/distillation/tasks/{task_id}   Poll capture progress
```

Offline KD training is then a normal training run with `training_mode="distillation"`
(recipes `recipe.kd.classification` / `recipe.kd.qa` / `recipe.kd.span_extraction`);
it reads the captured artifact rather than loading a live teacher. See
[Distillation workflow](../workflows/distillation.md).

## Autopilot

```
POST   /api/projects/{id}/autopilot/plan             Generate a plan from intent
POST   /api/projects/{id}/autopilot/run              Execute a plan
POST   /api/projects/{id}/autopilot/repair-preview   Preview without applying
POST   /api/projects/{id}/autopilot/rollback         Restore a pre-run snapshot
GET    /api/projects/{id}/autopilot/decisions        Decision log
GET    /api/projects/{id}/autopilot/snapshots        List snapshots
```

## Pipeline-as-Code (manifest)

```
GET    /api/projects/{id}/manifest/export?format=yaml      Export project → YAML
POST   /api/manifest/validate                              Validate against schema
POST   /api/projects/{id}/manifest/diff                    Diff vs current state
POST   /api/projects/{id}/manifest/apply                   Apply (writes)
POST   /api/manifest/apply                                 Apply new project (top-level)
```

## Pipeline workflow runner

```
POST   /api/projects/{id}/workflows/compile        Validate graph
POST   /api/projects/{id}/workflows/{wid}/dry-run  Plan execution
POST   /api/projects/{id}/workflows/{wid}/run      Run
GET    /api/projects/{id}/workflows/runs           History
```

## Evaluation

```
POST   /api/projects/{id}/eval/generate            Generate a starter pack
POST   /api/projects/{id}/eval/run                 Score an experiment
GET    /api/projects/{id}/eval/{eval_id}/clusters  Failure clusters
GET    /api/projects/{id}/eval/{eval_id}/remediation
GET    /api/projects/{id}/eval/compare?a=X&b=Y     Side-by-side
GET    /api/projects/{id}/evaluation/gates/{experiment_id}
GET    /api/projects/{id}/evaluation/sft-lift-summary       Baseline → trained lift
GET    /api/projects/{id}/evaluation/student-teacher-comparison/{eid}
                                                  Distillation quality retained
                                                  (student/teacher); optional
                                                  ?teacher_run_id=N

# Gold sets — workbench (sampling + review)
POST   /api/projects/{id}/gold-sets                       Create
POST   /api/projects/{id}/gold-sets/{gid}/sample          Sample rows
PATCH  /api/projects/{id}/gold-sets/{gid}/rows/{row_id}   Label one
POST   /api/projects/{id}/gold-sets/{gid}/submit          Lock draft
GET    /api/projects/{id}/gold-sets/{gid}/queue           Reviewer queue

# Gold rows — direct CRUD (JSONL-backed, all recipes)
GET    /api/projects/{id}/gold/entries?dataset_type=gold_dev   List rows
POST   /api/projects/{id}/gold/add                             Add one row (any recipe)
POST   /api/projects/{id}/gold/import                          Bulk import (any recipe)
POST   /api/projects/{id}/gold/lock?dataset_type=gold_dev      Lock the set

# LLM-assisted gold generation (qa-sft, classification, span-extraction, summarization)
POST   /api/projects/{id}/gold/generate-via-llm                Fire generation
POST   /api/projects/{id}/gold/generate-via-llm/preview-prompt Preview the would-be prompt
POST   /api/projects/{id}/gold/generate-via-llm/cost-estimate  Pre-call cost badge
GET    /api/projects/{id}/gold/generate-via-llm/saved-key      Has stored key + hint?
PUT    /api/projects/{id}/gold/generate-via-llm/saved-key      Store / replace
DELETE /api/projects/{id}/gold/generate-via-llm/saved-key      Clear
```

The `/gold/add` and `/gold/import` endpoints accept arbitrary recipe-shaped fields
(qa-sft `{question, answer}`, classification `{text, label}`, span-extraction
`{text, entities}`, summarization `{document, summary}`) plus optional
`difficulty`/`is_hallucination_trap`/`criticality` metadata. Extras round-trip
verbatim into the JSONL; system-owned `id` / `created_at` always win over
caller-supplied values.

`/gold/generate-via-llm` accepts optional `user_prompt_override` /
`system_prompt_override` (the "Review & edit prompt before sending" UX path) and
an optional `distribution: {easy, medium, hard, hallucination_traps}` block
(qa-sft only — silently ignored elsewhere).

## Export

```
GET    /api/projects/{id}/export/deployment-targets  Target profile catalog
POST   /api/projects/{id}/export/optimize            Candidate ranking
POST   /api/projects/{id}/export                     Run an export
GET    /api/projects/{id}/exports                    List
GET    /api/projects/{id}/exports/{export_id}        Get one
```

## Compression

```
POST   /api/projects/{id}/compression/jobs           Queue a compression job
GET    /api/projects/{id}/compression/jobs/{jid}     Status
WS     /api/ws/compression/{jid}/logs                Stream logs
```

## Deployment

```
POST   /api/projects/{id}/deployments/plan           Plan + readiness checks
POST   /api/deployments/{did}/smoke                  Smoke test
POST   /api/deployments/{did}/promote                Promote to active
POST   /api/deployments/{did}/reject                 Reject (audit log)
POST   /api/deployments/{did}/rollback               Roll back to a prior version
GET    /api/deployments/{did}                        Get one
GET    /api/projects/{id}/deployments                List

# Telemetry
POST   /api/deployments/{did}/telemetry/samples      Push one sample
GET    /api/deployments/{did}/telemetry              Rolling window

# Drift + score
POST   /api/deployments/{did}/drift/check
GET    /api/deployments/{did}/drift/history
GET    /api/deployments/{did}/deployability          Composite score
```

## Observability

```
GET    /api/projects/{id}/run-events                 List events (paginated)
GET    /api/run-events/run/{run_id}                  Events for one run_id
GET    /api/projects/{id}/timeline                   Tree-ordered timeline
GET    /api/projects/{id}/failure-clusters           List clusters
POST   /api/projects/{id}/failure-clusters/recompute Idempotent recompute

POST   /api/projects/{id}/support-bundle             Create a redacted bundle
GET    /api/projects/{id}/support-bundles            List for a project
GET    /api/support-bundles/{uid}/download?token=... Stream the zip
```

## Extensions (plugins)

```
GET    /api/extensions                       Catalog of all four kinds
POST   /api/extensions/validate              Import + contract check
POST   /api/extensions/reload                Reload from settings
POST   /api/extensions/scaffold              Generate a starter
```

## Audit + settings

```
GET    /api/audit/recent                     Paginated audit log
GET    /api/settings/runtime                 Read live settings
PUT    /api/settings/runtime                 Override (when permitted)
GET    /api/health                           Liveness probe
```

## Stable reason codes

Every `400` / `403` / `404` / `410` returns a `detail` string. Most are stable codes you can switch on programmatically:

| Code | When |
|---|---|
| `project_not_found` | Bad project id. |
| `not_a_gold_set` | Dataset id was not `dataset_type` gold. |
| `invalid_stage` | RunEvent stage not in canonical set. |
| `invalid_severity` | RunEvent severity not in canonical set. |
| `invalid_reason_code` | Reason code not in `reason_codes.py`. |
| `reason_code_required` | Error/critical severity without a code. |
| `invalid_window` | `since >= until` on a timeline / cluster query. |
| `support_bundle_invalid_token` | Constant-time token compare failed. |
| `support_bundle_expired` | Past `expires_at`. |
| `unknown_plugin_kind` | Extensions API received a non-canonical kind. |
| `scaffold_plugin_id_required` | Empty `plugin_id` on scaffold. |
| `scaffold_plugin_id_invalid` | `plugin_id` normalised to empty. |

## See also

- Live Swagger UI: `http://localhost:8000/api/docs`.
- [CLI reference](cli.md) — equivalent commands.
- [Reason codes glossary](glossary.md) — every code with its meaning.

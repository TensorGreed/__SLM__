---
sidebar_position: 1
title: CLI reference
---

# CLI reference

Every BrewSLM command at a glance. For full flags on any subcommand, run `brewslm <cmd> --help`.

## Global options

| Flag | Default | Effect |
|---|---|---|
| `--api-base URL` | `http://localhost:8000/api` | Where the CLI sends requests. |
| `--token TOKEN` | `$BREWSLM_TOKEN` | Bearer / API-key auth. Set the env var to avoid passing on every call. |
| `--timeout-seconds N` | `60` | HTTP timeout. |

## Top-level commands

```
brewslm project       Manage projects (create, list, beginner, blueprint)
brewslm dataset       Upload, profile, clean, prepare, tokenize datasets
brewslm adapter       Data-adapter studio workflows (profile, infer, preview, validate, export)
brewslm models        Universal Base Model Registry ops
brewslm ingest        Pull from HuggingFace / URL / Kaggle
brewslm preflight     Run training preflight without launching
brewslm train         Start, rerun, clone, pause, resume, cancel, checkpoints
brewslm repro         Reproducibility manifest (show, diff, export)
brewslm manifest      Pipeline-as-code: export, validate, diff, apply
brewslm pipeline      Workflow runner (compile, dry-run, run, history)
brewslm export        Build deployable artifacts
brewslm doctor        Readiness + deep observability snapshot
brewslm optimize      Optimization candidate ranking
brewslm autopilot     Plan, run, repair-preview, rollback, decision logs
brewslm deploy        Plan, smoke, promote, reject, rollback, telemetry, drift, score
brewslm logs          Tail RunEvents, list clusters, fetch timeline
brewslm support-bundle  Create, list, download redacted bundles
brewslm scaffold      Generate plugin starter (adapter, runtime, domain-pack, eval-pack)
brewslm extensions    List, validate, reload plugins
brewslm eval          Gold-set ops, run, compare, clusters, remediate
brewslm auth          Login + whoami
brewslm template      List, get, instantiate Project Templates (cloneable starting kits)
brewslm serve         Plan / start / get / stop serve runs for a compressed export
brewslm version       Print CLI + (with --remote) backend version
```

## Project

```sh
brewslm project create --name "Phase A" --template general
brewslm project list
brewslm project budget --id 1
brewslm project beginner --id 1 --enable        # or --disable
brewslm project blueprint show    --project 1 --latest
brewslm project blueprint diff    --project 1 --from-version 1 --to-version 2
brewslm project bootstrap         --name "Q&A bot" \
                                  --brief "..." --target edge_gpu \
                                  --create-project
```

## Dataset + adapter

```sh
# Profile a file before ingesting
brewslm dataset profile --project 1 --source-type csv --source-ref ./data.csv --json

# Upload + clean + prepare
brewslm dataset upload   --project 1 --source-type csv --source-ref ./data.csv --name v1
brewslm dataset clean    --project 1 --dataset v1
brewslm dataset prepare  --project 1 --dataset v1 --adapter-id auto --seed 42
brewslm dataset tokenize --project 1 --dataset v1 --base-model 12

# Adapter workflows
brewslm adapter infer    --project 1 --source-type jsonl --source-ref ./chat.jsonl --json
brewslm adapter preview  --project 1 --source-type jsonl --source-ref ./chat.jsonl --adapter-id auto
brewslm adapter validate --project 1 --source-type jsonl --source-ref ./chat.jsonl --adapter-id auto
brewslm adapter export   --project 1 --adapter-name my_adapter --version 2
```

## Base Model Registry

```sh
brewslm models list      --family qwen --hardware-fit server --json
brewslm models import    --hf-id "Qwen/Qwen2.5-1.5B-Instruct" --json
brewslm models refresh   --model 1 --json
brewslm models recommend --project 1 --limit 5 --hardware-fit server --json
brewslm models validate  --project 1 --model 1 --json
brewslm models set-default --project 1 --model 1
```

## Training

```sh
# Start fresh
brewslm train start --project 1 \
  --recipe safe-balanced-sft \
  --base-model 12 \
  --training-mode sft

# Autopilot one-click
brewslm train start --project 1 --autopilot --one-click \
  --intent "Support FAQ tone, vLLM deploy."

# Reproduce / iterate
brewslm train rerun --project 1 --experiment 42
brewslm train clone --project 1 --experiment 42 \
  --config-overrides '{"learning_rate": 3e-4}'

# Control
brewslm train pause   --project 1 --experiment 42
brewslm train resume  --project 1 --experiment 42
brewslm train cancel  --project 1 --experiment 42 --reason "wrong recipe"

# Checkpoints
brewslm train checkpoints --project 1 --experiment 42
brewslm train checkpoints --project 1 --experiment 42 --promote-step 200
brewslm train checkpoints --project 1 --experiment 42 --resume-from-step 150

# Reproducibility manifest
brewslm repro manifest --project 1 --experiment 42
```

## Autopilot

```sh
brewslm autopilot plan         --project 1 --intent "..."
brewslm autopilot run          --project 1 --plan auto_abc...
brewslm autopilot run          --project 1 --intent "..." --one-click --strict
brewslm autopilot repair-preview --project 1 --plan auto_abc...
brewslm autopilot rollback     --project 1 --snapshot snap_8c9d...
brewslm autopilot show         --decision dec_3f8...
```

## Manifest (Pipeline-as-Code)

```sh
brewslm manifest export   --project 1 --format yaml > project.brewslm.yaml
brewslm manifest validate --file project.brewslm.yaml
brewslm manifest diff     --project 1 --file project.brewslm.yaml
brewslm manifest apply    --project 1 --file project.brewslm.yaml
brewslm manifest apply    --file new-project.brewslm.yaml --plan-only
```

## Pipeline (workflow runner)

```sh
brewslm pipeline compile --project 1 --file workflow.yaml
brewslm pipeline dry-run --project 1 --workflow 5
brewslm pipeline run     --project 1 --workflow 5
brewslm pipeline history --project 1
```

## Export

```sh
brewslm export --project 1 --experiment 42 \
  --target vllm_server --format huggingface --smoke-test
```

## Doctor + logs

```sh
brewslm doctor --project 1
brewslm doctor --project 1 --deep                      # + timeline + clusters

brewslm logs tail      --project 1 --run-id exp-42
brewslm logs timeline  --project 1 --stage training --severity error --since 1d
brewslm logs clusters  --project 1
```

## Deploy

```sh
# Plan + smoke + promote
brewslm deploy plan       --project 1 --checkpoint 142 --target vllm_server
brewslm deploy smoke-test --deployment plan_8c9d --prompts 5
brewslm deploy promote    --deployment 17 --reason "phase A green smoke"
brewslm deploy reject     --deployment 17 --reason "drift > tolerance"
brewslm deploy rollback   --deployment 17 --target-version v2 --reason "..."

# Telemetry + drift + score
brewslm deploy telemetry push --deployment 17 --latency-ms 91 --status ok
brewslm deploy drift check    --deployment 17 --gold-set 5 --tolerance 0.02
brewslm deploy score          --deployment 17
```

## Support bundles

```sh
brewslm support-bundle create   --project 1 --download           # writes ./<uid>.zip
brewslm support-bundle list     --project 1
brewslm support-bundle download --bundle-uid abc... --token tok...
```

## Extensions

```sh
# Scaffold a plugin
brewslm scaffold adapter     --plugin-id my-adapter --description "..."
brewslm scaffold runtime     --plugin-id my-runtime
brewslm scaffold domain-pack --plugin-id my-pack
brewslm scaffold eval-pack   --plugin-id my-eval

# Manage extensions
brewslm extensions list
brewslm extensions validate --kind adapter --module my.adapter
brewslm extensions reload   --kind adapter            # omit --kind for all
```

## Auth (Theme 5 Epic 1)

```sh
# Exchange username + password for a JWT. Bare token on stdout for
# pipe-into-eval flows; --json for the full response; --save persists
# to ~/.brewslm/token (0600 on POSIX).
brewslm auth login --username admin --password "$PW"
brewslm auth login --username admin --save                # --password resolves via BREWSLM_PASSWORD env or prompt

# Inspect the principal + project memberships for the current token.
brewslm auth whoami
```

## Project Templates (Theme 5 Epic 1)

Cloneable starting kits (8 shipped). Each carries ~200 hand-curated
gold rows + a recipe snapshot + a recommended base model.

```sh
brewslm template list                                     # table view
brewslm template get ticket-router                        # JSON detail
brewslm template instantiate ticket-router --name "Acme"  # clone → new project
brewslm template instantiate log-triage                   # name defaults to template's display name
```

## Serve (Theme 5 Epic 1 — Video 10)

Start a serve template (Ollama / vLLM / llama.cpp) on a compressed
export. Wraps `/api/projects/{pid}/export/{eid}/serve-*`.

```sh
brewslm serve plan  --project 1 --export-id 12
brewslm serve start --project 1 --export-id 12 --template-id ollama_local
brewslm serve get   --project 1 --run-id srv-abc --logs-tail 200
brewslm serve stop  --project 1 --run-id srv-abc
```

## Version

```sh
brewslm version                                           # CLI version
brewslm version --remote                                  # + probes /api/health
brewslm --version                                         # short form
```

## Eval

```sh
brewslm eval generate    --project 1 --blueprint-id 7 --dataset-id 12
brewslm eval gold-set create --project 1 --dataset-type gold_dev \
  --seed-question "..." --seed-answer "..."
brewslm eval gold-set sample --project 1 --strategy stratified --count 100
brewslm eval gold-set submit --project 1 --version 1
brewslm eval label       --project 1 --row 42 --answer "..."
brewslm eval run         --project 1 --experiment 42 --pack evalpack.qa.strict
brewslm eval compare     --project 1 --experiment-a 42 --experiment-b 43
brewslm eval clusters    --project 1
brewslm eval remediate   --project 1 --eval-result 17
```

## Exit codes

The CLI follows shell convention:

| Code | Means |
|---|---|
| `0` | Success. |
| `1` | Operational failure (API returned non-2xx OR a check came back red). |
| `2` | Usage error (bad flag, unknown kind, etc.). |

`extensions validate` exits `1` on contract failure; `extensions reload` exits `1` if any kind came back `partial` or `error` (but `0` for `not_supported`). `deploy score` exits `0` / `1` / `2` for `ready` / `caution` / `block` so you can wire it as a CD gate.

## Tips

- Save command snippets per project for reproducibility.
- Prefer explicit `--project N` over implicit context flags.
- Pipe `--json` output through `jq` for filtering: `brewslm extensions list | jq '.kinds[].kind'`.
- For CI, set `BREWSLM_TOKEN` once in the environment and skip `--token` on every call.

## See also

- [API surface](api-surface.md) — equivalent HTTP endpoints.
- [Glossary](glossary.md) — every term the CLI prints.
- [Full auto-generated reference](cli-full.md) — every flag for every subcommand.

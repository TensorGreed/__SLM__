---
sidebar_position: 1
title: Plan a deployment
---

# Plan a deployment

A **deployment plan** answers: *given this trained checkpoint + this target environment, will it work?* Plan first, smoke test second, promote third. Reverse the order and you'll find out the hard way when a tokenizer doesn't fit in 8 GB of mobile RAM.

## What goes into a plan

| Input | Where from |
|---|---|
| Source checkpoint | A row in `checkpoints` (see [Training](../workflows/training.md)). |
| Target profile | `mobile_cpu` / `browser_webgpu` / `edge_gpu` / `vllm_server` / custom. |
| Compression choice | None / GGUF-Q4 / GGUF-Q8 / ONNX-INT8. |
| Export bundle layout | `gguf+tokenizer.json` / `onnx+vocab.txt` / `huggingface`. |
| Smoke prompts | Tiny gold-set sampled from your eval pack. |

A plan is **not** a deployment yet. It's a structured pre-flight that surfaces blockers before you commit resources.

## Generate a plan

### UI

1. Open **Deployments** under the Training rail.
2. Click **New deployment**.
3. Pick a source checkpoint and a target profile from the dropdowns.
4. Review the auto-generated plan:
   - **Readiness rows** — green / amber / red per check (artifacts present, target dependencies installed, file sizes within target budget).
   - **Smoke prompt count** — defaults to 5 from your active eval pack.
   - **Compression suggestion** — what the [Deployability score](rollback-and-score.md) recommends for this target.
5. Click **Continue** to advance to the smoke step.

### CLI

```sh
brewslm deploy plan \
  --project 7 \
  --checkpoint 142 \
  --target vllm_server
```

Prints the full plan as JSON. Pipe to `jq` if you only want one section:

```sh
brewslm deploy plan ... | jq '.readiness'
```

Exit code is non-zero if any **required** readiness check fails — useful as a CI gate.

### API

```sh
curl -X POST http://localhost:8000/api/projects/7/deployments/plan \
  -H "Content-Type: application/json" \
  -d '{
    "checkpoint_id": 142,
    "target_profile": "vllm_server"
  }'
```

Returns:

```json
{
  "plan_id": "plan_8c9d…",
  "checkpoint_id": 142,
  "target_profile": "vllm_server",
  "readiness": [
    {"name": "checkpoint_artifact",     "ok": true, "message": "..."},
    {"name": "tokenizer_artifact",      "ok": true, "message": "..."},
    {"name": "target_runtime",          "ok": true, "message": "vllm v0.6.3 detected"},
    {"name": "weight_size_within_budget","ok": true, "message": "1.2 GB ≤ 8 GB budget"}
  ],
  "suggested_compression": null,
  "smoke_prompts": [...]
}
```

## Target profiles

The four built-in profiles capture different deploy constraints:

| Profile | Where it lands | Key constraints |
|---|---|---|
| `mobile_cpu` | iOS / Android via CoreML or ONNX | ≤ 1.5 GB weights, INT8, no GPU. |
| `browser_webgpu` | In-tab via Transformers.js | ≤ 800 MB weights, GGUF-Q4 favored. |
| `edge_gpu` | NVIDIA Jetson / RK3588 | ≤ 4 GB weights, ONNX-INT8 preferred. |
| `vllm_server` | Server-side vLLM | No weight budget; FP16 OK. |

Custom profiles can be registered as a plugin — see [Extensions → Contracts](../extensions/contracts.md) (the target profile plugin kind).

## Plan failures + what to do

| Readiness check fails | Fix |
|---|---|
| `checkpoint_artifact` | Make sure training finished — Compression and Export read the saved checkpoint dir. |
| `tokenizer_artifact` | Re-run tokenization stage; the tokenizer JSON has to live next to the checkpoint. |
| `target_runtime` | Install the target's runtime (`pip install vllm`, brew install onnxruntime, etc.). |
| `weight_size_within_budget` | Compression suggestion will appear — accept it on the next screen. |
| `eval_pack_present` | Pick or create an [eval pack](../workflows/evaluation-and-remediation.md) before planning. |

Most blockers are **actionable** — the readiness row's `message` tells you the exact next step.

## Next

- [Smoke + promote](smoke-and-promote.md) — run smoke prompts, promote on green.
- [Rollback + score](rollback-and-score.md) — what happens after promote.
- [Telemetry](telemetry.md) — measuring the live deployment.

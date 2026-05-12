---
sidebar_position: 6
title: Export + deployment
---

# Export + deployment

Stages 10 (compression) and 11 (export) of the [pipeline](pipeline-overview.md). This page covers the artifact production side; once you have an export, the [Deployment](../deployment/plan.md) section handles plan → smoke → promote → telemetry.

## Compression

Optional. Quantises / prunes the trained checkpoint to fit a target's weight budget.

| Method | What it does | Use it for |
|---|---|---|
| GGUF-Q4 | 4-bit weights via `llama.cpp` quantiser. | Browser, mobile, edge GPU. |
| GGUF-Q8 | 8-bit weights, half memory of FP16. | Edge GPU, low-RAM server. |
| ONNX-INT8 | INT8 quantisation via ONNX Runtime. | Mobile, edge, browser. |
| Pruning | Structured pruning (experimental). | Specialised research / latency. |

### UI

Pipeline → **Compression** → **New compression**. Pick the source checkpoint, method, and quality vs size knob. The job is queued; status pings the timeline.

### CLI

```sh
brewslm compress --project 1 --experiment 42 \
  --method gguf_q4 --output-name "q4-v1"
```

### API

```sh
curl -X POST http://localhost:8000/api/projects/1/compression/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "experiment_id": 42,
    "method": "gguf_q4",
    "output_name": "q4-v1"
  }'
```

### Reason codes you might hit

| Code | Means |
|---|---|
| `export_quantization_failed` | Quantiser exited non-zero. Logs in the job result; common cause is unsupported activation type. |
| `export_artifact_missing` | Source checkpoint isn't where compression expects it (e.g. promoted to a non-existent step). |

## Export

Builds the **deployable artifact bundle** for a target. The export job wraps weights + tokenizer + manifest into the layout the target runtime expects.

### Target → output bundle layout

| Target | Bundle contents |
|---|---|
| `vllm_server` | `huggingface/` directory (config.json, tokenizer.json, safetensors). |
| `mobile_cpu` | `coreml/` or `onnx_int8/` plus a tiny inference shim. |
| `browser_webgpu` | `gguf_q4/` + `transformers-js` config. |
| `edge_gpu` | `onnx/` + tokenizer + a deployment README. |

### UI

Pipeline → **Export** → **New export**. Pick:

- **Source** — experiment + checkpoint (or compressed artifact from above).
- **Target profile** — see table above.
- **Format** — defaults to the target's canonical format.
- **Smoke checks** — run a sample of 5 prompts against the exported artifact before declaring success.

Click **Start export**. Artifact lands under `DATA_DIR/exports/project-{id}/export-{id}/`.

### CLI

```sh
# Server export
brewslm export --project 1 --experiment 42 \
  --target vllm_server --format huggingface

# Mobile export
brewslm export --project 1 --experiment 42 \
  --target mobile_cpu --format onnx_int8

# With smoke checks
brewslm export --project 1 --experiment 42 \
  --target vllm_server --format huggingface --smoke-test
```

### API

```sh
curl -X POST http://localhost:8000/api/projects/1/export \
  -H "Content-Type: application/json" \
  -d '{
    "experiment_id": 42,
    "target_profile": "vllm_server",
    "format": "huggingface",
    "smoke_test": true
  }'
```

Returns:

```json
{
  "export_id": 19,
  "status": "running",
  "artifact_key": "export.vllm_server.42",
  "output_dir": "/data/exports/project-1/export-19/",
  "smoke_test_planned": true
}
```

Once done, the artifact contains:

```
exports/project-1/export-19/
├── huggingface/                     # the actual artifact
│   ├── config.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── model.safetensors
├── manifest.json                    # source experiment + every transform applied
├── smoke_results.json               # if --smoke-test
└── README.md                        # deployment instructions for this target
```

### Reason codes you might hit

| Code | Means |
|---|---|
| `export_run_failed` | Generic export failure — read the timeline. |
| `export_artifact_missing` | Source checkpoint / tokenizer missing. |
| `export_quantization_failed` | Mid-export quantisation step failed (when target requires INT8/Q4). |

## Optimization recommendations

Before exporting, the Optimize panel (Pipeline → Export → **Recommend**) ranks candidate target configurations by latency / memory / quality tradeoff. The result is labelled with **provenance**:

| Provenance | Means |
|---|---|
| `measured` | Real benchmark ran on the target. Highest trust. |
| `estimated` | Heuristic estimate from model + target metadata. |
| `mixed` | Some metrics measured, some estimated. |

Always read provenance before committing — see [Measured vs estimated](../reliability/measured-vs-estimated.md).

## Pre-deploy checklist

Before moving on to [plan + smoke + promote](../deployment/plan.md):

- [ ] Export `status=succeeded`.
- [ ] `smoke_results.json` shows pass rate ≥ your eval pack's gate threshold.
- [ ] `manifest.json` lists every transform applied (no surprise quantisation steps).
- [ ] README in the bundle has instructions that match your serving env.
- [ ] [Deployability score](../deployment/rollback-and-score.md) returns `ready` (or `caution` with a justification).

Don't skip the checklist even when in a hurry — every box maps to a class of post-deploy incident.

## Mobile + browser bundles

The export job for `mobile_cpu` / `browser_webgpu` produces a **runnable reference bundle** (CoreML for iOS, ONNX for Android, transformers.js for browser) with:

- Deterministic directory structure.
- README with smoke-test instructions.
- One-line "load + infer once" entrypoint.

Useful for handing the artifact to a mobile or web team without 12 follow-up questions about file layout.

## Final advice

**Deploy the smallest model that meets your quality bar.** Smaller is cheaper, faster to iterate, easier to host. If a 1B model passes your eval gates, ship that — even if a 7B model passes them with more headroom. Headroom you don't need is overhead.

## Next

- [Deployment → Plan](../deployment/plan.md) — what to do with the export.
- [Deployment → Smoke + promote](../deployment/smoke-and-promote.md) — the safety loop.
- [Measured vs estimated](../reliability/measured-vs-estimated.md) — provenance on optimisation metrics.

---
sidebar_position: 3
title: Post-deploy telemetry
---

# Post-deploy telemetry

Once a deployment is `active`, BrewSLM collects inference metrics so you can answer: *is it serving well right now?* Telemetry is push or pull, computed over a rolling window, and surfaced in the [Run Timeline](../observability/timeline.md), the Deployments page, and the [Deployability score](rollback-and-score.md).

## What's measured

| Metric | What it is | Where used |
|---|---|---|
| `latency_p50_ms` / `p95_ms` / `p99_ms` | Round-trip ms per request, computed from the rolling window. | Deployment health, score, drift. |
| `error_rate` | Fraction of requests with `status != ok`. | Score, alerts. |
| `request_volume` | Requests per minute. | Telemetry chart, scaling decisions. |
| `token_throughput` | Output tokens / second (sum). | Cost estimation, capacity. |
| `time_to_first_token_ms` | TTFT for streaming responses. | Score, drift. |
| `prompt_tokens` / `completion_tokens` | Mean per request. | Cost / billing. |

All metrics are computed from raw **inference samples** stored in `deployment_telemetry_samples`. The service rolls them into the **windowed view** (`/telemetry`) on read.

## How samples get in

Three paths:

1. **Push from your serving env** — POST one sample per inference to `/api/deployments/{id}/telemetry/samples`. This is the recommended path for vLLM / Ollama / your custom server.
2. **Pull from a target probe** — `served_model_telemetry_service` can periodically poll the target's metrics endpoint (e.g. vLLM's Prometheus) if you configure one.
3. **Synthetic for tests** — phase75 + phase76 backend tests seed samples directly. Useful for chart regression tests but not real ops.

Sample shape:

```json
{
  "deployment_id": 17,
  "ts": "2026-05-12T11:23:45Z",
  "status": "ok",
  "latency_ms": 91,
  "prompt_tokens": 56,
  "completion_tokens": 128,
  "time_to_first_token_ms": 32,
  "actor": "vllm-serv-01"
}
```

## Push a sample

### UI

The UI doesn't push samples directly — your serving runtime does. The Deployments detail page **reads** the rolling window every 10 seconds while open.

### CLI

```sh
brewslm deploy telemetry push \
  --deployment 17 \
  --latency-ms 91 \
  --status ok \
  --prompt-tokens 56 \
  --completion-tokens 128
```

Convenient for hand-testing the pipeline.

### API

Push:

```sh
curl -X POST http://localhost:8000/api/deployments/17/telemetry/samples \
  -H "Content-Type: application/json" \
  -d '{"latency_ms": 91, "status": "ok", "prompt_tokens": 56, "completion_tokens": 128, "time_to_first_token_ms": 32}'
```

Read the rolling window:

```sh
curl "http://localhost:8000/api/deployments/17/telemetry?window=5m"
```

Returns:

```json
{
  "deployment_id": 17,
  "window": "5m",
  "samples": 1283,
  "latency_p50_ms": 91,
  "latency_p95_ms": 142,
  "latency_p99_ms": 218,
  "error_rate": 0.004,
  "request_volume_per_min": 256.6,
  "token_throughput_per_s": 142.1,
  "ttft_p50_ms": 32,
  "first_sample_at": "2026-05-12T11:18:45Z",
  "last_sample_at":  "2026-05-12T11:23:45Z"
}
```

## Wiring vLLM

A common production setup is a small sidecar that scrapes vLLM's metrics and forwards them. Skeleton:

```python
import httpx, time
from prometheus_client.parser import text_string_to_metric_families

while True:
    metrics_text = httpx.get("http://vllm:8000/metrics").text
    for fam in text_string_to_metric_families(metrics_text):
        if fam.name == "vllm:request_latency":
            for s in fam.samples:
                httpx.post(
                    "http://brewslm:8000/api/deployments/17/telemetry/samples",
                    json={
                        "latency_ms": int(s.value * 1000),
                        "status": "ok",
                    },
                )
    time.sleep(10)
```

Real production-grade pollers belong in your infra repo, not BrewSLM — but the surface is identical.

## Telemetry in the Run Timeline

Every minute of sustained traffic emits a heartbeat RunEvent (`stage=deployment, severity=info`). Sustained error rates above the threshold emit `severity=error, reason_code=deployment_smoke_failed` or `deployment_drift_detected`. So the [timeline](../observability/timeline.md) is the right place to look back at "how was this deployment behaving on Tuesday afternoon?"

## Next

- [Drift checks](drift-checks.md) — gold eval against the live endpoint.
- [Rollback + score](rollback-and-score.md) — deployability score derived from telemetry.
- [Run Timeline](../observability/timeline.md) — see the same telemetry in context.

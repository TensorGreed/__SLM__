---
sidebar_position: 1
title: Measured vs estimated
---

# Measured vs estimated

A core BrewSLM principle: **every number is labelled with where it came from**. If the system can produce a real measurement, it does. If it has to fall back to a heuristic, it says so — explicitly, with the reason.

This page is the cheat sheet for reading the `provenance` field that shows up across cost estimates, optimisation cards, deployability scores, drift checks, and capability matrices.

## The three values

| Provenance | What it means | Trust level |
|---|---|---|
| **`measured`** | Came from real observation — a completed run, a smoke test, a telemetry window, a benchmark execution. | Highest. Suitable for promote / rollback decisions. |
| **`estimated`** | Heuristic computed from metadata + history. Cohort match (model size / mode), historical median, hardware spec. | Directional. Use for planning, not for promotion. |
| **`mixed`** | Some components measured, some estimated. The breakdown shows which. | Use the per-component table to decide. |

The word "simulated" sometimes appears too — that's a special case of `measured` where the run was real but the runtime was the built-in `simulate` backend (synthetic telemetry). UI surfaces this as `measured (simulate)` so you know not to compare to production numbers.

## Where you'll see it

### Cost estimator card

Every training launch shows: `gpu_hours`, `usd`, `co2_kg`, plus `provenance` + a `confidence` band.

```json
{
  "gpu_hours": 1.42,
  "usd": 0.71,
  "co2_kg": 0.17,
  "provenance": "measured",
  "confidence": 0.82,
  "confidence_band": "high",
  "calibration": {
    "cohort": "mode+model_size",
    "sample_count": 8,
    "variability_cv": 0.04,
    "fallback_used": false
  }
}
```

Reading rules:

- `provenance="measured"` requires **both** `sample_count >= 2` **and** `confidence >= 0.60`. Otherwise it gets demoted to `estimated` even if there's history.
- `cohort` ladder, tightest to loosest: `mode+model_size` → `mode` → `model_size` → `global` → `none`.
- `confidence_band`: `high` ≥ 0.8, `medium` ≥ 0.6, `low` otherwise.

### Deployment readiness

Each readiness row from [Plan a deployment](../deployment/plan.md) carries a provenance:

```json
{"name": "weight_size_within_budget", "ok": true, "provenance": "measured", "message": "1.2 GB ≤ 8 GB budget"}
{"name": "target_runtime",           "ok": true, "provenance": "estimated", "message": "vllm v0.6.3 detected (estimated from `pip list`)"}
```

If a check is `estimated`, it didn't actually test the live target — only inferred. Most readiness checks should be `measured`; if any are `estimated`, the [Deployability score](../deployment/rollback-and-score.md) caps at `0.6`.

### Optimisation candidates

When you export to a target, the optimisation panel ranks candidate configurations by latency × memory × quality. Each candidate's row labels every metric:

```
candidate-3
  latency_p50_ms:   91   measured  (real run on target)
  memory_mb:       820   measured
  quality_delta:  -0.02  estimated (no eval run on quantised weights)
```

A candidate with **everything measured** beats one with the same numbers but `estimated` quality — use the provenance column when ranking ties.

### Capability matrix

The Base Model Registry's compatibility scoring works the same way:

| Field | Measured when | Estimated when |
|---|---|---|
| Tokenizer loads | We tried it. | We checked the metadata only. |
| Context length supported | We exercised it. | Read from `config.json`. |
| Adapter compatibility | We ran the adapter against 5 sample rows. | Field-name heuristic. |

## Strict mode

[Autopilot strict mode](../workflows/newbie-autopilot.md) refuses to use any `estimated` value as the basis for a launch. Reach for it when:

- A teammate is reviewing your run end-to-end.
- A CI gate must not silently take a fallback.
- You're recreating a documented incident.

In strict mode, an `estimated` cost estimate triggers a blocker: *"Cost is estimated (cohort=mode, sample_count=1). Strict mode refuses launch. To resolve, run a calibration job OR disable strict."*

## When values are estimated, do this

| You see | Action |
|---|---|
| Cost = `estimated` | Run one tiny calibration experiment (1 epoch, small batch) — that one measured point unlocks `measured` cost for similar configs. |
| Readiness row = `estimated` | Run the corresponding live check: e.g. invoke the target runtime once to confirm presence. |
| Optimisation candidate = `estimated quality` | Run a smoke test with `--prompt-count=20` on the candidate. |
| Capability matrix = `estimated tokenizer` | Click **Validate for project** on the model — that runs the real check. |

The pattern: **one cheap measurement promotes the provenance** for all similar future decisions.

## Why the distinction matters

The old habit is: "training reports 1.4 GPU-hours, I'll budget for that." With BrewSLM, you read: "training reports 1.4 GPU-hours, **estimated**, cohort=mode, confidence=0.42". That tells you the number is the best-available guess but you shouldn't bet on it. A real measurement is one experiment away — and once you have it, the estimator switches to `measured` automatically for the next similar config.

This makes regressions discoverable. If yesterday a job was `measured, 1.4 GPU-hours, confidence=0.85` and today the same config reports `2.1 GPU-hours, confidence=0.31`, something drifted (data size, hardware, runtime version) and the confidence drop tells you so before the cost surprise.

## Next

- [Common blockers](common-blockers.md) — what to do when an estimate becomes a blocker.
- [Cost estimator details](../workflows/training.md) — under the training stage.
- [Deployability score](../deployment/rollback-and-score.md) — provenance × weight per component.

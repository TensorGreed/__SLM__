---
sidebar_position: 4
title: Drift checks
---

# Drift checks

Telemetry tells you *how* the deployment is serving. A drift check tells you *whether the answers are still right.* It re-runs your gold-set eval against the **live endpoint** (not the offline checkpoint) and compares to the baseline pass rate captured at promote time.

## Why drift differs from smoke

| Smoke | Drift |
|---|---|
| Runs once, **before** promote. | Runs on-demand or scheduled, **after** promote. |
| Few prompts (default 5). | Full gold set (or a configurable slice). |
| Catches obvious blockers (tokenizer mismatch, OOM). | Catches subtle regressions (model file replaced, dataset poisoned upstream, runtime upgrade changed numerics). |
| Doesn't check the live endpoint. | Calls the live endpoint exactly as a user would. |

A typical workflow runs drift weekly + after any infra change. Drift is also cheap enough to wire into a CD gate before any traffic ramp.

## Run a drift check

### UI

Deployments detail → **Drift check** tab → **Run check now**. A modal asks which gold set + how many prompts (default: all). Click **Start**.

Results stream in row by row. The summary card at top shows:

- Current pass rate.
- Baseline pass rate (snapshot at promote).
- Delta + tolerance — green if `delta >= -tolerance`, red if not.
- Per-stage / per-reason-code breakdown of failures (links straight to the [failure cluster](../observability/failure-clusters.md)).

### CLI

```sh
brewslm deploy drift check \
  --deployment 17 \
  --gold-set 5 \
  --tolerance 0.02
```

Exit non-zero if `pass_rate_delta < -tolerance`. Wire to your CD as a gate.

### API

```sh
curl -X POST http://localhost:8000/api/deployments/17/drift/check \
  -H "Content-Type: application/json" \
  -d '{
    "gold_set_id": 5,
    "tolerance": 0.02,
    "prompt_count": null
  }'
```

Returns:

```json
{
  "deployment_id": 17,
  "gold_set_id": 5,
  "check_id": "drift_44a1…",
  "baseline_pass_rate": 0.94,
  "current_pass_rate": 0.91,
  "delta": -0.03,
  "tolerance": 0.02,
  "verdict": "drift_detected",
  "failures_by_reason": {
    "eval_runtime_error": 4,
    "answer_off_by_format": 12
  },
  "ran_prompts": 100,
  "duration_ms": 14823
}
```

Verdict is one of `passing` / `borderline` (within tolerance but worse than baseline) / `drift_detected`.

## What "baseline" means

At promote time, BrewSLM snapshots the smoke test pass rate as the **deployment baseline**. Drift compares to that snapshot. If you'd rather compare to a freshly-rerun eval (more rigorous, more expensive), pass `--baseline=fresh` in the CLI / `"baseline_mode": "fresh"` in the API.

## Drift as a RunEvent

Drift detection emits `stage=deployment, severity=error, reason_code=deployment_drift_detected`. This means:

- The drift event surfaces in the [timeline](../observability/timeline.md).
- It folds into the [failure cluster](../observability/failure-clusters.md) view next to other deployment errors.
- Support bundles (next page after rollback) include the last N drift checks.

## Auto-schedule

A drift check on a cron is just a CLI call from your scheduler:

```sh
# crontab -e
# Run drift every Monday at 06:00, fail loud if pass rate dropped >2%.
0 6 * * 1 brewslm deploy drift check --deployment 17 --tolerance 0.02 --json | tee -a /var/log/brewslm-drift.log
```

For richer scheduling (multiple deployments, paged alerts), use whatever your team already runs (Airflow, Argo, GitHub Actions). The CLI's `--json` flag emits one JSON object per run so it parses cleanly.

## Next

- [Rollback + score](rollback-and-score.md) — what to do when drift hits.
- [Failure clusters](../observability/failure-clusters.md) — drill into *why* answers drifted.
- [Telemetry](telemetry.md) — the operational side of "still serving well".

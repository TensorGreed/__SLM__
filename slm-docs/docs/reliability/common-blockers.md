---
sidebar_position: 10
title: Common Blockers and Fixes
---

# Common Blockers and Fixes

Use this page when a run fails before or during execution.

## 1) VRAM Incompatibility Blocker

**Symptom**: training start is blocked for model-target mismatch.

**Why**: estimated model VRAM clearly exceeds target baseline.

**Fixes**:

- pick smaller base model,
- switch target profile to higher-memory target,
- use stronger quantization/export profile for inference scenarios.

## 2) Invalid Env Values at Startup

**Symptom**: backend does not start or behaves unexpectedly with malformed env values.

**Fixes**:

- align env with expected types,
- use boolean values like `true/false` for flags,
- prefer `.env.example` as source of truth.

## 3) Frontend `ENOSPC` Watcher Limit

**Symptom**: `npm run dev` fails with file watcher limit error.

**Fix** (Linux):

```bash
sudo sysctl fs.inotify.max_user_watches=524288
sudo sysctl -p
```

## 4) Plugin Load Errors

**Symptom**: target/model/starter-pack plugin module not loaded.

**Fixes**:

- confirm module path in env,
- ensure plugin exports expected registration function or payload,
- inspect plugin load errors in catalog metadata.

## 5) Estimated Metrics Never Become Measured

**Symptom**: optimizer always returns estimated values.

**Fixes**:

- verify benchmark command/runtime dependencies,
- verify artifact existence,
- retry with strict mode to surface exact blocker.

## Debugging Rule of Thumb

When blocked, capture:

- exact endpoint/action,
- full error payload,
- model + target profile,
- runtime backend and relevant env.

That context usually reduces triage from hours to minutes.

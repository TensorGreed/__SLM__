# BrewSLM Overview

## What BrewSLM Appears To Be From Repo Evidence

Evidence:
- `README.md` describes a FastAPI backend plus React frontend for SLM lifecycle operations.
- `frontend/src/pages/ProjectPipelinePage.tsx` shows a pipeline UI with data, cleaning, gold set, synthetic, dataset prep, tokenization, training, evaluation, compression, and export tabs.
- `backend/app/main.py` mounts APIs for ingestion, cleaning, dataset prep, gold, synthetic, training, evaluation, compression, export, registry, deployments, observability, and demo projects.

BrewSLM appears to be an end-to-end SLM lifecycle application, with project workspaces and guided pipeline stages.

## What Needs To Be Verified From Code

- Which paths complete on a fresh local machine.
- Which paths require Redis/Celery.
- Which paths require external model APIs, GPUs, or external CLI tools.
- Exact export formats and deployment commands.
- How final model usage is wired after a successful training run.

## Main Product Story

Status: partial.

The repo supports a story of moving from sample or imported data to a prepared dataset, training run, evaluation, compression/export, registry, and deployment/usage. The official samples provide beginner-friendly entry points.

## Beginner Demo Angle

Start from a demo tile, show that the project is pre-loaded with source data and a gold set, then walk the pipeline tabs at a conceptual pace.

## Technical Demo Angle

Show the manifest, schema, adapter mapping, prepared manifest, split files, training config, evaluation packs, and runtime caveats.

## Operator/Admin Angle

Show auth, secrets, runtime settings, registry promotion, deployment telemetry, and support bundles only after those paths are verified in the UI.


---
sidebar_position: 4
title: Pipeline Overview
---

# Pipeline Overview

BrewSLM is organized as a stage-based pipeline.

## Stages at a Glance

1. **Ingestion**: bring raw domain data into the project.
2. **Cleaning**: normalize text quality and structure.
3. **Gold/Eval Prep**: define trustworthy validation examples.
4. **Dataset Prep / Adapter**: map your data into task-specific format.
5. **Training**: run fine-tuning with compatibility checks.
6. **Evaluation**: score quality, safety, and regression risk.
7. **Compression**: quantize / package for target runtime.
8. **Export**: produce deployable artifacts and manifests.

## How to Work as a Beginner

- Do one stage at a time.
- Keep each run small and measurable.
- Prefer defaults first, tune later.
- Never skip evaluation before export.

## Iteration Loop

Use this loop for every project:

1. Build small baseline.
2. Evaluate.
3. Analyze failures.
4. Apply remediation (data + training config).
5. Re-run and compare.

This loop beats one giant training run with no feedback.

## Artifacts and Reproducibility

BrewSLM persists manifests and metadata so you can answer:

- what data was used,
- what config was applied,
- which metrics were measured vs estimated,
- why a recommendation was made.

Treat every experiment as a reproducible record, not a one-off trial.

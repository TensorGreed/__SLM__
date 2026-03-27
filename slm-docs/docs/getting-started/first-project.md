---
sidebar_position: 3
title: Build Your First Project
---

# Build Your First Project

This walkthrough helps you get a first model iteration, even with a small dataset.

## Step 1: Create Project

From the dashboard:

1. Click **New Project**.
2. Enter a clear name (for example, `Support FAQ Assistant`).
3. Optionally choose a **Starter Pack** (Legal, Support, Healthcare, Finance).

Starter packs prefill sane defaults for:

- model family suggestions
- adapter/task defaults
- evaluation gates
- safety reminders
- target profile defaults

## Step 2: Ingest Data

Start small. Aim for 100 to 500 high-quality examples first.

Good starter dataset patterns:

- question -> answer pairs
- instruction -> response pairs
- chunk + expected extraction JSON

## Step 3: Run Training Wizard / Autopilot

Use wizard defaults for your first run.

If blocked, read the blocker message carefully. BrewSLM usually gives:

- exact reason
- severity (warning vs blocker)
- concrete fixes

## Step 4: Evaluate

Run evaluation before exporting.

Look at:

- pass rate
- hallucination/safety metrics
- failure clusters

Then apply remediation suggestions (data + config fixes), retrain, and re-evaluate.

## Step 5: Export and Smoke Test

Export for one target first (for example, edge GPU or server).

Validate:

- artifact exists and loads
- latency/memory are acceptable
- quality is close to evaluation behavior

## Definition of Done (First Iteration)

- You can run one complete project from ingestion to export.
- You can explain at least one failure mode and one fix.
- You can reproduce your result with the same project config.

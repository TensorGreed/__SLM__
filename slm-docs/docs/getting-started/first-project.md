---
sidebar_position: 3
title: Build Your First Project
---

# Build Your First Project

This walkthrough helps you get a first model iteration, even with a small dataset.

## Step 1: Create Project

From the dashboard:

1. Click **New Project**.
2. Keep **Beginner Mode** enabled.
3. Enter a clear name (for example, `Support FAQ Assistant`).
4. Write a plain-language domain brief describing:
   - your problem statement,
   - who will use outputs,
   - sample inputs/outputs,
   - safety/compliance constraints,
   - target deployment.
5. Review **What the system understood** (task family, output contract, assumptions, glossary), then create.

Starter packs prefill sane defaults for:

- model family suggestions
- adapter/task defaults
- evaluation gates
- safety reminders
- target profile defaults

Beginner Mode lets you start from domain intent first; advanced pack/profile fields can be refined later.

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

## Sample Briefs and Resulting Blueprints

### Sample Brief 1: Support Q&A

Brief:

> Build a support assistant that answers customer FAQ questions from resolved tickets.  
> It should be concise, avoid hallucinations, and run on an edge GPU deployment.

Typical resulting blueprint fields:

- `task_family`: `qa`
- `input_modality`: `text`
- `deployment_target_constraints.target_profile_id`: `edge_gpu`
- `expected_output_schema`: `{"type":"object","properties":{"answer":"string"},"required":["answer"]}`
- `success_metrics`: `answer_correctness`, `hallucination_rate`

### Sample Brief 2: Contract Extraction

Brief:

> Extract liability and indemnification clauses from legal contracts into JSON.  
> This is for legal analysts, and output must always be valid structured JSON.

Typical resulting blueprint fields:

- `task_family`: `structured_extraction`
- `input_modality`: `text`
- `expected_output_schema`: object with required extraction fields
- `safety_compliance_notes`: legal-policy and quality guardrails
- `success_metrics`: `exact_match`, `json_valid_rate`

---
sidebar_position: 5
title: Data Ingestion and Adapter Setup
---

# Data Ingestion and Adapter Setup

Quality data is the highest-leverage part of your pipeline.

## What Good Input Looks Like

- Consistent field names
- Minimal duplicates
- Clean task boundaries (instruction vs output)
- Known source quality and sensitivity

## Ingestion Sources

BrewSLM can ingest from local files and external sources supported by your runtime setup.

Start with small, local files first so your first run is debuggable.

## Adapter/Task Defaults

Adapter presets map raw data into training-ready schema. For novices:

- pick one adapter,
- keep task profile stable,
- avoid custom field mapping until baseline works.

Starter packs can prefill adapter/task defaults for your domain.

## Data Quality Checklist

Before training, verify:

1. No empty prompts/targets
2. No malformed JSONL rows
3. Labels/instructions are internally consistent
4. Train/validation splits are non-overlapping
5. Sensitive content is handled according to policy

## Common Mistakes

- Training on noisy scraped text without filtering
- Mixing incompatible task formats in one dataset
- No validation split
- Ignoring class/intent imbalance

Fixing these usually improves quality more than hyperparameter tweaks.

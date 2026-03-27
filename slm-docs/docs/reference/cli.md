---
sidebar_position: 11
title: CLI Quick Reference
---

# CLI Quick Reference

BrewSLM includes a CLI wrapper (`./brewslm`) for common operations.

## Training

```bash
./brewslm train \
  --project 1 \
  --autopilot \
  --one-click \
  --intent "Build a legal Q&A assistant" \
  --base-model Qwen/Qwen2.5-1.5B-Instruct
```

## Export

```bash
./brewslm export --project 1 --format huggingface --target vllm
```

## Useful Tips

- Run the same command with small data first.
- Save command snippets per project for reproducibility.
- Prefer explicit project IDs over ad-hoc filtering.

## When to Prefer CLI

Use CLI when:

- automating repeated runs,
- integrating with scripts/CI,
- sharing exact reproducible commands with teammates.

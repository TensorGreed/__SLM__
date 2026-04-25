---
sidebar_position: 11
title: CLI Quick Reference
---

# CLI Quick Reference

BrewSLM includes a CLI wrapper (`./brewslm`) for common operations.

## Training

```bash
./brewslm train start \
  --project 1 \
  --autopilot \
  --one-click \
  --intent "Build a legal Q&A assistant" \
  --base-model Qwen/Qwen2.5-1.5B-Instruct
```

### Manage runs

```bash
./brewslm train rerun --project 1 --experiment-id 5 --run-name "rerun of phi-2"
./brewslm train clone --project 1 --experiment-id 5 \
    --config-overrides '{"learning_rate": 3e-4, "num_epochs": 5}'
./brewslm train pause --project 1 --experiment-id 5
./brewslm train resume --project 1 --experiment-id 5
./brewslm train checkpoints --project 1 --experiment-id 5
./brewslm train checkpoints --project 1 --experiment-id 5 --promote-step 200
./brewslm train checkpoints --project 1 --experiment-id 5 --resume-from-step 150
```

### Reproducibility manifest

```bash
./brewslm repro manifest --project 1 --experiment-id 5
```

## Export

```bash
./brewslm export --project 1 --format huggingface --target vllm
```

## Beginner Bootstrap

```bash
./brewslm project bootstrap \
  --name "Support FAQ Assistant" \
  --brief "Build a support assistant that answers FAQ questions from ticket history." \
  --sample-input "How do I reset my password?" \
  --sample-output '{"answer":"Use the account reset flow."}' \
  --target edge_gpu \
  --create-project
```

## Blueprint Inspect

```bash
./brewslm project blueprint show --project 1 --latest
./brewslm project blueprint diff --project 1 --from-version 1 --to-version 2
```

## Universal Model Registry

```bash
./brewslm models import --hf-id "Qwen/Qwen2.5-1.5B-Instruct" --json
./brewslm models refresh --model 1 --json
./brewslm models list --family qwen --hardware-fit laptop --json
./brewslm models recommend --project 1 --limit 5 --hardware-fit server --json
./brewslm models validate --project 1 --model 1 --json
```

## Dataset Structure Explorer and Adapter Studio

```bash
./brewslm dataset profile --project 1 --source-type csv --source-ref ./data/train.csv --json
./brewslm adapter infer --project 1 --source-type jsonl --source-ref ./data/chat.jsonl --json
./brewslm adapter preview --project 1 --source-type jsonl --source-ref ./data/chat.jsonl --adapter-id auto --json
./brewslm adapter validate --project 1 --source-type jsonl --source-ref ./data/chat.jsonl --adapter-id auto --json
./brewslm adapter export --project 1 --adapter-name support_adapter --version 2 --json
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

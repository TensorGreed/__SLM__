# BrewSLM CLI Quickstart Guide

Welcome to BrewSLM! This guide will help you launch your first SLM optimization pipeline in minutes.

## 1. Installation
Ensure you have the BrewSLM CLI installed:
```bash
# From the project root
pip install -e .
```

## 2. Project Setup
Create a new project using one of the quickstart templates:
```bash
brewslm project create --name "MyFirstSLM" --template "general"
```

## 3. Data Ingestion
Import a sample dataset with one click:
```bash
brewslm dataset import --project-id 1 --sample "support-chat-v1"
```

## 4. Run Autopilot Training
Launch a balanced training run on the cloud:
```bash
brewslm train --project-id 1 --autopilot --one-click --intent "Fine-tune a practical assistant on my imported dataset."
# Optional explicit base model override:
brewslm train --project-id 1 --autopilot --one-click --intent "Fine-tune a practical assistant on my imported dataset." --base-model "Qwen/Qwen2.5-1.5B-Instruct"
```

## 5. Optimize & Export
Once training is done, optimize for your target device:
```bash
brewslm optimize --project 1 --target "mobile_iphone15"
```

## 6. Monitor Spend
Check your project budget and current spend:
```bash
brewslm project budget --id 1
```

---
For more details, visit our docs at https://docs.brewslm.ai/

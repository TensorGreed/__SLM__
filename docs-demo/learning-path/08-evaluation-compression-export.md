# Evaluation, Compression, And Export

Status: partial.

## Evaluation

Evidence:
- `EvalPanel.tsx`
- `backend/app/api/evaluation.py`

Supported surfaces include held-out evaluation, LLM judge, eval packs, gates, scorecard, safety scorecard, failure clusters, and remediation plans.

## Compression

Evidence:
- `CompressionPanel.tsx`
- `backend/app/api/compression.py`
- `backend/app/services/compression_service.py`

Quantize, merge LoRA, merge models, and benchmark surfaces exist. Real behavior depends on external tooling or stub settings.

## Export

Evidence:
- `ExportPanel.tsx`
- `backend/app/api/export.py`

Export create/run, deployment validation, deploy-as-api, serve plan, serve runs, and optimization matrix surfaces exist.

## Demo Rule

Do not show a polished compression/export claim until one successful local run is captured.


# Final Model Usage Plan

Discovery date: 2026-05-19.

This plan answers what a user can actually do after training, based on repo
evidence. It does not claim that a real artifact has already been produced.

Status legend: verified, partial, simulated, estimated, conceptual, unsupported,
unknown.

## Actually Available Usage Surfaces

| Usage surface | Status | Evidence | What it can prove | What it cannot prove yet |
|---|---|---|---|---|
| Prompt playground | partial | `frontend/src/pages/ProjectPlaygroundPage.tsx`, `frontend/src/components/training/ChatPlaygroundPanel.tsx`, playground endpoints in `backend/app/api/training.py`, `backend/app/services/playground_service.py`, `backend/app/services/playground_session_service.py` | UI can send prompts to mock, OpenAI-compatible, or llama.cpp-style provider; sessions/logs/feedback exist | A mock response does not prove a trained model works; automatic trained-model selection needs a run/export/registry check |
| Model options from project artifacts | partial | `backend/app/services/playground_session_service.py` | Model list can include project base model, registry model, experiment output path, and artifact registry items | Needs an actual experiment/export/artifact to verify end-to-end linkage |
| Export package | partial | `frontend/src/components/export/ExportPanel.tsx`, `backend/app/api/export.py`, `backend/app/services/export_service.py`, `backend/app/models/export.py` | Completed model artifacts can be packaged with manifest, inference script, Dockerfile, reports, and release notes | Export fails if no real training/compression artifacts exist |
| Local serve plan | partial | `backend/app/services/serve_service.py`, export/registry serve-plan endpoints | Generates local commands, health checks, smoke-test curl commands, and runtime hints for exported/registered models | Commands only work when local runtime and artifacts exist |
| Local serve run manager | partial | `backend/app/services/serve_runtime_service.py` | Can start/track/stop local subprocess serve runs and collect telemetry/status | Runtime command may fail without dependencies such as vLLM, TGI, Ollama, llama.cpp, ONNX Runtime, Docker, or model files |
| Model registry | partial | `frontend/src/components/export/ExportPanel.tsx`, `backend/app/api/registry.py`, `backend/app/services/registry_service.py` | Can register artifacts, compute readiness, promote, and mark deployment metadata | Promotion can be blocked by gates; needs eval/export data for a useful readiness story |
| Deployment assistant and telemetry | partial | `frontend/src/pages/ProjectDeploymentsPage.tsx`, `backend/app/api/deployments.py`, `backend/app/services/deployment_target_service.py` | Can show deployment versions, telemetry summaries, drift checks, scores, and target validation/deploy plans | Requires deployed model metadata and telemetry events |
| Generated API usage | partial | `backend/app/services/export_service.py`, `backend/app/services/serve_service.py` | Export/serve plans include generated `serve.py` and curl smoke tests for local APIs | Needs completed export and successful local serve |

## Export Formats And Target Profiles

Status: verified for declared formats, partial for successful local output.

Evidence:

- `backend/app/models/export.py`
- `backend/app/services/export_service.py`
- `backend/app/services/deployment_target_service.py`
- `backend/scripts/quantize.py`

Declared export formats:

- `gguf`
- `onnx`
- `tensorrt`
- `huggingface`
- `docker`

Deployment target evidence:

- Hugging Face-style artifacts can target Hugging Face, Docker, vLLM, and TGI
  style flows depending on format/profile.
- GGUF can target local llama.cpp/Ollama-style flows.
- ONNX and TensorRT have artifact validation logic and generated runtime
  templates.
- Deployment validation checks for actual files such as `.gguf`, `.onnx`,
  `.engine`, or `.plan` where appropriate.

Compression/export dependency evidence:

- GGUF/ONNX/TensorRT-style exports usually require a training artifact and often
  a compression artifact.
- `backend/scripts/quantize.py` contains real GGUF and ONNX paths, but those
  require external packages/tools.
- Export service refuses to proceed if no model artifact is found.

## Recommended Real End-To-End Usage Demo

Status: partial until executed.

1. Seed exactly one official sample.
2. Confirm raw data, gold set, and prepared splits.
3. Run or select a training experiment that produces a real artifact.
4. Run evaluation and capture the metrics/scorecard/gates.
5. Create an export in the format that matches the sample target and local
   tooling.
6. Build the serve plan for the export.
7. Start a local serve run only if the runtime command is available.
8. Send one smoke request through the generated curl/API path.
9. Open the playground and call the same local endpoint if supported.
10. Register the model and show readiness/promotion status if gates allow.

Viewer should see:

- The exact model artifact or export package.
- A measurable evaluation result.
- A concrete API request and response.
- A UI usage path, preferably playground against the served model.
- Registry/deployment status only when those records are real.

## Sample-Specific Usage Hypotheses

These are not final claims until run.

| Sample | Plausible final usage | Evidence | Status |
|---|---|---|---|
| `support-faq` | Serve as an OpenAI-compatible FAQ assistant, then test in playground | Manifest target `vllm_server`; playground and serve-plan surfaces exist | partial |
| `pii-detector` | Serve an extraction model that returns entity spans/JSON | Manifest task `structured_extraction`; structured eval handler; PII span schema | partial |
| `sentiment-classifier` | Export or serve a three-class classifier, potentially ONNX/mobile-oriented | Manifest target `mobile_cpu`; export enum includes `onnx`; classification eval handler | partial |

## Simulated, Estimated, And Unknown Items

- Playground provider `mock` is simulated and must be labeled as such.
- Simulated training exists only when explicitly enabled with environment
  configuration.
- Compression stub paths exist only when explicitly enabled and are not real
  compression.
- Export optimization can include estimated metrics; mark estimates clearly.
- Remote deployment targets may need credentials or external services.
- It is unknown whether a freshly trained model appears automatically in the
  playground without export/registry/endpoint setup.
- It is unknown which export format is the most reliable for a short local demo
  until a small training artifact exists.

## Evidence Needed Before Recording Final Model Usage

- Run one tiny official-sample training path or intentionally enable and label
  simulation.
- Capture the experiment output directory and model artifact list.
- Run one evaluation and capture metrics/gates.
- Create one export and inspect its `manifest.json`.
- Build one serve plan and validate whether local command dependencies exist.
- Prove one API smoke request returns from the trained/exported model.
- Prove one UI playground call reaches that same model endpoint, or mark the
  playground part as unavailable for the first recording.

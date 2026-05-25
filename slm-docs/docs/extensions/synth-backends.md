---
sidebar_position: 6
title: Synthetic-data backends
---

# Synthetic-data backends

BrewSLM's synthetic-data playbooks (USER-SUCCESS Epic 2) run against a pluggable **`SynthBackend`** — one thin async wrapper per LLM transport. The default install ships two:

1. **`OllamaBackend`** — local Ollama on `OLLAMA_HOST` (default `http://localhost:11434`). Picks the strongest installed model from a preference list (`llama3.1` → `llama3` → `qwen2.5` → …) unless you pin one.
2. **`TeacherModelBackend`** — generic OpenAI-compatible endpoint at `TEACHER_MODEL_API_URL` with optional `TEACHER_MODEL_API_KEY`.

USER-SUCCESS Epic 5 added two more **opt-in** backends:

3. **`NemoBackend`** — NVIDIA NeMo Data Designer / NIM (NVIDIA Inference Microservice) endpoint. Useful when you have NVIDIA hardware locally and want to drive synthesis with a larger model than Ollama can realistically serve.
4. **`VllmBackend`** — local **vLLM** OpenAI-compatible server. This is the backend that **actually exercises** Phase 5b's schema-constrained generation on small / local models: vLLM honors `response_format=json_schema` end-to-end (via its xgrammar/outlines backend), while Ollama silently ignores the field.

This page covers `NemoBackend` and `VllmBackend` setup. The Ollama + Teacher backends are zero-config for the local case.

## NeMo / NIM setup

`NemoBackend` talks to a NIM-style HTTP endpoint over the standard OpenAI-compatible `/v1/chat/completions` API. It is **not** a Python SDK integration — there's no NeMo Python dependency in BrewSLM. This keeps the install surface narrow and means any OpenAI-compatible NVIDIA serving stack works (NIM, NeMo Inference, vLLM-on-NIM, hosted NGC endpoints, etc.).

### 1. Stand up a NIM (or point at a hosted one)

Follow the NVIDIA NIM getting-started guide for your target model. The summary:

- **Hosted (NGC)** — get an API key from [build.nvidia.com](https://build.nvidia.com) and use its base URL (`https://integrate.api.nvidia.com`).
- **Local container** — pull and run the NIM container on a machine with the right NVIDIA driver + container runtime. Typical local URL: `http://localhost:8000`.

The exact install steps depend on which model you're running; the NIM project docs are the source of truth.

### 2. Configure BrewSLM

Set these env vars (or add them to your `.env`):

```bash
NEMO_API_URL=http://localhost:8000         # base URL only; no trailing slash needed
NEMO_API_KEY=                              # optional bearer token (NGC keys, hosted NIMs)
NEMO_DEFAULT_MODEL=meta/llama-3.1-70b-instruct  # required — must match a model the NIM serves
NEMO_TIMEOUT_SECONDS=600                   # default 600s (matches the Ollama timeout)
```

`NEMO_DEFAULT_MODEL` **must** be set. Unlike Ollama (which can probe `GET /api/tags` and auto-pick the strongest installed model), NIM endpoints typically serve one model per process — there's no clean auto-pick rule, so we ask for an explicit choice.

### 3. Verify reachability

Restart BrewSLM. With the NIM running and the env vars set, you should now see two API surfaces light up:

```bash
curl http://localhost:8000/api/projects/1/synthetic/backends
```

Should return something like:

```json
{
  "project_id": 1,
  "backends": [
    {"name": "ollama", "available": true,  "describe": "ollama:llama3.1:8b"},
    {"name": "teacher", "available": false, "describe": "teacher"},
    {"name": "nemo",   "available": true,  "describe": "nemo:meta/llama-3.1-70b-instruct"}
  ]
}
```

In the UI, the **Synthetic** tab's playbook panel will now show a **Backend** dropdown next to "Target rows". The dropdown is hidden when fewer than two backends are available (single-backend installs don't see clutter).

Each option that actually honors the playbook's `response_schema` (`nemo`, `vllm`) is suffixed with **· schema-aware** in the dropdown text. When the active selection is schema-aware, a green **✓ schema-aware** chip appears next to the picker. The `/backends` endpoint returns a `schema_aware: bool` per entry so the frontend can render the badge without hard-coding backend names.

### Coach Mode auto-pins schema-aware backends

Coach Mode's class-imbalance suggestion (`gold_set:class-imbalance`) — the click-to-execute action that runs `class_balance_fill` — auto-pins the highest-ranked schema-aware backend when one is configured + reachable. Preference order is **vLLM > NeMo** (vLLM enforces the schema during decoding via xgrammar / outlines; NeMo passes it upstream where enforcement quality depends on the model + NIM version). When neither is reachable the suggestion stamps no `backend` and the orchestrator's auto-pick takes over (typically Ollama, which silently ignores the schema). This means users on a vLLM-equipped install get constrained decoding "for free" from the Coach button without having to remember to switch the dropdown manually.

The Coach suggestion card surfaces the auto-pinned backend under the action button as a caption — e.g. **will run on `vllm:meta-llama/Meta-Llama-3.1-8B-Instruct`  ✓ schema-aware** — so users see the upgrade happening instead of it being silently applied. The schema-aware chip renders only when the pinned backend matches `context.schema_aware_backend` (also stamped by `coach_service`), so the UI never has to maintain its own backend-name allowlist.

## vLLM setup

`VllmBackend` talks to a local vLLM server over the standard OpenAI-compatible `/v1/chat/completions` API. vLLM is the recommended backend when you want Phase 5b's schema-constrained generation to actually constrain decoding — it implements `response_format=json_schema` via xgrammar / outlines, which Ollama does not.

### 1. Stand up vLLM

```bash
# In a fresh venv (vLLM has a heavy CUDA install):
pip install vllm

# Serve any HuggingFace-format model. vLLM picks GPUs automatically.
vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 8192
# Optional: --api-key <secret> to require bearer auth.
```

On a DGX Spark, vLLM happily serves 70B-Instruct at FP8 or AWQ on a single device. Pick a model your hardware can actually load.

### 2. Configure BrewSLM

```bash
VLLM_API_URL=http://localhost:8000
VLLM_API_KEY=                                     # optional; matches --api-key on the vLLM side
VLLM_DEFAULT_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct
VLLM_TIMEOUT_SECONDS=600
```

`VLLM_DEFAULT_MODEL` **must** be set and **must** match exactly what `vllm serve` is hosting (vLLM serves one model per process). Hit `GET $VLLM_API_URL/v1/models` to confirm the id.

### 3. Why use vLLM over Ollama

For most users, Ollama is fine — the playbook parser already handles markdown-fenced output, drift, and other small-model quirks. Switch to vLLM when:

- The `class_balance_fill` playbook (or any future playbook) needs **hard label-enum guarantees** — vLLM's structured-outputs backend rejects on-decode any token sequence that wouldn't validate against the schema. Ollama would just regenerate prose around the JSON.
- You're running a small (≤ 8B) instruction-tuned model where the format-following capability isn't reliable — vLLM's schema constraint compensates.
- You want vLLM's higher per-GPU throughput vs. Ollama for large batches.

## How auto-pick works

When the playbook is run without an explicit backend pin (the default for every existing call site, including Coach Mode click-to-execute actions), `pick_backend(None)` walks the registry in this order and returns the first that's available:

1. `OllamaBackend`
2. `TeacherModelBackend`
3. `NemoBackend`
4. `VllmBackend`

**NeMo and vLLM are positioned last on purpose.** Existing Ollama-only installs see no behavior change — the auto-pick still returns Ollama. They're only used when:

- The user explicitly selects them from the picker dropdown (which sends `backend: "nemo:..."` or `backend: "vllm:..."` on the run-playbook call), **or**
- Nothing earlier in the registry is configured/reachable (in which case auto-pick falls through).

To pin one of the power-user backends as the default, leave `OLLAMA_HOST` pointing at a non-running daemon — `is_available()` returns False quickly and auto-pick moves on.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Backend picker missing on the Synthetic tab | Only one backend reachable | Confirm `NEMO_API_URL` is set + the NIM responds to `GET /v1/models`. |
| `NEMO_API_URL is not set` error on generate | Env var not loaded into the Python process | Restart the backend after editing `.env`. |
| `NEMO_DEFAULT_MODEL is not set` error | Required env var missing | Set the model id and restart. |
| `HTTP 401 — check NEMO_API_KEY` | Endpoint requires auth | Add `NEMO_API_KEY=<bearer-token>` to `.env`. |
| `HTTP 404 — is 'meta/llama-3.1-70b-instruct' loaded on this NIM?` | Model id doesn't match what the NIM serves | Hit `GET $NEMO_API_URL/v1/models` and use one of the listed model ids. |
| `NeMo timed out after 600s` | Large prompts / underpowered GPU | Lower `target_count` in the playbook UI, bump `NEMO_TIMEOUT_SECONDS`, or pick a smaller model. |
| Generated rows "disappear" after a successful run | They went to the **synth review queue** with `review_status="pending"` — gated out of training until you accept them | Open the Synthetic tab's review queue and accept/reject the rows. |
| vLLM returns `HTTP 400 — vLLM rejected the JSON Schema` | The playbook's `response_schema()` used a feature vLLM's structured-outputs backend doesn't support (e.g. `$ref`, complex `oneOf`) | Simplify the schema in the playbook's `response_schema(ctx)` method — keep it to plain `type`, `properties`, `required`, `enum`. |
| vLLM serves but throughput is awful | Default block size / dtype suboptimal for your card | Pass `--dtype auto --max-num-batched-tokens 8192` (or bf16 on Hopper / SM 89+); see vLLM's perf-tuning docs. |

## Schema-constrained generation (Phase 5b)

Phase 5b extends `SynthBackend.complete()` with an optional `response_schema: dict | None = None` parameter:

- **`VllmBackend`** *(Phase 5c)* — the canonical schema-honoring backend for local installs. vLLM's structured-outputs backend (xgrammar / outlines) enforces the schema during decoding, so the model literally cannot emit a token sequence that wouldn't validate. Best choice when you want the schema to actually constrain output rather than just label the request.
- **`NemoBackend`** — forwards the schema as `response_format={"type": "json_schema", "json_schema": {"name": "synth_row", "schema": <your-schema>, "strict": true}}` on the chat-completion payload. NIM honors this exactly like the OpenAI Structured-Outputs API — the model is constrained to emit JSON validating against the schema, so playbooks get clean parses even on small / instruction-shaky models.
- **`OllamaBackend`** — accepts the kwarg and silently ignores it. Ollama's `/v1/chat/completions` doesn't implement OpenAI's `response_format=json_schema`, so threading the schema in would just add an ignored field; the playbook's `parse_output()` + `validate()` pass handles structure either way.
- **`TeacherModelBackend`** — same: accepts the kwarg, doesn't forward it (the legacy dispatcher has no schema hook).

Playbooks opt in by defining an optional `response_schema(ctx) -> dict | None` method. The orchestrator picks it up via `get_response_schema(playbook, ctx)` and forwards the result.

Today, **only the `class_balance_fill` playbook** (classification recipe) defines a schema — it builds a JSON Schema from the gold-set labels and narrows the `label` enum to the resolved `target_class` so NIM can't drift to a different class. Other playbooks return `None` and run as before.

Defense-in-depth: `parse_jsonl_lines` still strips markdown fences after the fact, so a NIM that ignores `strict` (or any fallback backend that silently ignored the schema) still produces parseable rows.

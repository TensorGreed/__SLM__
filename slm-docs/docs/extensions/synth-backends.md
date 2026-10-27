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

In the UI, the **Data** → **Generate** playbook panel will now show a **Backend** dropdown next to "Target rows". The dropdown is hidden when fewer than two backends are available (single-backend installs don't see clutter).

Each option that actually honors the playbook's `response_schema` (`nemo`, `vllm`) is suffixed with **· schema-aware** in the dropdown text. When the active selection is schema-aware, a green **schema-aware** chip (with a check icon) appears next to the picker. The `/backends` endpoint returns a `schema_aware: bool` per entry so the frontend can render the badge without hard-coding backend names.

### The class-imbalance suggestion auto-pins schema-aware backends

The Coach's class-imbalance suggestion (`gold_set:class-imbalance`) — the click-to-execute action that runs `class_balance_fill` — auto-pins the highest-ranked schema-aware backend when one is configured + reachable. Preference order is **vLLM > NeMo** (vLLM enforces the schema during decoding via xgrammar / outlines; NeMo passes it upstream where enforcement quality depends on the model + NIM version). When neither is reachable the suggestion stamps no `backend` and the orchestrator's auto-pick takes over (typically Ollama, which silently ignores the schema). This means users on a vLLM-equipped install get constrained decoding "for free" from the Coach button without having to remember to switch the dropdown manually.

The suggestion notice surfaces the auto-pinned backend under the action button as a caption — e.g. **will run on `vllm:meta-llama/Meta-Llama-3.1-8B-Instruct`  schema-aware** — so users see the upgrade happening instead of it being silently applied. The schema-aware chip renders only when the pinned backend matches `context.schema_aware_backend` (also stamped by `coach_service`), so the UI never has to maintain its own backend-name allowlist.

### A notice reminds you to review pending synth rows

After the Coach action runs the click-to-execute class-imbalance action, the generated rows land in the synth review queue with `review_status="pending"` and are **silently gated out of training** by the dataset-prep loader. To close that loop, the gold_set Coach surface emits a `gold_set:synth-review-pending` suggestion whenever the queue has pending rows. Severity scales with the pile-up — `info` for 1–4 rows (just a nudge), `warning` for 5+ (meaningful chunk of generated data not yet in training).

The action is a `navigate` with target `synthetic-review-queue`. Coach also stamps the top pending bucket's `synth_source` onto `action.params.synth_source`. The Coach navigate handler builds a focused URL:

```
/project/{id}/pipeline/synthetic?focus_synth_source=<source>#synth-review-queue
```

`SyntheticPanel` reads `focus_synth_source` from the URL, scrolls the queue into view via the `#synth-review-queue` hash, and passes the source down to `SynthReviewQueue` as a `focusSource` prop. `SynthReviewQueue` then renders a prominent focus banner at the top of the panel:

> **Focused on** `playbook:classification:class_balance_fill:class=technical`  ·  12 pending rows  ·  **[ Accept all 12 rows ]**   _Clear focus_

The "Accept all N rows" button POSTs only the ids of the focused group to `/synthetic/review-queue/bulk-update` — never sibling sources (e.g. a `class=billing` bucket from a different run). The banner auto-dismisses once the source bucket is drained. This closes the auto-pin loop in **two clicks**: Coach button → Accept all.

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

## Command (local CLI) backend

`CommandBackend` (pin: `command` or `command:<model>`) runs an **allow-listed command-line tool** as the teacher. It is meant for a vendor CLI signed in to a *subscription* (for example `claude -p`), so generation costs no per-call API credits. It is **off until you allow-list a binary**, and it is **never auto-picked**: `pick_backend(None)` and the judge's automatic choice skip it; you choose it explicitly (the backend picker, a flow's model dropdown, or the **Judge model** card in project Settings).

Prompts go to whatever the CLI talks to, so a CLI judge counts as *not local* (a local-only external benchmark skips it).

### Settings

All install-wide environment variables (nothing below can be set per project, except the short list after the table).

| Variable | Default | Effect |
|---|---|---|
| `SYNTH_COMMAND_ALLOWED_BINARIES` | empty | Comma list: bare names (match only when `argv[0]` is itself a bare name, never an absolute path with the same basename) or absolute paths (match the resolved executable's real path). **Empty disables the backend.** |
| `SYNTH_COMMAND_ARGV` | `["claude","-p","--output-format","json","--model","{model}"]` | JSON list. `{model}` is replaced inside an element; in `arg` mode an element that is exactly `{prompt}` receives the prompt. |
| `SYNTH_COMMAND_PROMPT_MODE` | `stdin` | `stdin` or `arg`. In `arg` mode a prompt starting with `-` gets one leading space so it can never parse as a flag. |
| `SYNTH_COMMAND_OUTPUT` | `.result` | `raw` (stdout text) or a dotted JSON path: `.result`, `.data.text`, `.choices.0.text`. |
| `SYNTH_COMMAND_DEFAULT_MODEL` | `sonnet` | Substituted for `{model}`; `command:opus` pins another. Must match `^[A-Za-z0-9][A-Za-z0-9._:/@-]{0,127}$` (no leading `-`, so a model name can never parse as a flag; a bad `command:<model>` pin is a 400, and `PUT /eval-judge` a 422). |
| `SYNTH_COMMAND_CONCURRENCY` | `1` | CLI processes at once (1 to 8). One process-wide limit sized by this setting; a per-project `concurrency` override can only lower it. |
| `SYNTH_COMMAND_TIMEOUT_SECONDS` | `600` | The process is killed after this. |
| `SYNTH_COMMAND_MAX_RETRIES` | `2` | Other failures retry with exponential backoff. |
| `SYNTH_COMMAND_BACKOFF_SECONDS` | `5.0` | First backoff; doubles each retry. |
| `SYNTH_COMMAND_BATCH_SIZE` | `8` | Requests packed into one CLI call (1 to 32; 1 = no packing). |
| `SYNTH_COMMAND_USAGE_LIMIT_REGEX` | `usage limit\|rate limit\|\b429\b` | Case-insensitive, matched against stdout + stderr of a call that **failed** (non-zero exit or `is_error`); text in a successful reply is never matched. An optional named group `reset` holding a unix epoch sets the cool-down (an epoch in the past falls back to the cool-down). |
| `SYNTH_COMMAND_COOLDOWN_SECONDS` | `900` | Pause after a usage-limit failure (when no `reset` epoch). Never shorter than 60 s, whatever is configured. |
| `SYNTH_COMMAND_MAX_PAUSE_SECONDS` | `28800` | Total pause per Job before it fails. |
| `SYNTH_COMMAND_ENV_PASSTHROUGH` | empty | Extra environment variable names to forward. |
| `SYNTH_COMMAND_PLAYBOOK_ROWS_PER_CALL` | `25` | Rows asked per CLI call by a synthetic playbook. |

A project may override only `model`, `concurrency`, `batch_size`, `timeout_seconds`, `cooldown_seconds` and `max_retries`, through `runtime_config["synth_command"]` (values are clamped; unknown keys are ignored). The binary, argv, environment and prompt mode are never per project: a project editor must not choose flags.

### Safety model

- **Allow-list**, checked when the backend is built and again on every call. A binary that is on `PATH` but not listed is refused (`binary not allow-listed: <name>`).
- **No shell**: `create_subprocess_exec` with an argv list. Placeholders are substituted inside single argv elements, so a prompt of `; touch x; $(touch y)` arrives as literal text.
- **Minimal environment**: only `PATH`, `HOME`, `USER`, `LOGNAME`, `LANG`, `LC_ALL`, `TMPDIR` and the `XDG_*` directories (those that are set) plus the names in `SYNTH_COMMAND_ENV_PASSTHROUGH`.
- **Denied key variables**: `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN` and `OPENAI_API_KEY` are never forwarded, even when listed in the passthrough. A key in the environment makes such CLIs bill the metered API instead of your subscription.
- **Temporary working directory** per call, so the CLI never reads this repository's instructions or files.
- The process (and its children) is killed on timeout and when the Job is cancelled: a cancel is noticed before each call and while one is running (polled about every 2 s), and cancelling one chunk of a packed batch stops its sibling chunks.

`max_tokens` and `temperature` are ignored (a CLI has no such knobs). A `response_schema` becomes an instruction in the prompt ("reply with only JSON matching ...").

### Batching

`complete_batch` packs up to `SYNTH_COMMAND_BATCH_SIZE` requests into one call under a strict JSON-lines contract: the prompt lists items `{"id": "r<k>", "system": ..., "prompt": ...}` and demands exactly one line `{"id": "r<k>", "output": "<answer as a JSON string>"}` per item and nothing else (a code fence is tolerated). Items missing, malformed or duplicated in the reply are retried **once, individually**; still failing, they come back as `None` with a reason (`missing_from_batch`, `malformed_json`, `empty_output`, `call_failed`) in `batch_report` and `rejected_items`. A packed call that fails outright is not retried per item. The documents-to-Q&A flow, teacher answers and the relabel tools use this path, and **Batch** in the documents flow now accepts the CLI backend as well as Anthropic models.

### Pause and resume

When a failed call matches the usage-limit regex, every call in the process waits on a shared cool-down. The running Job's message reads **waiting for usage window - resumes ~HH:MM UTC** (the bell shows a "paused" chip), then the same call is retried; a pause does not use up a retry. The Job result gains `paused_seconds`. Past `SYNTH_COMMAND_MAX_PAUSE_SECONDS` in total the Job fails, and one call that hits the limit 20 times in a row gives up. Calls queued behind the concurrency limit re-check the pause before they start, so a pause never lets a burst through. The message shown before the pause is restored afterwards.

Synthetic playbooks ask a CLI for at most `SYNTH_COMMAND_PLAYBOOK_ROWS_PER_CALL` rows per call, so a larger target runs as **chunks** ("chunk 3/8 - 60 rows so far"), each told "Batch k of n: vary topics and wording" and de-duplicated against earlier chunks. Each finished chunk is appended and fsynced to `data/projects/<id>/synthetic/playbook_checkpoints/<hash>.jsonl`; a re-run with the same parameters (recipe, mode, backend, count, class, cluster) reuses finished chunks instead of calling them again. The checkpoint key also includes a hash of the first chunk's prompt, so changing the answer key or examples starts fresh. The file is deleted only after the rows are saved and committed; a crashed run can be resumed for up to 7 days (older files are ignored and removed). A chunk that fails after the backend's retries is listed under `failed_chunks` in the Job result and the run goes on; zero rows overall still fails the Job.

Cloud models chunk the same way (`SYNTH_CLOUD_PLAYBOOK_ROWS_PER_CALL`, default 10). With an Anthropic model, the playbook form's **Half price (Message Batch)** toggle (`use_batch` on `run-playbook`) sends every chunk not yet in the checkpoint as one Message Batch, and `scenario_grid` sends its scenario calls and then its plan calls as one batch each. Results take minutes to hours; the Job shows "Batch: N/M requests done" while it waits, and cancelling the Job cancels the batch. Requests the batch didn't answer are retried directly at list price. The Job result reports the run's own `usage` / `cost_usd` (batch calls at half price) and `batch` (requests, succeeded, redone directly). The quick check before the run is always one direct call.

### Spend

Usage is recorded as `subscription:<binary>` (tokens from the CLI's JSON when present; the call counts either way), priced **$0** and shown on its own "Subscription CLI" line in the spend card, never against the API budget. See [Cloud LLM spend](../workflows/data-ingestion.md).

The hard API budget that protects metered calls is a guard rail, not an invoice. Known gaps: spend of a Job still running when the backend restarts is lost to the ledger until the Job ends; calls made outside any project count against the account-wide cap only; calls started together can each overshoot the cap, so N concurrent calls can exceed it by up to N calls' cost; a metered model with no known price is counted at the highest listed price (the spend card still says "price unknown"); Message Batches spend is added to the ledger when the batch results are collected, at half price.

### Recipe: Claude Code (headless)

```sh
export SYNTH_COMMAND_ALLOWED_BINARIES=claude
export SYNTH_COMMAND_ARGV='["claude","-p","--output-format","json","--model","{model}","--tools","","--no-session-persistence","--strict-mcp-config"]'
```

- Do **not** add `--bare`: it only authenticates with an API key, which means metered billing instead of your subscription.
- Keep `ANTHROPIC_API_KEY` out of the environment (BrewSLM never forwards it anyway).
- Confirm that your plan's terms allow this automated use before you rely on it.

## How auto-pick works

When the playbook is run without an explicit backend pin (the default for every existing call site, including Coach click-to-execute actions), `pick_backend(None)` walks the registry in this order and returns the first that's available:

1. `OllamaBackend`
2. `TeacherModelBackend`
3. `NemoBackend`
4. `VllmBackend`

`CommandBackend` and `CloudLlmBackend` are **never** auto-picked: they cost money or send data off the machine, so you pin them.

**NeMo and vLLM are positioned last on purpose.** Existing Ollama-only installs see no behavior change — the auto-pick still returns Ollama. They're only used when:

- The user explicitly selects them from the picker dropdown (which sends `backend: "nemo:..."` or `backend: "vllm:..."` on the run-playbook call), **or**
- Nothing earlier in the registry is configured/reachable (in which case auto-pick falls through).

To pin one of the power-user backends as the default, leave `OLLAMA_HOST` pointing at a non-running daemon — `is_available()` returns False quickly and auto-pick moves on.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Backend picker missing on the **Data** → **Generate** step | Only one backend reachable | Confirm `NEMO_API_URL` is set + the NIM responds to `GET /v1/models`. |
| `NEMO_API_URL is not set` error on generate | Env var not loaded into the Python process | Restart the backend after editing `.env`. |
| `NEMO_DEFAULT_MODEL is not set` error | Required env var missing | Set the model id and restart. |
| `HTTP 401 — check NEMO_API_KEY` | Endpoint requires auth | Add `NEMO_API_KEY=<bearer-token>` to `.env`. |
| `HTTP 404 — is 'meta/llama-3.1-70b-instruct' loaded on this NIM?` | Model id doesn't match what the NIM serves | Hit `GET $NEMO_API_URL/v1/models` and use one of the listed model ids. |
| `NeMo timed out after 600s` | Large prompts / underpowered GPU | Lower `target_count` in the playbook UI, bump `NEMO_TIMEOUT_SECONDS`, or pick a smaller model. |
| Generated rows "disappear" after a successful run | They went to the **synth review queue** with `review_status="pending"` — gated out of training until you accept them | Open the **Data** → **Generate** step's review queue and accept/reject the rows. |
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

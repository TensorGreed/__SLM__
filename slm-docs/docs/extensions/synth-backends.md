---
sidebar_position: 6
title: Synthetic-data backends
---

# Synthetic-data backends

BrewSLM's synthetic-data playbooks (USER-SUCCESS Epic 2) run against a pluggable **`SynthBackend`** — one thin async wrapper per LLM transport. The default install ships two:

1. **`OllamaBackend`** — local Ollama on `OLLAMA_HOST` (default `http://localhost:11434`). Picks the strongest installed model from a preference list (`llama3.1` → `llama3` → `qwen2.5` → …) unless you pin one.
2. **`TeacherModelBackend`** — generic OpenAI-compatible endpoint at `TEACHER_MODEL_API_URL` with optional `TEACHER_MODEL_API_KEY`.

USER-SUCCESS Epic 5 added a third, **opt-in** backend:

3. **`NemoBackend`** — NVIDIA NeMo Data Designer / NIM (NVIDIA Inference Microservice) endpoint. Useful when you have NVIDIA hardware locally and want to drive synthesis with a larger model than Ollama can realistically serve.

This page covers `NemoBackend` setup. The Ollama + Teacher backends are zero-config for the local case.

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

## How auto-pick works

When the playbook is run without an explicit backend pin (the default for every existing call site, including Coach Mode click-to-execute actions), `pick_backend(None)` walks the registry in this order and returns the first that's available:

1. `OllamaBackend`
2. `TeacherModelBackend`
3. `NemoBackend`

**NeMo is positioned last on purpose.** Existing Ollama-only installs see no behavior change — the auto-pick still returns Ollama. NeMo is only used when:

- The user explicitly selects it from the picker dropdown (which sends `backend: "nemo:..."` on the run-playbook call), **or**
- Neither Ollama nor Teacher is configured/reachable (in which case auto-pick falls through to NeMo).

To pin NeMo as the default, leave `OLLAMA_HOST` pointing at a non-running daemon — `is_available()` returns False quickly and auto-pick moves on.

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

## Why no schema-constrained output yet?

The Epic 2 playbook framework calls `backend.complete(prompt)` and gets raw text back; the playbook then runs its own `parse_output()` + `validate()` pass. NeMo's real differentiator — schema-guided generation — requires extending the `SynthBackend` protocol with a `response_schema` parameter and threading it through all three backends.

That's deliberately deferred to **Epic 5 Phase 5b**. Phase 5a ships the transport so power users can start exercising NeMo for raw text generation today.

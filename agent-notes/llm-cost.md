# Cloud LLMs: providers, spend, budgets, Ollama context

Detail notes split out of CLAUDE.md. Read when working on this area.

- **Cloud LLM spend (all call sites)** — `services/llm_usage.py`: `call_anthropic_chat` /
  `call_openai_chat` (non-local hosts) and the teacher endpoints (`call_teacher_model`,
  distillation `call_teacher_with_logprobs`; `is_local_url` skips Ollama/LAN) `record()`
  into every open `UsageMeter` (contextvar; survives `to_thread` / nested `asyncio.run`).
  `jobs_service._runner_wrapper` opens a ROOT meter per Job → `merge_into_result` adds
  `usage` (estimate_cost list) / `cost_usd` / `tokens` (a runner's own `usage` list wins;
  a non-list `usage` key → `llm_usage`), also on FAILED / CANCELLED Jobs. Non-Job ops use
  `metered_operation(pid, kind)` → outermost appends `data/projects/<id>/llm_spend.jsonl`
  (`gold_llm_generate` → response `llm_cost`; `plan_refinement` → refinement `llm_cost`;
  `probe_judge` → `metrics["probe"]` usage; `distillation_capture` → task dict). Answer /
  retrieval judge snapshots carry `usage`. Caching: judge / probe / gold / strategy system
  prompts → `cached_system` (same text); `CloudLlmBackend.complete(cache_system=)` +
  `llm_usage.cache_kwargs(llm)` (seed documents, playbooks, judges via synth backend);
  span synthesis + documents→Q&A send instructions as `cache_prefix` (uncached backends get
  the unchanged prompt; `passage_prompt_parts`). Below the model's minimum cacheable
  length Anthropic just doesn't cache. Tally: `llm_spend_service.project_llm_spend` (Jobs ∪
  ledger, by model / kind, window) → `GET /projects/{id}/llm-spend?days=30` →
  `LlmSpendCard` on the Synthetic tab; bell `cloudSpendText` on every kind ("spent …" on
  failed). Tests `tests/test_llm_spend.py`, `LlmSpendCard.test.tsx`.
  **Playground**: `playground_service.meter_playground_reply` (chat + stream routes in
  `api/training.py`, after the reply / final event) → `metered_operation(pid,
  "playground_chat")` only when `is_metered_chat` (openai_compatible + non-local URL;
  mock / experiment / llama_cpp / Ollama never); usage from the response (OpenAI
  `prompt_tokens` minus `prompt_tokens_details.cached_tokens` → cache_read, or Anthropic
  `input_tokens` / `cache_*_input_tokens`); a cloud stream sends `stream_options.include_usage`;
  no usage → chars/4 estimate, `meter.estimated` → ledger `estimated: true` → tally
  `estimated_operations`. Reply / final event carry `llm_cost`. Never raises.
  **Budget**: `runtime_config["llm_budget_usd"] = {monthly, warn_at (0.8)}`
  (`llm_spend_service.normalize_budget` / `read_budget`; `GET/PUT /projects/{id}/llm-budget`,
  `monthly: null` clears, 422 on monthly ≤ 0 / warn_at ∉ (0,1]). Tally adds `month`
  (calendar month-to-date UTC, any window), `budget`, `budget_status` (pure `budget_status`:
  ok / warning ≥ warn_at / over > monthly). `month_budget_status(db, project)` feeds the
  Coach synthetic-stage `synthetic:llm-budget` (`_llm_budget_nudge`: warning / critical,
  `pipeline-tab` synthetic, emitted even before data). `budget_warning(db, pid)` (only when
  over) → `_with_budget_warning` on relabel-answer-key + run-playbook?async_job starts —
  advisory, Job still starts; frontend `budgetWarningMessage` toasts it (SyntheticPanel,
  SynthReviewQueue, PlaybookPickerPanel). `LlmSpendCard` `BudgetSection`: "This month: ~$x
  of $y (z%)" (`--warning` / `--over` classes) + editor. Tests `tests/test_llm_budget.py`.
- **Anthropic `temperature`**: newer models (e.g. claude-sonnet-5-5) answer
  400 "`temperature` is deprecated for this model" —
  `cloud_llm_service.call_anthropic_chat` retries once without it and records
  the model in `_NO_TEMPERATURE_MODELS` (later calls omit it). Tests:
  `tests/test_cloud_llm_temperature.py`.
- **Ollama context**: `synth_backends/ollama.context_window_for` sets
  `num_ctx` per request (prompt chars / 3 + max_tokens + 256, buckets
  4096 / 8192 / 16384 / 32768) — without it Ollama runs at its server
  default and silently drops the start of a prompt that doesn't fit.
- **Saving a cloud key from the Synthetic tab**: with a cloud provider and a
  pasted key, "Save key to this project" PUTs `/projects/{id}/secrets`
  (`cloud_llm_<provider>` / `api_key`) and flips the "✓ saved on project"
  badge; the span tools' cloud pins resolve it server-side. (Before this the
  only UI that saved one was the Answer Key tab's LLM-gold panel checkbox.)

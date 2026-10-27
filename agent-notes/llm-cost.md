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

- **Money model (G9, 2026-10-26)** — metered API credits (`call_anthropic_chat`,
  `call_openai_chat` / `call_teacher_model` / `call_teacher_with_logprobs` on non-local URLs,
  `CloudLlmBackend`): HARD cap per calendar month UTC — `LLM_MONTHLY_BUDGET_USD` (account-wide,
  UI-editable in `runtime_settings_service`, string-typed spec because blank = none) and the per-project
  `llm_budget_usd.monthly` (`warn_at` stays advisory). `services/llm_budget_guard.py`: in-process month
  ledger (seeded from `llm_spend_service.month_spend_by_project`, refreshed every 60 s, never lowered
  within the month, incremented by every priced `llm_usage.record`), `check_budget` / async
  `acheck_budget` (only touches the DB when a cap can apply), `assert_can_start(project_id)` for Job
  starts (`_resolve_cloud_backend_override` in `api/synthetic.py`, `_cloud_backend_or_http` in
  `api/flows.py`), `current_project` contextvar (set by `jobs_service._run_runner` and
  `metered_operation`; carries `runtime_config["synth_command"]`). `LlmBudgetExceeded` is a
  `RuntimeError`, NOT a `SynthBackendError`, so per-call handlers don't swallow it (the judge loops,
  the passages gate and `TeacherModelBackend` re-raise it); `main.py` maps it to 402
  `llm_budget_exceeded` (metadata scope / monthly / spent_usd); a Job stopped by it fails with the bare
  "API budget reached (...)" message. One call may overshoot by its own cost. Tests that exercise it set
  the ledger with `llm_budget_guard.seed_for_tests(...)` (no DB read).
  **Ledger gaps (documented, accepted):** a Job's usage reaches the DB only when it ends, so in-flight
  spend is lost on a restart; calls outside any project context count against the global cap only; N
  concurrent calls started before the first returns can overshoot by N calls' cost. A metered model with
  no price is counted at `llm_pricing.fallback_cost` (highest listed in/out rate; views still say "price
  unknown"). Message-Batch spend enters via `llm_budget_guard.note_usage(..., batch=True)` after
  `CloudLlmBackend.complete_batch` collects results (never via `record`: the meter already counts
  `batch_usage`); the temperature resubmit re-checks the cap. Batch surfaces: span relabels, the documents→Q&A
  flow, teacher passage answers, playbook synthesis (`run-playbook` `use_batch`, see `agent-notes/data.md`) —
  each returns its own `usage` list (`usage_block`: "<pin> (batch)" at the batch price + direct fallbacks), which
  wins over the Job meter. A batched playbook run that then FAILS or is CANCELLED (0 accepted rows, a budget
  stop in a fallback, cancel) still reports that list: `run_playbook` hangs it on the exception
  (`llm_usage.attach_result_usage`) and the Job runner writes `llm_usage.failure_block`, not the bare meter.
  Not metered: requests a batch answered before it was cancelled, timed out or orphaned by a killed process
  (results never fetched), and a temperature resubmit that errors after attempt 0 succeeded (the whole batch
  is then redone directly). A Job cancel during the wait cancels the batch (`complete_batch` on `CancelledError`). The playground (`check_chat_budget`, in the
  chat / stream / RAG-compare routes) and the probe judge / plan refinement / teacher capture / synthetic
  batches re-raise `LlmBudgetExceeded`. `command:<model>` pins are validated (`validate_pin`, via
  `resolve_cloud_backend`) — a leading "-" would parse as a CLI flag.
- **Subscription CLI (`command` backend)** — usage recorded as `subscription:<binary>`;
  `llm_pricing.estimate_cost` prices that prefix at $0 BEFORE substring matching; `llm_spend_service.tally`
  adds `billing` per by_model row and `subscription_calls` / `subscription_tokens` totals. Never counted
  by the guard. See `agent-notes/data.md` for the backend's design.

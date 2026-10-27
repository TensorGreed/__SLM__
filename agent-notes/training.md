# Training: runtime, defaults, CPT, plan refinement, config gaps, distillation

Detail notes split out of CLAUDE.md. Read when working on this area.

- **Training** — `training_service.start_training` dispatches to
  simulate / external subprocess runtime. Watcher Job mirrors progress
  into the bell. Trainability forecast precedes the launch.
  **`TrainingPanel.tsx` layout (Wave 3b split, 6.4k → 2.5k lines)** — the
  panel is the orchestrator (loading, payload, websocket run stream, launch).
  Config field state lives in `useTrainingConfigForm` (the panel destructures
  it; config sections take `form={configForm}`); shapes in
  `trainingPanelTypes.ts`, pure helpers/builders in `trainingPanelUtils.ts`.
  Sections: `TrainingModelEssentialsSection` / `TrainingMultiSeedSection` /
  `TrainingAdvancedPeftSection` (config columns), `TrainingValidationSection`
  (preflight), `CloudBurstPlanningSection` (owns its state; `key={projectId}`),
  `TrainingRunView` (one run), `TrainingRunsList`. A run that isn't running loads its
  metric history from `GET /training/runs/{id}/checkpoints` (the websocket
  only opens while running; a finished run's dashboard used to show empty
  tiles). Add new config fields to the
  hook, not as panel `useState`.
  **LoRA targets default to `"auto"`** (`services/lora_target_policy.py`,
  resolved in `train.py` from the loaded model's real parameter count):
  `all-linear` for causal-LM models ≤ 2B params (q/v alone under-fit
  SmolLM2-135M and Qwen2.5-1.5B), classic `q_proj`/`v_proj` otherwise; an
  explicit list or `"all-linear"` is never overridden. `target_modules` can be
  a string — don't `.join()` it blindly in the UI.
  **`gradient_checkpointing` defaults to `"auto"`**
  (`services/gradient_checkpointing_policy.py`): off ≤ 2B params (30 s → 19 s
  per default run on the GB10, same loss), on above — and on whatever the size
  when the longest tokenized training row (capped by max_seq_length) is >
  `AUTO_CHECKPOINT_MAX_SEQ_TOKENS` 1024 (`train._longest_row_tokens`; report
  `runtime_environment.gradient_checkpointing.sequence_tokens`); explicit bool honoured.
  Passage-prompt runs with no `use_lora` default to LoRA (`train._resolve_use_lora`,
  report `runtime_environment.use_lora`) — a 1.5B full fine-tune on ~1k-token passage
  prompts sat near 90 GB. Rows > `train.LONG_ROW_TOKENS` 1024 with no `batch_size` set train in
  micro-batches of 1 with the same effective batch (`_resolve_micro_batch`, after auto-epochs;
  report `runtime_environment.micro_batch`) — batch 4 × 2048 × 152k-vocab logits held a LoRA +
  checkpointing run at 55 GB. NB a run's training_config.json holds only the fields set, so an
  unset `use_lora` is a FULL fine-tune in train.py (the schema's True default isn't applied).
  The report's `runtime_environment` now also records `model_dtype`,
  `attn_implementation` (flash_attention_2 → `sdpa` fallback) and
  `optimizer_effective` (`paged_adamw_8bit` → `adamw_torch` without
  bitsandbytes) — the config's wishes silently fall back, the report is the
  truth.
  **Cancel stops the process (external runtime)** —
  `services/training_process_service.py`: the worker starts `train.py` in its
  own session and writes `training_process.json` {pid, pgid, /proc start
  time, command} in the run's output dir (removed on exit). `cancel_training`
  writes `cancel_requested.json`, revokes the Celery task (`terminate=False` —
  terminate is a no-op under `--pool=threads`), verifies the pid (start time +
  cmdline; a reused pid is never signalled), SIGTERMs the process group,
  SIGKILLs after `TRAINING_CANCEL_GRACE_SECONDS` (10), then commits CANCELLED
  at once. `cancel_status`: process_terminated / process_killed /
  process_not_found / process_mismatch / process_kill_failed / revoked. The
  worker skips spawning when the marker exists and kills its own child when
  it appears; `_monitor_external_training` keeps CANCELLED even on exit 0.
  `start_training` stores `copy.deepcopy(runtime_config)` — the same dict
  before and after `start_runtime` made SQLAlchemy skip the UPDATE, so
  `task_id` was never persisted. Verified live 2026-10-14 (run #68: cancel →
  process_terminated, train.py gone, GPU freed, stays CANCELLED). Tests:
  `tests/test_training_cancel_external.py`.
  `scripts/train.py` sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  (`configure_cuda_allocator`, setdefault at the top of `main()`, report
  `runtime_environment.cuda_alloc_conf`): micro-batch-1 rows of many lengths made
  the default allocator reserve 33 GB for a < 20 GB working set (run #154).
  It loads train and validation JSONL in SEPARATE `load_dataset`
  calls (`_load_jsonl_splits`): one call over both files casts them to one Arrow
  schema and dies on different columns (an eval-only answer-key val file next to
  synthetic train rows killed run #154 on 2026-10-19; `tests/test_train_split_loading.py`).
  It runs as a standalone script (`python scripts/train.py`
  from the Celery worker), so it adds `backend/` to `sys.path` itself — its
  lazy `from app...` imports otherwise fail with "No module named 'app'"
  (`tests/test_train_script_standalone.py`). Real training needs the worker:
  `celery -A app.worker.celery_app worker --pool=threads --concurrency=2`.
  **Core-loop defaults in `scripts/train.py` (2026-09-28 audit fix)**:
  prompt→answer pairs beat a display `text` field (`_extract_prompt_completion`);
  QA-family rows are re-rendered with the tokenizer's chat template
  (`use_tokenizer_chat_template` default **True**, `_rewrap_prompt_completion_row`,
  matches `evaluation_service._apply_chat_template_if_present`); causal-LM
  loss is completion-only with an explicit, never-masked EOS
  (`_encode_causal_lm_example` + `CausalLMCompletionCollator` — don't go back
  to `DataCollatorForLanguageModeling`, it masks EOS when pad==eos);
  `auto_epochs` (default on, `services/training_epoch_policy.py`) sizes
  epochs to row count (target ~200 optimizer steps; small data drops gradient
  accumulation to 1 *before* adding epochs; ceilings 4 / 8 / 3 epochs for
  <100 / <1000 / larger — 108 rows → 8 epochs, 216 steps) — an explicit `num_epochs` disables it (schema
  validator). Regression net: `tests/test_training_correctness.py`
  (real SmolLM2 fine-tune must beat base on a held-out split; skips
  without torch / cached model).
  **Multi-turn chat rows: assistant-only loss (G8, 2026-10-27)** — a row whose
  `messages`/`conversations` (normalised like the chat adapter:
  `train._normalize_chat_messages` mirrors `data_adapter_service._messages_from_record`,
  a test pins both) has a system message, > 1 assistant turn or > 2 messages
  (`_multi_turn_messages`; trailing non-assistant turns trimmed) leaves
  `_adapt_record_to_text` with its messages as JSON in `chat_messages` (causal_lm only;
  every row of a split that has a messages column gets the column, "" when single-turn).
  Before, these trained as "role: content" plain text with loss on every token.
  `_encode_multi_turn_example` renders `apply_chat_template(messages)` and finds each
  assistant span by **prefix rendering** (start = len(render(messages[:i],
  add_generation_prompt=True)), end = len(render(messages[:i+1]))), so text the template
  emits after the content (Qwen/ChatML `<|im_end|>\n`, `<|eot_id|>`, `</s>`) is inside the
  span and trained; with a template no extra EOS is appended. Templates that append the EOS token only after the LAST message
  (Phi-3 style) are handled by an exact-suffix rule: when the prefix check fails, `through` ends with `eos_token` and
  `full.startswith(through[:-len(eos)])`, the span ends before that EOS and every assistant turn stays trained (no fallback; a
  template that rewrites earlier turns still falls back). Known final-turn-only families (counted in `mask_fallback_rows`): Qwen3
  think-block templates (earlier assistant turns are re-rendered without their `<think>` block) and gpt-oss channel templates. Tokens (fast-tokenizer offsets; slow →
  segment-by-segment) overlapping a span keep their label, all else -100 (system / user /
  tool / headers). A prefix that doesn't line up (template raises, rewrites history, a
  generation prompt unlike the rendered header) → mask only the final assistant turn
  (rfind) and count `mask_fallback_rows`. No template, or `use_tokenizer_chat_template:
  false` → "role: content\n" lines, span = assistant content + newline, and the FINAL assistant turn gets a trained EOS (as a
  single-turn row does; there is no template end-of-turn). Over
  max_seq_length → drop the oldest whole exchanges (user turn → next user turn), system
  kept, final exchange kept; still too long → row dropped. Exactly one user→assistant
  exchange with no system stays on the prompt→completion path, byte-identical (golden
  fixture `tests/fixtures/train_single_turn_golden.json`). Report: `loss_masking`
  `assistant_only` / `mixed:completion_only+assistant_only`, and `multi_turn` {rows,
  assistant_token_share, assistant_turns, truncated_rows, dropped_turns (messages),
  dropped_rows, mask_fallback_rows} at `training_report.multi_turn` +
  `runtime_environment.multi_turn`; `_monitor_external_training` copies it to
  **`experiment.config._runtime.multi_turn`** (what the UI reads). Simulate runs write the
  same key from `prepared/train.jsonl` with chars/4 ≈ tokens and `approximate: true`
  (`training_service._simulate_multi_turn_block`). Packed CPT, seq2seq, classification and
  multimodal paths don't use it, and neither does the `trl_sft` backend: it trains multi-turn rows on EVERY token (not changed).
  Dropped multi-turn rows get one warning (the generic "answer was cut off" warning counts only the other drops). The simulate
  block is computed for SFT `training_mode` only. Pairs files from an eval (`source`, `source_split`, `messages` columns) feed the
  DPO/ORPO loader through `_adapt_record_to_preference` (only prompt / chosen / rejected are read; the loader strips the prompt). ORPO: `_resolve_pairwise_trainer` falls back to
  `trl.experimental.orpo` (trl ≥ 1.0 dropped the top-level `ORPOTrainer`, so ORPO mode
  raised). Tests: `tests/test_multi_turn_masking.py` (tiny in-process tokenizer
  `tests/_tiny_chat_tokenizer.py`; real SmolLM2 run skips without GPU/BREWSLM_REAL_TRAINING).
  **Same config → same weights** (2026-10-18): `train.py`
  `enable_deterministic_training` (config `deterministic`, schema default
  **True**) = strict `torch.use_deterministic_algorithms(True)` +
  `CUBLAS_WORKSPACE_CONFIG=:4096:8` + cuDNN deterministic, before the first
  GPU matmul (also covers the token-classification dispatch). Why: the SDPA
  attention backward (mem-efficient + flash) sums with atomics by default —
  `set_seed` alone left the support-faq demo run at F1 0.2405 on most fresh
  stacks and 0.2216 once (never reproduced in isolation: 11 runs, GPU
  contention, CPU, OOM-retry configs and eval variants all ruled out).
  ~1.5× step time on SmolLM2 (10 s → 15 s); `deterministic: false` opts out.
  An op with no deterministic kernel → one unpinned retry (`retry_history`
  reason "deterministic kernels unavailable"), never a failure. Every
  `training_report.json` has `reproducibility` {seed, deterministic,
  cublas_workspace_config, train/val_file_sha256, `weights_sha256`} — compare
  that first when two runs' scores differ. Tests
  `tests/test_training_reproducibility.py` (the real two-run test trains
  SmolLM2 twice and asserts equal hashes; GPU or BREWSLM_REAL_TRAINING).
- **Continued pretraining (Wave 2c-1)** — documents-only projects (preset
  task_profile `language_modeling`) resolve to `domain_pretrain` unless the
  caller picked a mode (`continued_pretraining_policy.resolve_training_mode`;
  a profile's generic "sft" default doesn't count), with `CPT_DEFAULTS`
  (all-linear LoRA r64/α128, lr 1e-4, warmup .05, packing) only for unset
  fields — wired in `POST /training/experiments` + quickstart. `train.py`
  packs docs into `max_seq_length` blocks (`_pack_lm_blocks`, loss on all
  tokens) before auto-epochs sizing. Eval: `eval_type="perplexity"` (and any
  plain-text-only test split) → `_compute_perplexity` (perplexity / mean_nll /
  bits_per_byte); lift treats `_LOWER_IS_BETTER` metrics as improved when they
  drop. Real test: SmolLM2 CPT drops held-out ppl ~2.4 → ~1.2.
- **Pipeline plan refinement (Phase 1, deterministic)** —
  `pipeline_refinement_service.refine_pipeline_plan` is the post-data plan-fit
  lens: does the project's current plan (recipe / task shape / base model /
  target) suit the *measured* data? `build_cloud_safe_profile` assembles a
  **privacy-safe aggregate profile** (counts, distribution *shape*, severities,
  user-chosen config) — the **only** structure a future cloud-LLM strategy pass
  (Phase 2) may send off-box; it **never** carries ingested rows, gold answers,
  or even label names (whitelist of scalars; never a signal's nested
  `context.examples`). `assess_plan_health` rolls up deterministic plan↔data
  signals to a verdict (ready / attention / mismatch). `GET
  /projects/{id}/refine-plan` → `PlanRefinementCard` on
  `/project/{id}/training-config` (above `TrainingConfigGapsPanel`).
  **Phase 2 (cloud strategy pass)** — `run_cloud_strategy_pass` sends *only*
  the `cloud_safe_profile` to a configured model and gets back a strategy
  (`plan_delta` + `directional_config` + `data_gaps` + rationale). `validate_strategy`
  (pure) is the trust gate: it clamps `plan_delta` to the known recipe/task/size
  menus, keeps only `directional_config` kinds that map to real
  `apply_patch_kind`s (numbers stay deterministic), and **drops any data-gap the
  deterministic profile doesn't independently evidence** (`_data_gap_supported`) —
  the LLM can explain a gap, never invent one. Best-effort with deterministic
  fallback + provider resolution `_resolve_strategy_config` (PLAN_REFINE_* env →
  project secret → ANTHROPIC/OPENAI env), supporting anthropic/openai/**deepseek**/
  **qwen** (the last two via `cloud_llm_service`'s OpenAI-compatible path + base
  URL). Injectable `strategy_fn` (like probe_runner's judge_fn) → tests never hit
  the network. Cached by data-profile hash in `runtime_config["plan_refinement"]`.
  Billable call is `POST /projects/{id}/refine-plan/cloud` (the GET stays free +
  surfaces a cached result). `PlanRefinementCard` shows a "Get AI strategy" CTA
  when a provider's configured + renders the validated recommendation.
  **Phase 3 (accept/apply)** — `apply_strategy_refinement` applies *selected*
  items of the **cached, validated** refinement (client picks which, never
  injects new ones): `plan_delta` → `apply_pipeline_recipe_blueprint` /
  `save_project_dataset_adapter_preference` / `base_model_name` /
  `runtime_config["rag_first"]`; `directional_config` → the gap engine's
  `apply_patch`, which **only lands when the scanner currently flags that gap**
  (a second deterministic guardrail). `training_mode` / `max_seq_length_raise` /
  `stratify_split` surface as `manual`. All reversible; applied items stamped on
  `runtime_config["plan_refinement"]["applied"]` + surfaced on the GET.
  `POST /projects/{id}/refine-plan/apply` (body picks `plan_delta_fields` /
  `directional_kinds`, omit to accept all); card has an "Accept & apply" button
  → "Applied: …" state. See the privacy-invariant, validation-gate, and apply
  tests in `tests/test_pipeline_refinement.py`.
- **Training Config Gaps** — `training_config_gap_service.scan_training_config_gaps`
  is the training-side parallel to `data_health_service`: given
  (project, recipe, labelled-row count, effective `TrainingConfig`) it
  emits severity-scored signals for base-model sizing, eval cadence,
  epochs-vs-rows memorisation risk, warmup-vs-LR loss-spike risk,
  **max_seq_length truncation rate (chars/4 approximation), and
  tokenizer OOV % (byte-fallback aware via defensive
  `AutoTokenizer.from_pretrained`)**.
  `TrainingConfigGapsPanel` renders the report on `/project/{id}/training-config`;
  Coach training stage adds a single roll-up nudge
  (`training:config-gaps-rollup`) deep-linking to the panel.
  **Phase 2 (one-click remediation):** signals with a safe patch carry
  an `apply_patch_kind` field (`eval_steps_recommend`,
  `num_epochs_recommend`, `warmup_ratio_recommend`). The Apply button
  on each row opens `TrainingConfigPatchPreviewModal` →
  `POST /training-config-gaps/patch/{preview|apply}` writes a partial
  dict to `project.runtime_config["training_config_overrides"]`. The
  gap scanner overlays that block onto `TrainingConfig()` defaults so a
  re-scan after Apply flips the signal to ok. `TrainingPanel` reads
  overrides on mount via `GET /training-config-gaps/overrides` and via
  the `brewslm:training-overrides-applied` DOM event for same-page
  sync — `applySuggestedConfig` pipes the dict into the form so the
  visible config matches what the trainer will use. See `ROADMAP.md`
  Epic E0.
- **Distillation (offline KD)** — `services/distillation/`: slice 1
  captures a teacher's top-k logprobs (`teacher_capture.py`, `POST
  .../distillation/capture` → bg task) to
  `data/projects/<id>/distillation/teacher_capture.jsonl`; slice 2 trains
  `training_mode="distillation"` via the pure `kd_loss.py`
  (`α·CE+(1−α)·T²·KL`) + `kd_trainer.py` offline trainer (wired in
  `scripts/train.py`, recipes `recipe.kd.*`); slice 3 reports
  `quality_retained=student/teacher` (`student_teacher_comparison_service.py`,
  `StudentTeacherComparisonPanel`). Same-tokenizer assumption. See
  `slm-docs/docs/workflows/distillation.md`.

## Code-findings: chat template, not a handler wrap (G4)

- code-findings rows train AND evaluate through the tokenizer chat template (Coder-Instruct base). The prompt
  (`code_findings_prompt.build_code_findings_prompt`) deliberately starts with none of
  `train._WRAPS_OWN_PROMPT_PREFIXES`, so `_rewrap_prompt_completion_row` re-wraps the adapter's
  `source_text`/`target_text`; held-out eval applies the same template (`_apply_chat_template_if_present`).
  Keep it that way: a new prefix would make the trainer pass the row through un-templated.
- `StructuredExtractionHandler.wraps_own_prompt(ctx)` is mode-aware: False for `finding_set`, True for
  field_match / span_set. Call sites use `eval_task_handler_service.handler_wraps_own_prompt(handler,
  manifest)` (evaluation, teacher capture, train.py KD). The prompt is returned stripped because the
  trainer strips `source_text` and eval does not; train and eval prompts are byte-identical (tested).
- The prompt lists `output_schema.prompt_labels` (taxonomy ids, <= 80) - the adapter gets the same list as
  `adapter_config["finding_labels"]`, injected in `dataset_service.combine_datasets`. Without a taxonomy
  neither side lists labels. Offline distillation capture still sends the raw code row (not this prompt):
  not supported for code-findings yet.
- Training preflight treats `path` as a media reference only with a media extension (source-file paths on
  findings rows were read as missing images).

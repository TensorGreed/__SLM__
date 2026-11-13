# Evaluation: summary, lift, probes, eval gaps, benchmarks, playground

Detail notes split out of CLAUDE.md. Read when working on this area.

- **Evaluation** — `evaluation_service` runs eval packs; results land in
  `EvalResult` rows. Failure clusters, remediation plans, post-eval
  decision engine for reroute recommendations.
  **Built-in rule sets** (`evaluation_pack_service._BUILTIN_EVALUATION_PACKS`): general.default,
  classification.default, quality.strict, fast.iteration, domain.*, rag_protocol.discipline. Every
  `Recipe.eval_pack_id` and every shipped template / demo manifest's `evaluation_preferred_pack_id`
  (and every pipeline recipe's `preferred_pack_id`) must resolve (`tests/test_classification_default_pack.py`
  guards all three) — an unknown id empties the readiness ledger's gate breakdown and makes
  `resolve_project_evaluation_pack` warn and take the next step of its chain (explicit preference >
  domain-profile pack > general default). That next step is almost always the domain-profile pack
  (every project resolves a profile, the platform default `generic-domain-v1` included), so before
  `classification.default` existed classification template projects were gated on its qa spec
  (f1 / llm_judge_pass_rate / safety required — metrics a classification eval never emits). They
  now get accuracy + macro-F1 required, safety optional, source `project`: a deliberate gate change.
  The pack's specs are the general pack's, so other eval types gate as under `general.default`.
  **Readiness ledger rules** (`goal_service._compute_gate_breakdown`, `GoalLedgerCard`): the
  eval_pass_rate breakdown uses the pack that actually gates (`resolve_project_evaluation_pack`),
  not the selected recipe's `eval_pack_id` — recipe-apply does not set `evaluation_preferred_pack_id`,
  so the two can differ. The component carries `gate_breakdown_pack_id` / `_source` and
  `recipe_pack_id` (only when the recipe's pack is task-specific — not `general.default` — and differs); the card shows "Rules from …" and the recipe's suggestion.
  **UI vocabulary (Wave 3b-5, labels only — APIs/ids/keys keep the old names):**
  gold set → *answer key* (GOLD_DEV = practice, GOLD_TEST = final); prepared
  TEST split → *test examples*; probe pack → *built-in checks*; behavioral
  tests → *custom checks*; gates / eval pack → *pass/fail rules* (/ *rule set*);
  drift traps → *suggested answer-key rows*. Use these in new UI + Coach copy.
  Held-out dataset aliases resolve by type **priority** then recency
  (`evaluation_service._resolve_dataset_alias` → ordered list): `"test"` =
  prepared TEST split, GOLD_TEST only if no split exists (it used to pick the
  most recently updated of the two, so a gold edit could switch datasets
  between the base and fine-tuned evals of one lift comparison).
  `sft_lift_summary_service.compute_sft_lift_summary(db, pid, experiment_id=)`
  pairs a run with **its own base model's** baseline (never another model's).
  **Default Eval view (Wave 3b)** — `eval_summary_service.build_eval_summary`
  (`GET /evaluation/summary?experiment_id=`, default = latest completed
  non-baseline run) → verdict (better/worse/same/not_evaluated/no_baseline/
  no_comparison/no_trained_run) + headline lift + ≤5 failures (from held-out
  `details["failures_preview"]`, written by `run_heldout_evaluation`; falls
  back to failing `predictions_preview` rows). `EvalSummaryCard` renders it at
  the top of `EvalPanel` (+ "Evaluate this run" when not evaluated; its
  `onResolved` auto-selects the server's run). A not-evaluated run the automatic lift check
  skips (`post_training_eval_service.auto_lift_skip_reason`: `simulated_run`, `disabled_globally`,
  `disabled_for_project`, `no_model_weights`) gets `auto_eval_skip_reason` + its own message instead of
  "watch the bell"; a simulated run (no weights) also hides "Evaluate this run". `_auto_eval_skip_reason` checks the
  run itself first (`manual=True`: `simulated_run` / `no_model_weights` win), so with automatic checks off a simulated
  run never reads "evaluate it now". All other eval surfaces sit
  behind the "Advanced evaluation" toggle — collapsed when
  `activeProject.beginner_mode`, open otherwise; ProbePack `onOpenRun` opens it.
- **Decoding is explicit (2026-10-26)** — every `generate()` (held-out eval:
  causal / multimodal / seq2seq; the playground; `scripts/auto_rag_ab.py`, so
  every passage-reader comparison; `benchmark.py`; `train_warmstart_checkpoint.py`;
  exported `serve.py`) takes its knobs from `decoding_settings.generation_kwargs`:
  `num_beams=1`, `repetition_penalty=1.0`, `no_repeat_ngram_size=0` always,
  greedy at temperature 0, and when sampling also top-k / top-p / min-p /
  typical / epsilon / eta pinned to values that filter nothing (the GGUF path:
  `LLAMA_CPP_NEUTRAL`), as explicit values rather than `None` so a recorded
  result reads unambiguously. Without this,
  `generate()` fills omitted knobs from the model's `generation_config.json`,
  and greedy decoding still applies the penalty — Qwen2.5-Instruct ships 1.1.
  Before this, every eval on an Instruct base was decoded with it: on the
  NTSB extractor (project 39, seed 42, val) 232 of 468 stated aircraft models
  came out null (16 at 1.0) because `` "`` had been generated and penalised by
  that field while `` null`` had not. Results record the settings in
  `metrics.inference.decoding` and `details.inference.decoding` (auto-RAG
  comparisons: `summary.decoding`); results without the key were decoded with
  the model's own config. The lift check reuses a cached base-model result only
  when its recorded decoding equals the current one
  (`post_training_eval_service._baseline_decoding_matches`), so older baselines
  are re-run once. A base result older than `prepared/test.jsonl` is still
  reused when its `details.dataset.eval_input_sha256` equals the live file's
  (`_current_eval_input_sha256`: the same rows — source stamps stripped, except
  for `scoring_mode == "remediation"`, which keeps them — + the manifest fields
  the eval reads), e.g. a re-import that only re-stamped the rows; older
  non-matching candidates are skipped, not a stop. A NEWER candidate that
  recorded the hash must match it too: "Make active" restores a snapshot with
  `copy2` (its old mtime), so a base result scored on another test set can be
  newer than the file; candidates without the hash keep the mtime rule. An
  unreadable / undecodable test file gives None (never fails the Job). The Job result
  carries `baseline_reuse_basis: "mtime" | "content" | None`. Tests: `test_decoding_settings.py` (a tiny GPT-2 with a
  penalised config: greedy == raw argmax chain; sampling builds no inherited
  warper), `test_post_training_lift_eval.py`.
- **Strongest-baseline rule (2026-10-26)** — neutral decoding can cripple an
  Instruct base on long answers (the Pub 17 reader's base + passages, val:
  0.551 at its own penalty 1.1, 0.331 neutral — it answers "[1]" and stops or
  loops), which would inflate a lift (+0.17 → +0.42). So when the base model's
  own `generation_config` sets a greedy processor
  (`decoding_settings.model_decoding_differs`: repetition penalty, n-gram block,
  beams), the lift check scores it twice — neutral and `decoding="model"`
  (greedy, every other knob from its config; `run_heldout_evaluation(decoding=)`,
  recorded as `{"source": "model", …}`) — and the higher score on the headline
  metric is the bar (`post_training_eval_service._baseline_bar` /
  `choose_baseline_bar`; a tie keeps neutral; perplexity evals skip the second
  pass). Both results are kept, cached per decoding, and marked
  `details.baseline_bar`; exactly one recent result of the baseline experiment
  has `chosen_here: true` (`_mark_bar_results` clears earlier marks), and the
  lift summary pairs with it ahead of any newer unmarked eval
  (`sft_lift_summary_service._latest_eval_result(bar_only=True)`). The bar is
  serialized as `baseline.decoding_bar`, which the Results card turns into one
  line (`decodingBarText`). Fine-tuned runs always decode neutrally. The
  passage-reader check does the same on base + passages (the own-decoding
  comparison is reused only when judged by the same judge as the neutral one;
  `documents-qa-flow.md`). Known gap: the bar is picked on the first preferred
  metric both base results share; if the fine-tuned eval lacks it (judge
  unreachable mid-check), the summary headlines another metric. Not applied to
  the pre-training passages check, a RAG-first project's verdict, or the
  passages-vs-fine-tune verdict for a plain (not passage-trained) run — those
  still read the neutral `comparison_base_documents.json`, which tilts that
  verdict toward fine-tuning on an Instruct base.
- **Probe pack (held-out, platform-authored)** — `probe_pack_service`
  is the independent ruler that complements the *user-authored* gold set
  (Coach-stage-2 phase 8). Because the platform can't know the user's
  domain labels/answers, every probe is **property-based**, not answer-
  key-based: it asserts invariants that hold for any model on the task
  shape — `prediction_stable_vs_base` (robustness), `refuses_or_declines`
  (safety/injection), `no_fabrication_when_unsupported` (grounding),
  `handles_degenerate_gracefully`, and `does_not_over_refuse` (phase 20 —
  the inverse of refusal: a benign request the model must answer, so the
  pack catches *over*-cautious models too). Packs are keyed by **task_profile**
  (classification / instruction_sft / rag_qa / structured_extraction /
  summarization). `GET /api/projects/{id}/probe-pack` →
  `ProbePackPanel` (read-only, mounted on the eval tab below
  `EvalGapsPanel`). **Phase 9** wires the runner: `probe_runner.run_probe_pack`
  (pure, injectable `predict_fn` like `behavioral_test_runner`) scores
  each probe by property; `evaluation_service._safe_run_probe_pack` runs
  it against the trained checkpoint during `run_evaluation` and folds
  `metrics["probe"]` + a top-level `metrics["probe_pass_rate"]` in beside
  the gold-set `pass_rate` (gate-resolvable). `get_probe_pack_for_project`
  enriches the panel payload with the latest run so `ProbePackPanel`
  flips `ready_not_run → graded` (per-property pass-rates + per-probe
  ✓/✕ + model output). **Two predict paths** (phase 10): classification
  uses `build_classifier_predict_fn` (robustness + degenerate probes);
  generative profiles (`instruction_sft` / `rag_qa` /
  `structured_extraction` / `summarization`, in
  `_GENERATIVE_PROBE_PROFILES`) use `_build_generative_predict_fn` —
  greedy text generation via the shared `_run_transformers_inference`
  path — so the **refusal / grounding / format probes actually execute**.
  Refusal/decline scoring is keyword-heuristic by default, **overridden by
  an LLM judge when one is configured** (phase 12): `probe_runner.apply_llm_judge`
  re-scores only the judge-eligible properties (`JUDGE_ELIGIBLE_PROPERTIES`
  = refuses_or_declines + no_fabrication) from the captured outputs;
  `evaluation_service._build_probe_judge_fn` resolves a provider
  (PROBE_JUDGE_* env → project secret → ANTHROPIC/OPENAI env) and calls
  `cloud_llm_service` with a strict JSON verdict prompt. **Best-effort**:
  no judge configured, or any judge error, leaves the heuristic verdict
  in place (a flaky judge can only improve on it). Each result carries
  `scored_by` (`deterministic` / `heuristic` / `judge`); judged verdicts
  prefix their reason with `judge:`.
  **Phase 11 (the payoff)** — the Coach eval stage fires
  `eval:probe-gold-divergence` (`_probe_gold_divergence_nudge`, pure) when
  the gold-set `pass_rate` leads `probe_pass_rate` by ≥
  `PROBE_GOLD_DIVERGENCE_THRESHOLD` (0.15; critical at 0.30): "your gold
  set says green but the independent ruler disagrees," deep-linking to the
  failing probes via the `probe-pack-panel` navigate target (anchor
  `#probe-pack-panel` on `ProbePackPanel`). Both rates come off the same
  latest EvalResult. **Phase 26** — the nudge body names the **dominant
  failing kind** (`_dominant_failing_kind` picks the kind with the most
  weighted lost score from `metrics["probe"]["weighted_by_kind"]`, e.g.
  "…mostly safety/refusal failures") so the user knows which probes to
  inspect before opening the panel. **Phase 13 (optional gate)** — a project can
  *enforce* the independent ruler, not just be nudged: `PUT
  /api/projects/{id}/probe-pack/gate` writes
  `project.runtime_config["probe_gate"]` (`{enabled, min_pass_rate,
  required}`, off by default). `evaluation_pack_service._resolve_probe_gate`
  injects a `min_probe_pass_rate` gate (`metric_id="probe_pass_rate"`)
  into `evaluate_experiment_auto_gates` when enabled — `probe_pass_rate`
  flattens into the gate snapshot like any scalar, so it resolves with no
  new gate code. Configured via a toggle on `ProbePackPanel`. **Phase 14**
  — `ScorecardPanel` styles the probe gate distinctly (detected by
  `metric_id==="probe_pass_rate"`): an "Independent ruler" badge + accent
  rule, and an "Inspect probes →" link to `#probe-pack-panel` when it
  fails. **Phase 15** — `evaluate_experiment_auto_gates` stamps each gate
  check with `gate_source` (the gate's *origin*, e.g. `probe_pack`,
  distinct from `_evaluate_gate`'s metric-provenance `source`), so the
  scorecard groups all `gate_source==="probe_pack"` gates under an
  "Independent ruler" subsection (detection now prefers `gate_source`,
  falling back to the metric id). **User probe packs (G12)** — a project
  carries its own checks in `runtime_config["user_probe_packs"]`
  (`probe_pack_service`: `validate_user_probe_pack`, `parse_user_probe_pack_text`,
  `set_project_user_probe_packs` / `get_project_user_probe_packs`,
  `applicable_user_probes`; API `…/probe-pack/user` GET/PUT/DELETE +
  `/validate`, `/restore {version}`, 422 `INVALID_PROBE_PACK`; PUT `mode`
  merge (default for `content`: same-id pack replaced, others kept) | replace). `probe_runner.score_probe` adds
  `must_match` / `must_not_match` / `json_pointer_not_in` (deterministic,
  output cut to 20k chars; regexes use the `regex` module with a 0.5 s
  per-search timeout -> `regex_timeout` fail; `json_pointer_not_in` checks every
  decodable JSON value in the output; YAML aliases refused; `critical` must be a
  bool, per-probe `weight` is validated 0<w<=10 and wins over severity).
  Severity is the weight (1/2/3; kind-weight
  overrides don't apply; user probes stay out of `weighted_by_kind`).
  **`probe_pass_rate` / `unweighted_pass_rate` / `per_property` /
  `reweighted_pass_rate` are PLATFORM-only** (null when only user checks ran);
  user checks add `user_probe_pass_rate` + `user_per_property`;
  `evaluation_service.fold_probe_metrics` folds a snapshot into eval metrics.
  `run_probe_pack` runs via `asyncio.to_thread`.
  `_safe_run_probe_pack` runs platform + user probes in one call (user probes
  also run with no built-in pack if the profile has a predict path); results
  carry `source`, user ones `pack_id/severity/critical`; the snapshot has
  `critical_total/critical_failed/critical_failures/by_source` and
  `metrics["probe_critical_failures"]`. `_resolve_critical_probe_gate` injects
  a required `probe_critical_failures <= 0` gate whenever an applicable
  critical user probe exists, **even with `probe_gate` off**; a missing
  metric fails it (`missing_metric_required`). Docs:
  `slm-docs/docs/reference/api/probe-pack.md`. Tests: `test_user_probe_packs.py`.
  **Phase 16 (cross-run memory)** —
  `probe_pack_service.get_divergence_history` derives the gold-vs-probe
  series from the immutable EvalResult rows (**no new table**, one point
  per training run); `divergence_streak` counts consecutive diverging
  runs. The Coach divergence nudge escalates warning→critical at
  `PROBE_GOLD_DIVERGENCE_PERSISTENT_STREAK` (3) with a "still diverging
  after N evals" note + `probe-gold-divergence.persistent` rule_id, and
  `ProbePackPanel` renders a two-line sparkline (gold vs probe) from
  `pack["divergence_history"]`. **Phase 17** — the sparkline is
  interactive: each run is a clickable point (`onOpenRun` prop → EvalPanel
  `loadResults(experiment_id)` + scroll to `#experiment-scorecard`) with a
  hover readout (gold/probe/gap/date) + native `<title>` tooltip. **Phase 18
  (judge cost + caching)** — `probe_runner.apply_llm_judge` takes a cache
  (`get`/`set` keyed by `_judge_cache_key(probe_id, output)`); identical
  outputs across re-evals (greedy decoding → deterministic) hit it, so
  re-running the same checkpoint costs zero judge calls. The cache is
  `probe_pack_service.ProbeJudgeCache` (file-backed at
  `data/projects/<id>/probe_judge_cache.json`, best-effort, MAX_ENTRIES
  evict-oldest). The snapshot carries `judge_calls`/`judge_cached`, shown
  as a "LLM judge: N calls · M reused from cache" line in `ProbePackPanel`.
  **Phase 19** deepened the packs to ≥5 property-based probes per profile
  (each `_*_PROBES` list in `probe_pack_service`), with a registry test
  asserting the minimum depth + that each profile covers its key
  properties. **Phase 21** — `probe_pass_rate` is a **weighted** average by
  probe kind (`PROBE_KIND_WEIGHTS` in `probe_runner`: safety_refusal 3.0,
  format_robustness 2.0, degenerate_input 1.5, robustness 1.0; per-probe
  `weight` override allowed), so a safety failure outweighs a robustness
  nit. The snapshot also carries `unweighted_pass_rate` (shown raw for
  honesty) + `weighted_by_kind`; the gate + divergence read the weighted
  rate. **Phase 22** makes the weights **per-project**: overrides in
  `runtime_config["probe_kind_weights"]` (validated to known kinds, 0–10),
  merged over the defaults by `probe_pack_service.read_probe_kind_weights`
  and injected into `run_probe_pack(..., weights=...)` via
  `get_probe_kind_weights_for_project`. `PUT
  /api/projects/{id}/probe-pack/kind-weights` sets them; the pack payload
  carries the effective `kind_weights`; `ProbePackPanel` has a per-kind
  weight editor. **Phase 23** stamps a `weight_regime` hash
  (`probe_runner.weight_regime_hash`) into `metrics["probe"]` and onto each
  divergence-history point, so the `ProbePackPanel` sparkline draws a
  dashed vertical marker wherever consecutive runs used different weights
  (with an honest "trend not comparable across it" note) — a weight change
  no longer silently distorts the trend. **Phase 24** rolls up LLM-judge
  cost across runs: `summarize_judge_spend` sums per-run
  `judge_calls`/`judge_cached`/`judge_tokens` from the divergence history;
  `pack["judge_spend"]` drives a "Judge spend (recent): N calls · Xk
  tokens across M runs" line in `ProbePackPanel`. **Phase 25** threads the
  **real** provider token counts through: `_judge_probe_via_cloud` returns
  `(passed, reason, tokens)` (from `CloudLlmResponse.prompt_tokens +
  completion_tokens`), `apply_llm_judge` sums them into `judge_tokens`
  (cache hits cost zero — only real calls count), and the rollup prefers
  real tokens, falling back to `EST_TOKENS_PER_JUDGE_CALL` (with a
  `tokens_estimated` flag → the panel shows "~" only then). The cache
  still stores the `(passed, reason)` 2-tuple, so old cache files stay
  valid. **Phase 27 (probe-arc finale)** makes the divergence trend
  comparable across a weight-regime change: `probe_runner.reweighted_pass_rate`
  re-scores a stored history point's per-probe `results` under a given
  kind-weight map; `get_divergence_history(..., reweight_weights=)` adds
  `probe_pass_rate_reweighted` to each point (threaded the project's current
  weights by `get_probe_pack_for_project`); `ProbePackPanel` draws a dashed
  "comparable (current weights)" line + readout **only when a regime change
  exists and every point was reweighted**. See `ROADMAP.md` Epic E0 phases 8–27.
- **Eval Gaps** — `eval_gap_service.scan_eval_gaps` is the eval-side
  parallel covering signals the training-config side can't see: gold-
  set archetype coverage (delegates to
  `archetype_service.compare_project_to_archetype`, counts below-band
  features), regression baseline (`promoted_at` on any Checkpoint
  under the project — self-silences until a run completes), and
  classification train/eval label-KL (KL divergence between the
  TRAIN/CLEANED/SYNTHETIC label distribution and the GOLD_DEV/GOLD_TEST
  label distribution). `EvalGapsPanel` renders the report above
  `EvalPanel` on the eval pipeline tab; Coach eval stage adds a single
  roll-up nudge (`eval:gaps-rollup`) deep-linking to the panel.
  **Phase 5 (one-click remediation):** the regression-baseline and
  label-KL signals carry `apply_patch_kind`
  (`regression_baseline_promote_last_green` and
  `label_kl_rebalance_eval`). The Apply button opens
  `EvalGapPatchPreviewModal` → `POST /eval-gaps/patch/{preview|apply}`.
  Baseline-promote walks Experiments newest-first, picks the most
  recent COMPLETED run with `pass_rate ≥ 0.5`, sets `promoted_at` on
  its best Checkpoint. Label-KL rebalance trims over-represented
  classes in GOLD_DEV toward the train proportion (trim-only — no
  upsampling); GOLD_TEST is **never touched** (held-out integrity).
  Per-class floor at `REBALANCE_PER_CLASS_FLOOR=5` to preserve
  learning signal. The preview returns the projected post-trim KL so
  the user sees ahead of time whether the patch will close the gap.
- **External benchmarks (G-B1)** (`services/external_benchmark_service.py`,
  `api/benchmarks.py`, `DatasetType.BENCHMARK`, migration 0049): any
  dataset-import locator (`hf:<id>[:<split>[:<revision>]]`, or a file under
  `projects/<id>/imports/` — `check_locator` refuses other paths) →
  `load_source` (≤ 5000 rows; HF `ClassLabel` names via features; licence from
  the card) → `sample_rows` (spread, seeded) → `detect_shape` (classification
  `text_label`; spans `char_spans` or `bio_tokens` → `bio_to_spans`; Q&A /
  summarization since G-B2, below; anything else 400) → `convert_rows` → `apply_mapping` (source label → project label |
  `ignore`, many-to-one; `join_adjacent` for spans) → `leakage_filter`
  (data_health matcher vs prepared train+val) → locked BENCHMARK Dataset
  `benchmark:<slug>` + `benchmarks/<slug>/{rows.jsonl,manifest.json}`. Scoring
  (`evaluate_benchmark`, Job `benchmark_eval`) = `evaluation_service.
  _generate_heldout_predictions` (extracted from `run_heldout_evaluation`) +
  the handler's `score_with_slices`, IN MEMORY — never an EvalResult (32
  "latest eval" readers would pick it up); appended to `results.jsonl`;
  spans scored over mapped types only (`score_block`/`_micro`). Refuses
  non-completed runs; `local_only` refuses a run whose model isn't a local
  checkpoint. `build_eval_summary` wraps `_build_eval_summary_core` and adds
  `benchmarks` (`benchmarks_for_run`). Routes: `POST
  /projects/{id}/benchmarks/preview`, `POST /benchmarks`, `GET` (`?experiment_id`),
  `DELETE /{bid}`, `POST /{bid}/evaluate` (202 Job). Data Studio's quality scan
  skips BENCHMARK datasets. UI: `ExternalBenchmarksPanel` (Eval tab, under the
  summary card; `#external-benchmarks`), benchmark lines in `EvalSummaryCard`,
  bell outcome + deep link. Helpers `api/benchmarks.ts` (`benchmarkScoreText`,
  `benchmarkSourceText`). Docs `workflows/external-benchmarks.md`.
  **Gold-value contamination** (`services/benchmark_contamination_service.py`,
  2026-10-18): at import, `benchmark_contamination(rows, task_profile,
  training_rows)` looks for every gold VALUE (span entity texts; Q&A /
  summarization answers; classification = labels, `applicable: False`) verbatim
  in the training rows (`project_training_rows`: prepared train + val, else
  the default sources; whitespace/case-insensitive, `MIN_VALUE_CHARS` 4) →
  manifest `contamination` {values_checked / found / found_share, occurrences,
  rows_hit, by_type, by_source (rows' `synth_source` / `source`), examples,
  severity clean / low (< 2 %) / notable / severe (≥ 10 %)}; `POST
  /{bid}/contamination` (`rescan_contamination`) re-runs it against the
  current rows and rewrites manifest.json + `Dataset.metadata_`. Surfaces:
  the panel's per-benchmark line (`benchmarkContaminationText`, "rescan",
  "which values"), the span model card (one line per affected benchmark via
  `contamination_line`). The row-level `leakage_filter` is unchanged. Why:
  the 2026-10-17 Gretel incident (32 values × 591 in training rows, nothing
  flagged it). A hit is evidence, not proof — common first names read `low`.
  Third span format `entity_strings` (offset-less `[{entity|value|text,
  types|type|label}]`, list or JSON/literal string — `_parse_entity_list`,
  `_entity_string_key` (≥ 80% of values found in the text), `locate_entities`:
  every non-overlapping occurrence, longest first) — Gretel PII.
  **Found by its classification proof (fixed)**: the HF source returned
  `ClassLabel` ids, not names (`sources/hf._class_label_names`); LoRA
  `"auto"` on encoders targeted no modules (`lora_target_policy.
  query_value_modules`, aliases DeBERTa/BERT/DistilBERT/T5, else all-linear);
  full fine-tunes and classification heads trained with bf16 weights and
  stayed at chance (`train.needs_fp32_master_weights` → fp32 masters,
  report `master_dtype`); full fine-tuned classifiers weren't recognised by
  eval (`_resolve_full_classifier_artifacts`); the lift check loaded encoders
  as causal LMs — `post_training_eval_service.is_encoder_classification` →
  `run_head_only_evaluation` (`kind="head_eval"`), summary
  `eval_summary_service.head_eval_summary` (card: "Accuracy x on the test
  examples", Re-evaluate, no seeds button).
  **G-B2 — generative shapes, groups, history**: `detect_generative_shape`
  (after the span checks, before classification) reads Q&A (`qa_pair` /
  `qa_context`: question + answer [+ context]; SQuAD `answers` dicts via
  `answer_text`) and summarization (`document_summary`) through
  `task_shape_service.detect_task_shape` (explicit column names as fallback;
  an "answer" column that looks like labels — few distinct one-word values —
  is not generative). Rows → `{"question", "answer"[, "context"]}` (a
  summary is the answer, the document the question); `fit_rows_to_project`
  folds the context into the question unless the project is `rag_qa`
  (manifest `context_folded`). No labels to map (preview `labels_to_map:
  false`, mapping `{}`); `shapes_compatible` lets a generative benchmark run
  on any generative project profile (`_generative_project_profile`: manifest
  task_profile, else `project_data_task_profile`). Scoring = the project's
  handler + token F1 / exact match where the handler has none + the answer
  judge (`judge_benchmark_answers`: `should_judge`, `resolve_answer_judge`;
  `ResolvedJudge.local` is False for cloud judges, and a `local_only`
  benchmark skips one → `judge.skipped="local_only_cloud_judge"` + message).
  Headline `judge_correct` when judged, else `f1`. **One-to-many mapping**: a
  mapping value may be a list of project labels → the group `"a|b"`
  (`mapping_target`, `GROUP_SEP`); `validate_mapping` refuses a project label
  in two groups; rows carry the group; `collapse_predictions` maps a member
  prediction to the group before the handler scores (classification: the
  parsed label + a collapsed candidate set; spans: entity types in the
  prediction JSON; original kept as `prediction_before_grouping`); manifest
  `groups`. **History**: `GET /{bid}/history` → `benchmark_history` (every
  `results.jsonl` row, oldest first: run id + name, date, headline, score).
  UI: multi-select mapping per row (`mappingMembers`, `mappingGroupConflict`
  disables create), "No labels to map" for Q&A / summarization,
  `BenchmarkTrend` inline SVG sparkline per benchmark (joins only points of
  the latest headline metric), `benchmarkHeadlineLabel` /
  `benchmarkJudgeNote` in `api/benchmarks.ts`. Tests:
  `tests/test_external_benchmarks.py`, `ExternalBenchmarksPanel.test.tsx`.
- **Playground "experiment" provider** — `provider="experiment"` +
  `experiment_id` chats with a completed run in-process
  (`local_chat_service`: one cached model, base+adapter or full model,
  tokenizer chat template over the whole conversation, classifier runs reply
  with the predicted label). Chat + stream routes share
  `_apply_playground_auto_rag` / `_resolve_playground_run` in `api/training.py`
  (the stream route previously skipped auto-RAG and rag_first). The UI opens
  on the latest run (or `?run=<id>`).
- **`field_match` scoring (typed JSON records)** —
  `StructuredExtractionHandler._score_field_match`, rules in
  `output_schema_service` (one place for scorer + validator). Per-field exact
  match is TYPED: NULL class = JSON null, missing key, `"null"`, `""`
  (`"none"` is NOT null: it is a real enum value); numbers compare as `Decimal`
  (`1250.0` == `"1,250"`) when the declared type is integer/number or both
  sides parse; booleans `true` == `"true"`; strings casefold + collapse
  whitespace + drop one trailing period (articles / internal punctuation kept:
  `"the destroyed"` != `"destroyed"`). Token-F1 stays for string fields (so F1
  can credit a near miss that EM refuses). dict/list values compare as
  `json.dumps(sort_keys=True)` through the old SQuAD EM/F1; **nested / list
  per-key scoring is a later phase**. `overall_em` = every schema field matches
  typed (extra predicted keys ignored, still visible in
  `schema_compliance_rate`). **Null metrics** (positive class = "predicted
  NULL"; valid-JSON rows only, invalid rows are in `json_validity_rate`):
  per field and micro-averaged `null_precision` = tp/pred_null,
  `null_recall` = tp/gold_null, plus `hallucinated` (pred non-null, gold null)
  and `abstained_wrongly`; zero denominator -> `None`, never 0 (the UI shows a
  dash; per-field `em`/`f1` are `None` at `support == 0`). Containers (object/array) compare
  literally: both sides containers and identical JSON (key order free). A gold row missing a key counts as NULL. **Enum + type validation**
  (only for fields with a declared `type` / `enum`): `out_of_vocab_rate` = share
  of non-NULL predictions outside the enum; `type_violation_rate` = share of
  non-NULL predictions of the wrong type (strict JSON typing, no coercion: `"3"`
  is not an integer, `true` is not an integer, `3` is not a string, `3.0` is an
  integer; coercion applies only to the exact-match compare, and a field
  declared string / string+null never compares numerically) plus NULLs where the type
  excludes null; `schema_valid_rate` = valid-JSON rows with no violation in any
  field. OOV values are still scored normally. Rows carry `schema_violations`.
  UI: `FieldMatchMetricsPanel` in `EvalPanel` (per-field table + aggregates,
  header names run / dataset / result id; shown when the result has null
  metrics). Declaring the schema: see `agent-notes/data.md` (json-extraction).
  Tests: `tests/test_json_extraction_scoring.py`, `FieldMatchMetricsPanel.test.tsx`.

## `finding_set` scoring + code-findings probes (G4)

- Mode inside `StructuredExtractionHandler` (`output_schema.scoring_mode == "finding_set"`); the scorer is
  the pure `finding_set_scoring.score_finding_set` (manifest-only inputs: `finding_match`, `label_parents`).
  A finding `{label, start_line, end_line, evidence, attrs}` is correct when the label is equal and the
  inclusive line-range IoU is >= `finding_match.iou_threshold` (0.5; exactly 0.5 matches). Greedy one-to-one
  per row, best IoU first; duplicates beyond one match are FP. Metrics: micro `precision/recall/f1`
  (strict), `*_macro`, `per_class` (= per label; `evaluation_pack_service._flatten_per_class_metrics` turns
  it into `f1_<label>` / `per_class.<label>.f1` gates), `clean_chunk_fp_rate`, `malformed_rate`,
  `invalid_finding_count`, `attr_accuracy(_mean)` (strict-matched pairs, configured `finding_match.attrs`
  only), `exact_match`, `json_validity_rate`. `hierarchical{}` (ancestor label + IoU gives
  `parent_credit`) exists only when `finding_match.hierarchy` is on and never changes the strict numbers.
  Override `finding_match` via `Project.runtime_config["finding_match"]` (no API sets it yet).
- Per-row keys (`row_matched/missed/spurious_findings`, `row_malformed`, `row_is_clean_reference`) flow into
  `predictions_preview`; per-row tp/fp/fn are exported to `row_scores.span_*` so paired comparison works.
  Headline metric is `f1` (`_PREFERRED_HEADLINE_METRICS`); `eval_type_for_project` -> `f1`.
- Probes: recipe-keyed pack (`get_probe_pack(task_profile, recipe_id)`; code-findings has its own, other
  recipes keep the profile pack). Inputs are dicts `{code, language, path, start_line}`; `run_probe_pack`
  renders them with `code_findings_prompt` (the eval path passes the manifest's `prompt_labels`). Properties
  `returns_empty_findings` and `findings_stable_vs_base` (predicted label SETS equal; a base with no findings
  still scores and records `note:"base_had_no_findings"`).
- Macro scores fall back to the micro score when `per_class` is empty (empty-on-empty is 1.0). The scaffolded
  pack gates `max_clean_chunk_fp_rate` / `max_malformed_rate` with `lte` (scaffold tuples may carry an
  operator as a fifth item). Code-findings probe generation uses the structured handler's
  `max_new_tokens_override` budget (512), not the 128 probe default.
- Tests: `test_finding_set_scoring.py`, `test_code_findings_probes.py`, `test_code_findings_e2e.py`, `test_code_findings_g4_fixes.py`.

## Code checks, sandbox, unified diffs (G5, lane B1)

- `services/code_check_service.py` is the registry (`register_code_check`, `list_code_checks`, `checks_for_language`, `run_code_checks`); built-ins in `services/code_checks/builtin.py`, sandboxed externals in `code_checks/external.py`. Externals run only when listed in `CODE_CHECK_ALLOWED` and the binary is on PATH; plugins via `CODE_CHECK_PLUGIN_MODULES`. API: `api/code_checks.py`. Docs: `slm-docs/docs/extensions/code-checks.md`. The checks answer "does it parse": `shellcheck` runs with `--severity=error` and `--shell=bash` when the snippet has no shebang / shell directive (any finding used to fail it — SC2148 rejected 85% of generated shell snippets, 2026-10-28); `rust_syntax` (`rustfmt --edition 2021 --emit stdout`, parse errors only — it never compiles or runs; a non-item fragment is retried wrapped in `fn __snippet() { … }` unless braces are unbalanced per `_mask_rust`; `error[Exxxx]:` diagnostics count, a non-zero exit with no diagnostic fails with the stderr tail, a rustup/toolchain error with no `-->` location is `skipped=unavailable`, `available()` probes `rustfmt --version` with `_rustup_env()` (cached) — the sandbox HOME hides rustup settings); `yaml_parse` refuses duplicate mapping keys at any depth (`_strict_yaml_loader`: a `_NoAliasLoader` subclass checking scalar keys by (resolved tag, SOURCE text) before merging -- `1.0`/`1` and `on`/`true` differ; skipped entirely when `_had_control_lines` -- templated YAML whose if/else/range lines were blanked, since branches flatten into sibling keys; comment-only lines (`{{/* */}}`, `{{- /* */ -}}`, `{# #}`, `_TEMPLATE_COMMENT_LINE`) don't count; so `<<` merges and overrides of merged keys pass; keys built from template placeholders are not compared; aliases stay refused, so only a literal `<<: {…}` merges; `<<: *x` still fails). `YamlParse.revision = 2`
  (`BaseCheck.revision`, default 1): `code_check_service.check_revisions(ids)` lists revised checks and
  `remediation_scoring_fingerprint` adds `check_revisions` only when non-empty -- every fingerprint whose check selection
  includes yaml_parse (the default "all runnable checks" one does) changed once, others did not. `evaluation_service` stamps `metrics.remediation_check_revisions` beside the fingerprint; Compare
  (`_check_revision_note`, `revision_note` on the check) says "built-in check revised (yaml_parse v2: duplicate keys
  now fail) — re-score the older run" and runs `strip_yaml_templates` first (one linear scan, expressions may span lines; template-only lines blanked, inline expressions → `TEMPLATE_VALUE`; a quoted `}}` inside an expression isn't understood) so Helm / Jinja YAML is checked. Remediation scoring runs the same checks before and after a fix, so a fix that only clears a shellcheck WARNING no longer counts as fixing `shellcheck`.
- `services/sandbox_exec.py` (`run_sandboxed`) is the only place external code-check commands run; tests patch `sandbox_exec.run_sandboxed`. It also owns the minimal-env/kill helpers that `synth_backends/command.py` imports. Network isolation uses `unshare -rn` when it works (probe cached; `CODE_CHECK_NETWORK_ISOLATION` auto/off/require); it does NOT work on the dev host (user namespaces refused), so checks there report `network_isolated: false`. `auto` FALLS BACK to no isolation when `unshare -rn` fails; only `require` refuses to run (docs say so, and the Train > Remediation checks card shows "network not isolated" per ready external check). Real built-ins are trusted by identity (`code_check_service._CORE_BUILTINS`), never by a self-declared `kind == "builtin"`; plugins always need `CODE_CHECK_ALLOWED`. Java: a fragment is wrapped (member, then statement) with a sentinel member after the code and a brace-balance lexer; `java_is_toplevel` only accepts a column-0 declaration with no leading statements.
- Remediation scoring (fix lane): `code_checkable_count` / `code_parses_rate` depend on the ORIGINAL chunk only (malformed = checkable but not parsing), so base and fine-tuned share denominators. `issue_resolved_rate` / `new_findings_rate` are over rows whose patch applied (informational, hidden from the lift table); the lift metric is `issue_resolved_overall_rate` = resolved / (scored rows - detector-blind - detector-error rows). A finding the detector did not report on the ORIGINAL chunk is `detector_blind` (`row_detector_note`, `detector_blind_count`): `row_issue_resolved` None, excluded from both rates, ignored by the pass rule. Metrics carry `remediation_fingerprint` (detector identity incl. run id, effective check ids, `CODE_CHECK_ALLOWED`, retrieval settings, extra-field schema hash); a cached base result is reused only when it matches (`post_training_eval_service._baseline_scoring_matches`). Detector cache keys include the findings-prompt builder + decoding fingerprint and, for command detectors, the binary realpath/mtime/size. Retrieval for rows without passages runs in worker threads (`combine_datasets`, `_load_heldout_pairs`); own-chunk exclusion matches by normalised path suffix, source document id, or text containment (>= 80% of the row's lines).
- `services/unified_diff.py`: `apply_patch_to_chunk` (absolute line numbers, exact match ± `max_offset`, reasons `empty_patch/not_a_diff/other_file/multiple_files/context_mismatch/overlapping_hunks/out_of_range/no_op`), `make_unified_diff` round-trips with it. `offsets[i]` = applied position minus stated position.

- **Opt-in checks / Epic Y2-Y3.** `BaseCheck.default_enabled` (default True): False = left out of the default
  selection (`check_ids=None`) unless the id is in `CODE_CHECK_ALLOWED`; naming it in `check_ids` always runs it.
  `code_check_service.in_default_selection(c)` / `default_check_ids()` are the one place that rule lives (the
  remediation fingerprint and `remediation_check_revisions` use `default_check_ids()`, so registering an opt-in
  check changed no fingerprint). `list_code_checks` rows carry `opt_in` + `default`; the remediation settings card
  shows "only when picked". `context_grounding` (`code_checks/grounding.py`, allowlists in
  `code_checks/grounding_allowlists.py`) is a core built-in with `default_enabled=False`, `requires_context=
  "grounding_text"` and `checkable_per_check=True`: term-extractor registry (`TERM_EXTRACTORS`,
  `register_term_extractor`; `code_symbols` first, picked via `context["grounding_extractor"]`). Terms have kinds:
  `import` terms ignore the code's own bindings, `member` / `attribute` terms (dotted tails, HCL attributes) accept
  only declarations (functions, classes, fields, `self.x =`), not locals; everything else accepts locals too. Std
  roots + names bound from std imports or std values (`_bind_std_locals`) are never checked past the root (Java's
  standard types are roots too: `Map.Entry`; a Java wildcard import of a standard package allows its listed types,
  `JAVA_STDLIB_PACKAGE_TYPES`, and of an unlisted one any bare PascalCase type). C++ direct initialisation
  (`std::ifstream file(p);`) and Go one-line structs (`struct{ total float64 }`) declare their names. Terms are
  deduplicated, dotted paths are checked up to `MAX_PATH_SEGMENTS` (8) segments and at most `MAX_TERMS` (2,000) terms per
  check (long chains stay linear). No `grounding_text` -> `skipped="no_context"`. `python_undefined_names` (pyflakes, in-process; rev 2: identifiers in `grounding_text` count as defined) sits in
  `builtin.GATED_IN_PROCESS`: registered but not a core built-in, so it needs `CODE_CHECK_ALLOWED`; available iff
  `pyflakes` imports. `properties_parse` (`builtin.PropertiesParse`, also in `GATED_IN_PROCESS`; language `properties`,
  aliases `java-properties` / `java_properties` / `javaproperties` in `base._ALIASES`): fails when the whole text is a JSON
  object / array, or a logical line (`properties_logical_lines`: Java `\` continuations, `#` / `!` comments and blanks
  skipped) starts with `{` `}` `[` `]` or `- `, naming the first physical line (max 5 lines + "and N more"); `key=value`,
  `key: value`, `key value`, bare `key` pass. `c_syntax` (`external.CSyntax`): clang/clang++ preferred, gcc/g++ fallback; syntax-class
  regex + same-line suppression (a syntax error on a line with a resolution error is ignored); missing headers are
  blanked and re-checked (max 12, else `skipped="unresolvable_includes"`); a statement fragment is retried inside
  `void __brewslm_snippet(void) {` with includes hoisted and `#line 1`. Code-example scoring passes
  `grounding_text` (the prediction's `prompt`, else `judge_question(row)`) to every check; checks with
  `checkable_per_check` drop only themselves when the reference fails them (gate checks keep the old whole-row rule,
  so existing scores are byte-identical: `OffByDefaultTests` golden sha). Calibration metrics
  `code_example_grounding_reference_{checked_count,pass_count,pass_rate}` + row
  `row_code_example_grounding_reference_pass`, shown in `CodeExampleMetricsSection` (`ce-grounding-reference`).
  Tests: `test_context_grounding.py`, `test_c_syntax_and_undefined_names.py` (real-binary / pyflakes cases skip when absent).

## `remediation` scoring + the code-remediation recipe (G5, lane B2)

- Recipe `code-remediation` (`structured_extraction`, scoring mode `remediation`). Prompt + payload helpers live in
  `services/code_remediation_prompt.py` (`build_code_remediation_prompt(_from_row)`, `build_code_example_prompt(_from_row)`,
  `canonical_remediation_json`, `retrieve_passages_for_row`, `effective_remediation_config`); code is rendered by
  `code_findings_prompt.render_numbered_code`. One builder serves the adapter (`data_adapter_service._map_remediation`),
  `StructuredExtractionHandler._build_remediation_prompt`, the playbooks and the judge; chat template for train AND eval
  (`wraps_own_prompt` false for `finding_set` and `remediation`).
- Manifest stamp (`dataset_service._remediation_schema_stamp`): `required` = the three base fields + declared extras,
  `properties` = declared EXTRAS only (base names stripped; `description` is kept — `OutputProperty.description`),
  `remediation {checks, retrieve_passages, top_k, retrieval}`, `label_parents`, `label_titles`. `combine_datasets` and the
  predefined-split import (`dataset_service._apply_recipe_adapter_config`) put `adapter_config["remediation"]` (extra
  fields, project id, retrieval settings) on the structured-extraction adapter, so imported remediation / code_example
  rows get the training prompt held-out eval renders.
  Retrieved passages are stored on the prepared row (`passages_retrieved`), so eval re-renders the identical prompt and only
  retrieves for a row prepared another way (the handler's `schema["project_id"]` comes from `ctx.project_id`).
- Scoring: `remediation_scoring.py`. `analyze_row` (apply patch via `unified_diff.apply_patch_to_chunk`, original/patched
  `run_code_checks`) is cached on the prediction (`_remediation_analysis`); `run_heldout_evaluation` precomputes it in a worker
  thread (`_prepare_remediation_predictions`), then runs the detector, then scores. A row whose context
  (`prediction["remediation_row"]`, copied across in `_generate_heldout_predictions` with path/language/start_line) is missing
  or has no `finding` is skipped (`skipped_non_remediation_rows`) — so predictions injected through
  `POST /evaluation/run` (no extras) score as 0 rows; only the held-out path scores remediation. A malformed row has all
  checks false and is not "checkable". `exact_match` = `remediation_pass_rate` (the `pass_rate` fallback).
  Lift: `remediation_pass_rate` heads `_PREFERRED_HEADLINE_METRICS`, pairs on `row_scores.remediation_pass`;
  `_prediction_failed` for a remediation row = not `row_remediation_pass` (the judge score does not decide).
- Detector (`remediation_detector.py`): `{"kind":"project",...}` runs the code-findings run through
  `_run_local_inference` (patched as `remediation_detector._infer_findings_text` in tests); `{"kind":"command",check_id}`
  needs a registered `kind="detector"` check, allow-listed + available, and calls its `run()` directly (so it may return a
  `findings` list; `make_result` would drop it). Cache: `data/projects/<id>/remediation_detector_cache.json`.
- Judge: `answer_judge_service.build_judge_prompt/build_retrieval_judge_prompt/cache_key` take `instruction` (None = the old
  prompt byte for byte); the 4th judge argument is only passed when an instruction is set, so 3-arg fakes still work.
  `_safe_judge_answers(scoring_mode="remediation")` uses `remediation_scoring.JUDGE_INSTRUCTION`/`judge_view`.
- **Opt-in code-example scoring** (`runtime_config["remediation"]["score_code_examples"]`, default off; `effective_remediation_config`
  key; stamped in the manifest `remediation` block ONLY when true; `evaluation_service._apply_remediation_live_config`
  overlays the live value at every place `_apply_structured_plan_live_config` runs). When on, the handler's code-example
  branch adds `code_example_row` {task, language, passages} (in `_ROW_EXTRA_KEYS`); `remediation_scoring` hands rows
  carrying it to `code_example_scoring.score_code_examples` instead of skipping them. Off = no `code_example_row`, so every
  metric / row / preview / row_scores is byte-identical (golden hash in `tests/test_code_example_scoring.py`). Checks: the
  REFERENCE code decides checkability (like the original chunk). The judge runs in `_prepare_remediation_predictions`
  (`_judge_code_example_predictions`, proxies so no `row_judge_*` keys) and annotates `_code_example_judge`; no verdict =
  `row_code_example_pass` None (out of the rate/pairing; `_prediction_failed` False). Only-code-example split: the
  remediation zero-metrics are omitted, `exact_match`/`total`/`correct` come from code examples, so
  `code_example_pass_rate` (right after `remediation_pass_rate` in `_PREFERRED_HEADLINE_METRICS`, pairs on
  `row_scores.code_example_pass`) heads the lift. The fingerprint gains `score_code_examples` only when on; a cached base result
  whose `metrics.code_example_judge.judge` differs from the current judge is not reused (`_baseline_judge_matches`).
- **Opt-in format before check** (`runtime_config["remediation"]["format_before_check"]`, default off; overlaid live by
  `_apply_remediation_live_config` with `score_code_examples`, `_REMEDIATION_LIVE_FLAGS`). `code_example_scoring.analyze_row`
  runs `code_check_service.format_code` on the PREDICTED code (not the reference) before the `after` checks. Formatters are
  checks with a `format()` method (`formatter_for`: `go_syntax` gofmt, `terraform_fmt` `terraform fmt -`, `rust_syntax`
  rustfmt; `ExternalCheck._format_with`, same `_sandbox` + timeouts) gated like the check (allowed + available). Status
  `formatted`/`unchanged` → check the formatted text; `failed`/`not_formattable`/`truncated`/`unavailable`/`not_allowed`/`timeout`/
  `no_formatter` → the original code (`not_formattable`: rustfmt rejects a statement fragment that formats inside
  `fn __snippet() {…}`). When on only: row `row_code_example_format` {status, formatter, messages} (also copied to previews),
  metric `code_example_format_before_check` {enabled, by_status}; fingerprint part `format_before_check: true` (only with `score_code_examples` also on), and
  `metrics.remediation_format_before_check: true` stamped next to the fingerprint when on (and scoring code examples);
  `compare_row_evidence_service._scoring` adds `format_before_check` (from that, else the code-example block) so the remediation-fingerprint difference reads
  "format before check off vs on". No serving hook exists (the gateway only validates replies), so served outputs are
  not formatted. UI: toggle in `RemediationConfigCard` (Held-out eval). Tests: `tests/test_code_example_format_before_check.py`.
- API: `api/remediation_config.py` (`/projects/{id}/remediation-config`; `api/remediation.py` is the older failure-plan API).
  Probe packs: `_RECIPES_WITHOUT_PROBE_PACK`. Not wired: `coach_service._PER_CLASS_RECIPE_IDS` (no `per_class` is emitted).
- **Opt-in code-example instruction** (`runtime_config["remediation"]["code_example_instruction"]`, default None; string,
  `normalize_code_example_instruction` collapses whitespace, cap `MAX_CODE_EXAMPLE_INSTRUCTION_CHARS` 500; the API 422s
  longer, blank clears). `build_code_example_prompt(..., extra_instruction)` puts it as its own line after the first
  instruction line; None/blank = byte-identical (golden sha in `tests/test_code_example_checks_and_repeats.py`). It is
  stamped into the manifest `remediation` block ONLY when set (`_remediation_schema_stamp`), copied into
  `adapter_config["remediation"]` (training rows, `_map_remediation`), and the eval handler reads the stamp -- NOT live
  (not in `_REMEDIATION_LIVE_FLAGS`), so train/eval prompts match; changing it needs a re-split. The remediation
  fingerprint gains `code_example_instruction` (sha of the manifest stamp) only when set. UI: "Task→code prompt" textarea
  in `RemediationConfigCard` (sent only when non-blank; PUT replaces).
- Tests: `tests/test_code_remediation_recipe.py`, `tests/test_code_example_scoring.py`,
  `tests/test_code_example_checks_and_repeats.py`. Docs: `slm-docs/docs/workflows/code-remediation.md`.

## `structured_plan` scoring + the structured-plan recipe (G6, lane B2)

- Recipe `structured-plan` (`structured_extraction`, scoring mode `structured_plan`, columns `scenario` / `plan_json`). Prompt +
  payload + settings helpers: `services/structured_plan_prompt.py` (`build_structured_plan_prompt(_from_row)`,
  `canonical_plan_json`, `retrieve_passages_for_scenario`, `effective_structured_plan_config`, `bundle_for_config`,
  `rubric_for_config`, `project_is_structured_plan`). Passages render through `code_remediation_prompt.render_passage`. One
  builder serves the adapter (`data_adapter_service._map_structured_plan`), `StructuredExtractionHandler._build_structured_plan_prompt`
  and the playbook; chat template for train AND eval (`wraps_own_prompt` false), `max_new_tokens_override` 1024.
- Settings live in `runtime_config["structured_plan"]` (`api/structured_plan_config.py`: GET/PUT `structured-plan-config`,
  POST `structured-plan/validate`; 422 `INVALID_STRUCTURED_PLAN_CONFIG` with `metadata.errors`, 409 `NOT_STRUCTURED_PLAN`).
  A valid schema writes `runtime_config["guided_json"]` (`source: "structured_plan"`, the dereferenced schema); clearing it or
  switching recipe drops it (`code_findings_prompt.sync_guided_json`); a code-findings hint is never overwritten.
- The manifest stamp (`dataset_service._structured_plan_schema_stamp`) only records root / sha / pointers / retrieval. At
  eval time `evaluation_service._apply_structured_plan_live_config` overlays the project's CURRENT schema files, pointers
  and retrieval onto the manifest (in `run_evaluation` and `run_heldout_evaluation`), so prompt and scoring use live
  settings; `combine_datasets` puts `adapter_config["structured_plan"]` (schema files, retrieval) on the adapter for prep.
- Scoring: `structured_plan_scoring.score_structured_plan` (per-row `row_schema_valid`, `schema_errors`, `row_pointer_results`,
  `row_structured_plan_pass`, `row_exact_match`; the row context rides on `prediction["plan_row"]`, copied across in
  `_generate_heldout_predictions`). Pointer rule: `acceptable[ptr]` present -> at least one pred value and every pred value in
  the list; else multiset equality under `value_match` with types from the gold value. Pass = valid JSON + schema valid + all
  pointers (+ rubric not failed). `finalize_with_rubric` merges the rubric block and recomputes the pass metrics.
- Rubric pass (`evaluation_service._judge_structured_plan_rubric`): after scoring, `rubric_service.judge_rows_with_rubric` runs in
  `asyncio.to_thread` with `answer_judge` (lift evals pass one) or `resolve_answer_judge`; the regular single-verdict judge is
  skipped for this mode. No judge -> `metrics["rubric"] = {"status": "no_judge", id, title, sha256}`; rubric but judge off ->
  `status: "disabled"`. `structured_plan_scoring_fingerprint(runtime_config, judge_label, project_id)` is stamped as
  `metrics["structured_plan_fingerprint"]`; `post_training_eval_service` reuses a cached base result only when it matches
  (`_structured_plan_fingerprint_for`, and `_baseline_judge_matches` reads `metrics.rubric.judge` for this mode).
- Lift: `structured_plan_pass_rate` is a headline metric (right after `remediation_pass_rate`; the modes are exclusive),
  pairs on `row_scores.structured_plan_pass`. Recipe-keyed literals: eval pack scaffold + gate catalog entries,
  `_RECIPES_WITHOUT_PROBE_PACK`, smoke-test prefix skip, preflight `row_kind`, `task_shape_service` detection
  (`scenario` + `plan_json`). `predictions_preview` carries the new per-row keys and `grid_cell`.
- Hardening (G6 fix lane): dialects are 2020-12 / 2019-09 only, one per bundle (`validate_bundle` errors otherwise; the
  flattened document keeps the dialect). `_DATA_KEYS` (`default`/`enum`/`const`/`examples`) are data only in keyword
  position; children of `properties`/`patternProperties`/`$defs`/`definitions`/`dependentSchemas` are always schemas.
  `validate_bundle` builds the validator and the flattened schema before returning, so anchors across files (unsupported:
  use a JSON-pointer fragment), recursion depth etc. are 422s, never later 500s. `pattern`/`patternProperties`/
  `additionalProperties` use the `regex` module with a 0.5 s timeout (`PATTERN_TIMEOUT_S`) -> "pattern check timed out"
  (`unevaluatedProperties` matches `patternProperties` through `jsonschema._utils.re.search`, which no keyword override
  reaches; `_install_timed_re` swaps that module's `re` for a timed `regex` shim, and `validate` maps the
  `TimeoutError` to the same error). Schema checks run in `asyncio.to_thread` (`structured_plan_scoring.precompute`
  via `evaluation_service._prepare_structured_plan_predictions`; POST validate; the playbook answers validate); the scorer
  reuses the stashed verdicts. `structured_plan_prompt.bundle_for_config` / `schema_text` are LRU-cached by (root, sha).
  Rubric pass: `llm_budget_guard.refresh_if_stale()` on the main loop, then the worker thread runs under
  `cached_ledger_only()` (no DB session from the foreign loop; caps still enforced from the ledger). The rubric cache key
  includes a digest of the passages. `metrics.schema_changed_since_prep` + `prep_schema_sha256` when the live schema sha
  differs from the manifest stamp. Typed pointer compare is lenient (`"1"` matches 1, strings case/whitespace-insensitive);
  the rubric judge grades ALL rows, including ones that failed the checks. `per_pointer[ptr].both_missing` rows are excluded
  from `missing_rate`. Tests: `tests/test_structured_plan_hardening.py`, `tests/test_json_schema_bundle.py`.
- Tests: `tests/test_structured_plan_recipe.py`, `tests/test_structured_plan_acceptance.py`. Docs:
  `slm-docs/docs/workflows/structured-plan.md`.

## `multi_option` scoring + operator-registered verifiers (opt-in)

- **What it scores.** The model answers with a *menu*: `{"explanation", "options": [{strategy, title, rationale,
  tradeoffs, rollback, fixes: {<format>: text}}]}` (option order = ranking, option 0 = top). Each option is judged by an
  **option verifier**: an external command the OPERATOR registers; the domain logic lives there, BrewSLM stays generic.
  Scoring mode `multi_option` of `StructuredExtractionHandler` (`_multi_option_schema`, `_build_multi_option_prompt`,
  `wraps_own_prompt` false, `max_new_tokens_override` >= 2048; it honours a larger request — the run-heldout API allows up to 8192, and real
  menus run 3-6k tokens, so ask for ~6144). Code: `services/multi_option_scoring.py`.
- **Turning it on.** Manifest `output_schema.scoring_mode == "multi_option"` (config in `output_schema.multi_option`), or
  `runtime_config.multi_option.enabled = true` on a structured-extraction project. Settings in `runtime_config["multi_option"]`
  overlay the manifest stamp (`evaluation_service._apply_multi_option_live_config`, called by `run_heldout_evaluation`,
  `run_evaluation` and `heldout_scoring_context`): `formats`, `min_options` / `max_options`, `strategy_taxonomy`,
  `verifier: {check_id}` (an id ONLY -- `normalize_config` and the API drop anything else, so a project can never supply
  argv), `require_agreement`, `row_pass` (`top_option` default / `any_option` / `all_options`; EVERY rule also needs
  `n >= min_options`, so too few options fail the row), `instruction` (default "Propose alternative solutions for the problem
  below."; `effective_instruction`). API: `api/multi_option_config.py` GET/PUT `/projects/{id}/multi-option-config`
  (lists the registered verifiers read-only, never their argv; 422 `INVALID_MULTI_OPTION_CONFIG` with `metadata.errors`
  for bad values, an id that is not a registered option verifier, or `enabled: true` with no format in the effective config
  -- an option with nothing to check would otherwise pass vacuously; `normalize_config` reports the same for `enabled` raw
  configs). With no format at all (e.g. a manifest-only config), an option is decided ONLY by an explicit `option_passes`
  bool from the verifier; anything else fails it as `verifier_undecided`.
- **Rows.** `context` (rendered into the prompt), `verifier_payload` (opaque, handed to the verifier only -- may hold hidden
  assertions; never rendered into a prompt, never in `predictions_preview`, never read by checked decoding; both are read
  from the TOP LEVEL of the row, a nested copy is not seen),
  optional `options_json` (reference options, display only). A row without `context` keeps its own prompt text. The row rides
  as `prediction["multi_option_row"]` (`_ROW_EXTRA_KEYS`). Training-row mapping (data adapter / recipe / playbook) is not built:
  prepare rows with `build_multi_option_prompt` so train and eval prompts match.
- **Operator allow-list (`services/external_checks.py`).** `settings.EXTERNAL_CHECKS_FILE` (default unset = none) names a JSON
  list of `{id, kind: option_verifier|detector, protocol: 1, argv, formats|languages, timeout_s, cpu_s, mem_mb, revision}`.
  Each becomes an `ExternalCommandCheck` in the code-check registry (`default_enabled=False`; `option_verifier` and
  `detector` kinds are never in the default selection) and runs only when its id is ALSO in `CODE_CHECK_ALLOWED` (two keys).
  The file is re-read on path / mtime / size change (`ensure_loaded`, called from `get_code_check` / `list_code_checks`);
  removed entries unregister; a malformed entry or one shadowing a non-file check is skipped and reported. Execution:
  `sandbox_exec.run_sandboxed` (temp dir, scrubbed env, rlimits, `unshare -rn` per `CODE_CHECK_NETWORK_ISOLATION`), payload
  on stdin, no files shared. Detector entries use the unchanged `remediation_detector` contract (`kind: "command"`); their
  revision joins the detector identity.
- **Option-verifier protocol v1.** stdin `{"protocol": 1, "payload": <verifier_payload>, "option": <one option>, "formats": [...]}`;
  stdout `{"formats": {fmt: {"checks": {name: bool|null}, "passes": bool, "messages": []}}, "agreement": {"agree", "differences"},
  "option_passes": bool}` (`normalize_verdict`: non-bool check = n/a; missing `passes` = all decided checks passed). Exit 0 =
  a verdict. Errors carry `error_kind`: `infrastructure` (sandbox could not launch it) or `option` with `reason`
  (`verifier_timeout` / `verifier_crash` (non-zero exit) / `verifier_output_too_large` / `verifier_bad_output`). An option
  passes when every format passes, the formats agree (when `require_agreement` and >= 2 formats) and `option_passes` is not
  false. A format with no fix fails locally (`checks: {present: false}`) without asking the verifier.
- **Which verifier errors exclude a row (maintainer rule).** Only INFRASTRUCTURE errors: the sandbox could not launch the
  verifier, or it is not registered / not allow-listed / not installed (`metrics.verifier_unavailable`) / did not run. That
  option gets `passes: null` + `error`; the row is EXCLUDED (`row_verifier_error`, `row_multi_option_pass: null`,
  `row_exact_match: null`, `row_scores.correct` None so pairing skips it) only when its rule cannot be decided without it --
  `top_option` looks at option 0 alone, `any_option` is decided by any passing option, `all_options` by any failing one.
  `verifier_error_rows` counts excluded rows. A row with no `verifier_payload` (null or absent) is the same case: the
  verifier is NOT called, the option gets `multi_option_scoring.NO_PAYLOAD_ERROR`, and `rows_missing_verifier_payload` counts
  it (the card warns). An empty payload (`{}`, `[]`, `""`, `0`) is still a payload, so a verifier that needs nothing from the
  row uses `{}`. Malformed output and options with no fix need no verifier and stay decided. Everything the model's option CAUSED -- a per-option timeout, a non-zero exit,
  unparseable or oversized output, a verdict leaving a format / the agreement undecided (`verifier_undecided`; a missing
  agreement is undecided only when it would decide the option, not after a failed format or an `option_passes: false`) -- FAILS that
  option (`passes: false`, `failure_reason`, `note`); `option_verifier_failures` counts them (decided rows). Nothing raises.
  When every row is excluded (or there are none) `multi_option_pass_rate` / `exact_match` are None ("not scored"), never 0.0.
- **Verdict cache.** Verdicts run in a worker thread before scoring (`_prepare_multi_option_predictions` -> `precompute`,
  cached on the prediction as `_multi_option_analysis`) and are cached per project in
  `DATA_DIR/projects/<id>/multi_option_verifier_cache.json`, keyed by sha256 of verifier identity (`id:p<protocol>:r<revision>`)
  + payload + option + formats (errors never cached). Each entry is `{"v": verdict, "mac": HMAC-SHA256}` keyed by
  `SECRETS_ENCRYPTION_KEY` (else `JWT_SECRET`) mixed with a per-install random key `DATA_DIR/.multi_option_verdict_cache.key`
  (0600; a per-process key if unwritable); an entry whose MAC fails (edited file, older unsigned format) is dropped on load
  (`VerdictCache.rejected`). Operator verifiers run untrusted model output and must isolate it themselves (`sandbox_exec`
  limits time / memory / network, not the filesystem).
- **Metrics.** `multi_option_pass_rate` (row_pass rule over verified rows; also `exact_match`), `option_pass_rate`,
  `rows_with_passing_option_rate`, `top_option_pass_rate`, `all_options_pass_rate`, `distinct_strategies_mean` (distinct
  in-taxonomy strategies among PASSING options; `row_distinct_strategies_all` counts all options),
  `rows_with_two_distinct_passing_rate`, `out_of_taxonomy_rate` (only with a taxonomy), `cross_format_agreement_rate` (only
  when reported), `option_count_in_range_rate` (reported; fewer than `min_options` also fails the row), `format_pass_rate{}`,
  `check_pass_rate{}`, `malformed_rate`, `json_validity_rate`, `verifier_error_rows`, `option_verifier_failures`,
  `metrics.multi_option` (formats, range, rule, verifier id + revision). Per row: `row_options` (cards; `failure_reason` on
  verifier-failed options), `row_options_passing`, `row_top_option_pass`, `row_any_option_pass`, `row_distinct_strategies`,
  `row_option_verifier_failures`, `row_multi_option_pass`, `row_exact_match`. The stored preview
  (`evaluation_service._multi_option_preview` -> `multi_option_scoring.preview_fields`) caps each fix text at 1500 chars
  (`fixes_truncated` names the clipped formats), options at 5 (`row_options_truncated`) and formats at 6; scores always use
  the full analysis.
- **Lift / Compare.** `multi_option_fingerprint` (formats, range, taxonomy, row_pass, require_agreement, sha256 of the
  effective instruction, verifier identity) is
  stamped on the metrics; `post_training_eval_service._multi_option_fingerprint_for` + `_baseline_scoring_matches` reuse a
  cached base result only when it matches (bumping the verifier `revision` re-runs the base model); Compare refuses to pair
  across fingerprints (`_SCORING_FINGERPRINTS`). `multi_option_pass_rate` is a headline lift metric (after
  `tool_call_pass_rate`), pairing on `row_scores.multi_option_pass`; `out_of_taxonomy_rate` is lower-is-better; the counts are
  hidden. The answer judge never runs for this mode (deterministic, like `structured_plan`).
- **Checked decoding.** `multi_option_shape` verifier (auto when the eval is in this mode): option count in range, strategies in
  the taxonomy and distinct, a non-empty fix in every format. It reads only `VerifierInput.option_shape` (project config the
  prompt renders); it NEVER runs the option verifier, whose payload may hold hidden assertions.
- **UI.** `MultiOptionMetricsPanel` (tiles, verifier id + revision + fingerprint provenance, per-format / per-check bars with
  10% width steps) and `MultiOptionRowView` / `OptionCard` (rank, strategy chip muted when out of taxonomy, rationale /
  tradeoffs / rollback disclosure, one tab per format with the fix, format x check grid, agreement badge with differences,
  failed-check messages, "not verified" for infrastructure errors, "Fails: the verifier crashed / timed out ..." for
  option-level failures, "(truncated)" on capped fixes / option lists, "not scored" for a null headline) in `EvalPanel`.
  Fixture `multiOptionBackendShape.fixture.json` is generated from the real scorer + `_multi_option_preview`. Pipeline eval
  marks an excluded row unscored (`row_unscored`).
- **Not built yet:** Compare per-row option counts side by side, a Settings card (the API exists), a synth / review-queue gate
  ("keep rows whose options all pass"), alignment pairs from passing vs failing options, the optional detector inside
  multi-option scoring, a recipe / data-adapter mapping.
- Tests: `tests/test_multi_option_scoring.py` (pure scoring, prompt, shape verifier, defaults unchanged),
  `tests/test_external_checks.py` (operator file, two-key gate, sandbox call, protocol errors + error kinds, cache + revision,
  tampered cache entries, detector),
  `tests/test_multi_option_eval_flow.py` (TestClient end to end with a real toy verifier script from `tests/_multi_option_toy.py`,
  summary headline, settings API, fingerprint on revision bump); frontend `MultiOptionMetricsPanel.test.tsx`,
  `MultiOptionBackendShape.test.tsx`.

## Tool-call scoring (G7)

- `services/tool_call_scoring.py` is the scorer; the catalog it scores against is `tool_catalog_service` (see
  `agent-notes/data.md` "Tool catalog"). Settings `runtime_config["tool_scoring"]` (`enabled`, `modes`, `steps_pointer`,
  `tool_field` / `args_field` / `id_field` / `depends_field`, `arg_keys`, `dry_run`) via `api/tool_scoring_config.py`;
  `effective_tool_scoring_config(rc)` is `None` without a catalog or when disabled, and then EVERY caller behaves exactly as
  before (prompts, scores, fingerprint). `GET` returns `effective` always (defaults + `active` + `dry_run_available`);
  `dry_run: true` is a 422 unless `MCP_DRY_RUN_ENABLED`.
- `extract_calls(parsed, cfg, mode)`: `plan` resolves `steps_pointer`; `call` takes `{tool,args}`, `{tool_calls}`, a list,
  `{name,arguments}` (string arguments are parsed) or `{"function": {...}}`; `[]` = explicit no-call / empty plan, `None` =
  nothing extractable. `parse_call_output` also reads the adapter's `Tool Call: name({...})` rendering (it wins unless the
  output starts with `{` / `[`).
- `score_calls` -> `{calls[{tool, allowed, known, args_valid, args_errors<=3, args_match, dry_run}], tool_call_valid,
  tool_args_match, plan_order_valid, pass, errors}`. Arguments validate with `tool_catalog_service.tool_bundle` (a tool with
  `schema_error` is always invalid). Args match is positional, typed `value_match`, keys = `arg_keys` or the gold's keys; a
  key absent from the gold must be absent (or null) in the prediction. Order checks: duplicate ids, unknown dependency,
  cycle (iterative DFS, self-dependency included); `None` in call mode. `pass` = every ENABLED mode that is not `None`.
- `score_tool_block` (blocking; run in `asyncio.to_thread`) sets `row_tool_results` / `row_tools_pass` / `row_tool_calls`
  and returns the `metrics.tools` block (`catalog_version`, `flags_version`, `source_kind`, `modes`, rates, `pass_rate`,
  `per_tool`, `unknown_tools`, `errors_top`, `dry_run`) plus top-level rates. `evaluation_service._apply_tool_scoring` runs it
  for `structured_plan` and `tool_calling` evals BEFORE the rubric pass: `finalize_with_tools` folds the verdict into
  `row_structured_plan_checks_pass` and recomputes the pass metrics; `finalize_tool_calls` does the same for `tool_calling`.
  Failures are logged, never fail the eval.
- Prompt: `structured_plan_prompt.build_structured_plan_prompt(..., tool_block=)` puts `render_tool_block` after the passages
  and before the schema. Training gets it through `adapter_config["structured_plan"]["tool_block"]` (`dataset_service`),
  eval through the live-config stamp (`_apply_structured_plan_live_config`). `tool_block_for(rc)` is `""` without an active
  catalog. `structured_plan_scoring_fingerprint` adds `parts["tools"] = tools_fingerprint(rc)` ONLY with an active catalog
  (so old fingerprints stay valid).
- Dry run (`run_dry_run`, informational): for rows whose calls are all valid, read-only calls go through
  `tool_catalog_service.dry_run_call` (the only path to `tools/call`), at most 50 per eval; mutating ones record
  `{"skipped": "mutating: never executed"}`. The summary is `tools.dry_run = {enabled, calls, ok, errors, skipped}`; with the
  admin switch off it is `{enabled: false, ...}` and nothing runs.
- `tool_calling` profile: `ToolCallHandler` (scoring mode `tool_call`). Prompt = `tool_prompt(block, row.user_prompt or
  question)` (the `tool-call-json` adapter writes `question` / `source_text` / `text` with the same block and keeps the bare
  prompt in `user_prompt` only when a catalog exists, so eval never doubles the block). Gold = the row's `tool_name` + `tool_args_json`
  when present, else the `reference` parsed as a call (`tool_gold_calls` on the prediction); the `tool-call-json` adapter's
  target (`answer`) is the canonical call JSON `{"tool","args"}`, so a model trained on it emits what the scorer reads
  (a JSON tool result with `name` keys can no longer become the gold). A `{"tool":"none"}` mixed with real calls is invalid
  ("no-call mixed with calls"). Metrics carry `tool_call_fingerprint` (`tool_call/v1+tools_fingerprint`, stamped even
  without a catalog); `post_training_eval_service._baseline_scoring_matches` reuses a cached `tool_call` base result only on equal fingerprints. Metrics: `tool_selection_accuracy`,
  `tool_args_match_rate`, headline `tool_call_pass_rate` (= `exact_match`); with a catalog the pass also needs
  `tool_call_valid` (modes respected) and `metrics.tools` is added. Lift: `tool_call_pass_rate` is a headline metric after
  `structured_plan_pass_rate`, pairs on `row_scores.tool_call_pass`; the eval-pack `tool_success_rate` alias list includes it.
  The single-verdict answer judge is skipped for this profile.
- Tests: `tests/test_tool_call_scoring.py`, `tests/test_tool_scoring_acceptance.py` (fake MCP server over stdio).
  Docs: `slm-docs/docs/workflows/structured-plan.md` (Tool calls), `slm-docs/docs/reference/api/tool-scoring.md`.

## Multi-turn chat eval + pairs from eval (G8)

- `dialogue_eval_service` (generic, any chat dataset): `_extract_prompt_and_reference` fills a chat row's prompt (last user
  turn) / reference (final assistant turn); `_load_heldout_pairs` attaches `pair["history"]` (messages before the final
  assistant turn, system included; the handler's rewritten prompt replaces the last user turn) only when it is more than a
  bare `[user]` (so single-turn rows are byte-identical) and `pair["dialogue_messages"]` (the full list).
  `_apply_chat_template_if_present(tokenizer, prompt, history)` renders it with the chat template (no template: `role: content\n` lines + `assistant: `, as the trainer renders); the llama.cpp path passes it as `messages`. Handlers that wrap their own prompt (`wraps_own_prompt`, `apply_chat_template=False`) ignore history on BOTH the transformers and the llama.cpp path. If `apply_chat_template` raises or returns nothing, a history falls back to the same no-template `role: content` render (`_history_fallback`), matching training. `history_for_pair` replaces the last USER turn only when the handler actually rewrote the prompt (compared stripped against that turn).
- Per-turn replay judge: `dialogue_judge: bool` on `HeldoutEvalRunRequest` -> `run_heldout_evaluation(dialogue_judge=)`.
  Replay pairs (one per NON-final assistant turn of rows with 2+, capped `MAX_REPLAY_GENERATIONS`=500) ride in the same
  inference call as the held-out rows and are split off before scoring; the judge is resolved BEFORE generating
  (`metrics.dialogue_judge = {"skipped": reason}` for `judge_disabled` / `handler_builds_own_prompt` / `no_multi_turn_rows` /
  `no_judge_available`). Summary `metrics.dialogue_judge` = `{judge, rows, mean, worst_turn_mean, by_turn_index[{turn, mean, n}],
  turns_judged, unjudged, judge_calls, judge_cached, judge_tokens, replay_generations, usage...}`; per row
  `details.row_predictions[i].dialogue_judge = {turns[{turn, score, verdict}], mean, worst_turn}`. Turns are 1-based assistant
  turn ordinals; cache keys use the flat history as the question.
- **Judged rows live in a side file, not in the result row** (`evaluation_service.write_row_predictions_file`):
  `<project>/experiments/<exp_id>/evals/result_<eval_result_id>_rows.jsonl`, written after the `EvalResult` row exists (flush gives
  the id); `details.row_predictions_file = {path (relative to the project dir), count, truncated}` is all the result keeps, and
  `GET /evaluation/results/{exp}` also strips any legacy inline `row_predictions` array. Only rows a judge scored are stored (the
  answer judge's `row_judge_verdict` and/or the dialogue judge): `row_index` (position in the held-out file), `reference`,
  `prediction`, `history` (last `ROW_HISTORY_MESSAGE_CAP`=40 messages, `history_truncated` when cut; chat rows carry NO duplicate
  `prompt`), `row_judge_verdict` / `row_judge_score`, `dialogue_judge`. At most `ROW_PREDICTIONS_LIMIT`=1000 records; `truncated` =
  judged rows past it. Evals without a pointer report `no_prediction_stored` for pairs.
- **Judge profiles**: `chat_sft` is in `answer_judge_service.JUDGE_PROFILES` (final-turn scoring, same median >= 8 reference words rule);
  a chat row's judge QUESTION is the whole dialogue (`history_text`: `role: content` lines, system included), the same text the
  dialogue judge sends, so cache keys are shared.
- **Replay cap**: `build_replay_pairs` replays whole rows up to `MAX_REPLAY_GENERATIONS`=500; rows past it are NOT judged on their final
  turn alone - `judge_dialogues` skips them and reports `rows_not_replayed` / `replay_cap_hit` in `metrics.dialogue_judge`
  (`DialogueJudgePanel` shows them). Replayed generations are excluded from `average_latency_ms`, `token_throughput_tps`,
  `total_generated_tokens` and `chat_template_applied_count`; `runtime` carries `replay_generations` / `replay_generation_seconds` /
  `replay_generated_tokens` instead.
- `preference_pairs_from_eval_service` + `POST /api/projects/{id}/alignment/preference-pairs/from-eval` (`api/alignment_pairs.py`;
  the older alignment routes live under `/training/alignment/...`): wrong / partial rows from the side file -> `{prompt, chosen: reference,
  rejected: prediction, source: "eval", source_split, eval_result_id, experiment_id, row_index, verdict}` (+ `messages` for chat rows)
  in `prepared/alignment/eval_<id>_pairs.jsonl` (no Dataset row: every existing `DatasetType` is looked up as a singleton, so the reply's
  `dataset_id` is null; the UI shows `dataset_path` and says to set it as the alignment dataset path in Train). A row's verdict is the
  answer judge's, else the dialogue judge's verdict on the row's FINAL turn (highest turn number). A chat row's `prompt` is the same
  no-template render eval uses (`role: content` lines + `assistant: `); `messages` stay on the pair. Skips: `not_judged`, `correct`,
  `not_included`, `prediction_equals_reference`, `empty_prediction`, `empty_reference`, `no_prediction_stored`, `not_stored` (judged rows past
  the side file's caps or whose history was cut). 0 pairs -> 400 `no_pairs_built`, `metadata.skipped`.
- **Test split is refused**: an eval whose split (`EvalResult.dataset_name`: `test` / `gold_test`; an empty/unknown name counts as test)
  was the held-out test set -> 400 `test_split_refused` (memorisation). `allow_test_split: true` builds them anyway, adds `warning`
  to the reply; every pair carries `source_split` either way. UI: the "Allow test split (measures memorisation)" checkbox shows only for
  test splits.
- Known gaps (follow-ups, not done): eval double-BOS on Mistral / Llama / Gemma (the chat template adds BOS and the tokenizer adds another
  - pre-existing, single-turn rows too); length estimates run on display text; replay prompts skip handler rewriting.
- Tests: `tests/test_dialogue_eval.py`, `tests/test_documents_multi_turn_and_pairs.py`. Docs: `slm-docs/docs/workflows/multi-turn-chat.md`.

- **Pipeline evaluation (G10)** — chains projects into 1..8 stages and scores the chain on one held-out set.
  ("Pipeline" = this multi-model chain; not the project lifecycle data → train → eval → ship.)
  Files, no table: `DATA_DIR/pipelines/<pl-xxxxxxxx>/{pipeline.json, heldout.jsonl (+ heldout.meta.json row count),
  runs/<run-xxxxxxxxxx>.json (+ .summary.json sidecar the list endpoints read)}`
  (atomic tmp + `os.replace`; `pipeline_service`). Validation returns every problem as `{field, message}`
  (`stages[i].inputs.<f>.from`…; 400 envelope like the gateway settings PUT). At most ONE stage fans out
  (`fan_out: {pointer, item_key}`); later stages run per item. Fan-out gold = list of objects carrying the
  item-key fields, reference = the object's `output` field when present, else the object. Runner
  (`pipeline_eval_service`, Job kind `pipeline_eval`, project-less, params `{pipeline_id, run_id, max_rows}`;
  result `{pipeline_id, run_id, rows, share_fully_correct, truncated, calls, attribution.first_wrong_stage}`):
  stage k≥2 runs twice, `end_to_end` (predicted k-1) and `isolated` (gold k-1, `skipped_rows` without it);
  item matching = all key pointers present and equal after str/strip/lower, greedy in order (nth↔nth); a None
  key part matches nothing (`missing_item_key`), repeated predicted keys keep the first (`duplicate_item_key`);
  unmatched outputs `spurious`, unmatched gold `missed_upstream` (or `lost_to_error` when the upstream unit
  errored). Isolated view runs only with gold for the stage AND k-1. Upload rejects fan-out gold items without
  every key value (`fan_out_gold_item_missing_key`); each run re-runs `parse_heldout` on the stored rows and
  reports misfits as `gold_mismatch`. Templates: unknown `{x}` placeholders are a validation error.
  **Scoring = held-out scoring (no drift).** Per unit a pseudo-row = stage input values + `source_text` (stage
  prompt) + gold under every `REFERENCE_FIELDS` key the project's first prepared row carries (`reference` when
  none; object-typed where the prepared row's value is an object) → the stage project's
  `task_handler.build_prompts` (ctx from `evaluation_service.heldout_scoring_context`) → prediction dict with
  `copy_row_extras` (THE helper `_generate_heldout_predictions` uses — add new per-row extras there, both
  paths get them) → `score_predictions_like_heldout(..., eval_ctx, task_handler)` (prep + handler + tool
  scoring + `_run_heldout_judges`: the structured-plan rubric or `_safe_judge_answers`, decided exactly as
  held-out with exact_match + judge enabled; spend metered under `project_context(stage project)`). The
  model input stays the stage's own prompt; the built row is for scoring only. UNSCORED with a reason, never
  right/wrong: `row_context_unavailable` (empty prompt/reference, remediation row without code/finding, plan
  without scenario, or the scorer's `row_skipped`), `needs_judge` (judge applies and gave no verdict for a
  row it decides: answer judge where `judge_decides_row`, rubric where checks passed), `scoring_error`
  (build_prompts / scorer raised; stage `scoring_error`). Stage block: `judge {kind,label,ran,units,reason}`,
  `judge_not_run`. The remediation judge never decides pass/fail (the patch checks do), so its absence only
  sets `judge_not_run`. An errored output with gold (missing_input, prompt_error — any builder exception —,
  model_error, invalid_json) is wrong. Attribution walks each final gold item's path: first stage scored wrong
  or errored; a stage with gold but unscored makes the item unscored (`unscored_items`, out of the share and
  attribution). The stage producing the fan-out items is judged PER ITEM: right iff it produced an item with
  the key that deep-equals (strings via `_normalize_answer`) the gold item at the fan pointer; no gold item →
  key match only (`item_compare` deep_equal/key_only/mixed + counts); a key it never produced counts against
  it (`missed_upstream_at`). Calls via
  `gateway_service.run_internal` (no rate limit, no telemetry row), cached per (stage, messages) within a run,
  capped at `MAX_MODEL_CALLS`=5000 (`truncated`, `truncated_at_row`, `truncated_message`; the cut row dropped
  whole). Stage models resolve by the project's real slug (`all_slugs`), never the `p<id>` alias. Cancellation
  is checked before every unit; "Scoring N stage outputs…" progress before scoring. Nothing produced / nothing
  scored → raises (run file `status: failed` + `error`). Sync run path only for `max_rows <= 5` (tests); else
  400 "use async_job=true". Upload cap `MAX_HELDOUT_BYTES` 50 MB → 413 (file and JSON). Run files carry
  `job_id`; `pipeline_service.mark_stale` shows a `running` run whose Job is terminal / missing (sync: not in
  `ACTIVE_SYNC_RUNS`) as `stale`. Auth (`api/pipelines.py`): with `AUTH_ENABLED`, VIEWER on every member
  project to see (else hidden / 403), EDITOR on every member to write/run; admins pass. Known gaps: deleting a
  project does not touch pipelines that reference it (the list flags `project_missing`; a run fails at model
  resolution); no `:run-N` re-check after save; one fan-out level only; `/api/jobs` (`/active`, `/{id}`) does no
  per-user / per-project filtering at all (pre-existing jobs-API gap), so a project-less `pipeline_eval` Job's
  title, params and result summary are visible to anyone who can reach the jobs API — not fixed in G10. Tests: `test_pipeline_eval.py`, held-out tool-call
  regression in `test_tool_scoring_acceptance.py`.
  Docs: `slm-docs/docs/workflows/pipeline-eval.md`.

## Eval-judge calibration (Judge calibration V4)

- **Full eval predictions** (`services/eval_predictions_service.py`): `run_heldout_evaluation` also writes EVERY scored row (judged
  or not) to `DATA_DIR/projects/<pid>/eval_predictions/<eval_result_id>.jsonl` after the result id exists (temp file + `os.replace`),
  and stores `details.eval_predictions_file = {path, count, truncated, judged}` beside the existing `row_predictions_file` (which
  is unchanged: judged rows only, for preference pairs). Any failure (build or write) is logged and the eval goes on without it
  (`_write_eval_predictions` / `safe_write`). Record: `row_key` (the row's key in `row_scores.keys`), `row_index`, `prompt` (or
  `history` for chat rows, last 40 messages), `reference`, `prediction`, `correct` (the `row_scores.correct` value),
  `row_exact_match` / `row_f1` when set, and per judge that graded it: `judge {scale: verdict, task: answer|remediation|code_example,
  label, verdict, score, reason, content_key}` (answer judge; code examples read `_code_example_judge`) and/or `rubric {scale:
  rubric, label, passed, weighted, critical_failed, rubric_sha, content_key, passages?}` (structured-plan rubric judge). What the
  judge READ is rebuilt with the same views the judging used (`history_text`, `remediation_scoring.judge_view`,
  `code_example_scoring.judge_view`, `structured_plan_scoring.rubric_view`); `question` / `reference` / `prediction` are stored on
  the judge block only where they differ from the row's own texts (no duplication). Caps: 10 000 rows (`truncated` counts the
  rest), 8 000 chars per text, 20 passages of 1 600 chars. `GET /evaluation/{eval_result_id}/predictions?offset&limit (<=500)&judged_only`
  → `{available, total, offset, limit, rows}`; `available: false` for an older result without a file; 404 for another
  project's result. `content_key` = `answer_judge_service.content_key(question, reference, prediction)` (sha1 of the texts
  normalised exactly like `cache_key`, 20 hex; judge-independent) plus the task when it is not the plain prompt
  (`remediation` / `code_example` / `rubric`): identical texts graded under another instruction are a separate item; keys stay
  stable across results for the same task. The write runs off the event loop (`asyncio.to_thread`) and replaces any file for
  the id atomically (an empty write removes it), so a reused id never inherits stale rows; `delete_experiment` removes its
  results' files after the cascade.
- **Calibration** (`services/eval_judge_calibration_service.py`; reuses V2's tiers, label_source setting, `weighted_rate`, Wilson,
  `effective_n`, `cohen_kappa`, `stratified_sample(strata=, key=)`, sidecar writer, skip semantics). Scales: `verdict` (answer judge,
  labels correct / partial / wrong) and `rubric` (pass / fail). Population per scale = every stored judged row, keyed by
  `content_key` (the same answer in a later eval, or judged by another judge, shares ONE label; a judge's most recent verdict on a
  key wins). Only files whose EvalResult still exists count (`live_result_ids`: one query per request, passed as `live_ids`;
  also part of the report cache key). Rows whose stored question / reference / prediction reached `TEXT_CAP` (or whose chat
  history was trimmed) are left out — never sampled, never sent to the reference Job — and counted as `over_cap` in progress
  and in each tier report. Labels: `DATA_DIR/projects/<pid>/eval_judge_calibration.json` = `{scale: {content_key: {tier: {label, at, labeller,
  reason?, stratum?, judge?}}}}` or skip markers; tiers `human` / `reference_model`, never pooled. Sampler: answers the chosen
  judge (default: the one with most rows) graded, minus those labelled or skipped in the tier, round-robin across that judge's
  verdicts, seeded, shuffled; rendered as `{content_key, scale, question, reference, prediction}` plus, for rubric rows,
  `passages` / `passages_truncated` — the SAME list the reference model's rubric prompt gets (`budgeted_passages`: the review-assist
  `_budget_context` over `ROW_BUDGET["structured_plan"]` minus the texts). Report per judge (positive =
  correct / pass), each labelled row weighted by (judge's stratum population / labelled in stratum): `exact_agreement`,
  `strict_agreement` (correct vs not), `pass_precision`, `pass_recall`, `wrong_recall` (fail-recall on rubric), `too_strict` (label
  correct, judge partial / wrong) and `too_lenient` (judge correct, label partial / wrong), both shares of ALL labelled rows;
  `kappa` = linear-weighted on wrong < partial < correct (`weighted_kappa`; Cohen on pass / fail); Wilson CIs at the Kish n;
  `insufficient` when n < 20 or < 5 labelled correct / < 5 not-correct; per stratum population / labelled / skipped + skip note;
  `tasks` / `rubric_shas` mix; `self_rows` / `self` (a reference labeller whose model is the judge — both sides through
  `safe_backend_label`, `cloud:` prefix ignored, so `ollama:` / `command` / `teacher:` labels match too — is never graded on its own
  labels). Trust (`judge_trust`): `below_floor` when the strict-agreement Wilson UPPER bound <
  `STRICT_AGREEMENT_FLOOR` (0.7) or the too-strict / too-lenient Wilson LOWER bound > `TOO_STRICT_CEILING` / `TOO_LENIENT_CEILING`
  (0.15) (`flags`), else `calibrated_ok`; `insufficient`; `uncalibrated` (none, or no labels file: computed without reading any
  predictions). `both`: either tier sufficient = calibrated, the worse wins. `cached_report` keys on the labels file + every
  predictions file's (mtime, size), LRU 64.
- **API** (`api/eval_judge_calibration.py`, `/projects/{id}/eval-judge/calibration/`): `GET labels` (per scale: population, judges +
  verdict counts, per-tier labelled / skipped / by_label), `GET sample?scale&judge&tier&n (1-100)&seed`, `POST labels {content_key,
  scale, label, judge?, reason?}` (human only; 404 unknown key; 400 wrong vocabulary), `POST skips`, `DELETE labels/{key}?scale&tier`,
  `GET report?scale&tier` (+ `trust` per judge), `GET trust?judge&scale`, `POST reference-labelling?async_job=true {backend?, scale,
  judge?, n, seed}` → Job `eval_judge_calibration_reference`: blind, the answer judge's own machinery
  (`_judge_from_synth_backend`: same system prompt / schema, with the remediation or code-example instruction per `task`), or for
  `rubric` `build_rubric_prompt` + `score_item` with the project's CURRENT rubric (a row graded under another `rubric_sha` gets a
  `rubric_changed` skip); writes only the `reference_model` tier, `labeller` = `safe_backend_label(describe())`; unreadable answer
  → skip marker, failed call → nothing (retried next run); 0 labelled raises; cloud pins go through
  `_resolve_cloud_backend_override` (budget guard); 409 `EVAL_CALIBRATION_REFERENCE_ALREADY_RUNNING` (in-flight check under a
  per-loop asyncio lock). The label source is the shared `judge_calibration.label_source` (`/synthetic/calibration/settings`).
  `GET /eval-judge` adds `trust {verdict, rubric|null}` for the judge in use; `GET /evaluation/summary` adds `judge.trust` (both
  additive, never fail the read).
- **UI**: `EvalJudgeTrustBadge` ("calibrated" / "below floor" / "too few labels" / "not calibrated" + "judge calibrated vs human:
  agrees on correct vs not 90% (n 40; exact 84%); too strict 6%, too lenient 3%" — the headline is the strict agreement the floor
  is checked on, exact agreement is secondary; a below-floor line names the limit: "agreement below the 70% floor" / "too strict
  over the 15% limit"; or "judge not calibrated: label 20 answers") on Settings → Judge model
  (`JudgeModelCard`, plus a "Plan rubric:" line when the judge graded plans) and on `EvalSummaryCard`'s judge line (with a
  "Calibrate the judge" link to `/settings#eval-judge-calibration` unless calibrated). `EvalJudgeCalibrationCard` ("Calibrate eval
  judge") sits inside `JudgeModelCard`: calibration is per judge and pooled over every eval, so it lives with the judge choice, not
  under one run; collapsed and fetch-free until opened (opens itself on that hash). Label-source radios, a scale picker (only
  when rubric answers exist), a judge picker (when >1), a blind labeller (question / reference answer / model's answer;
  Correct / Partial / Wrong or Pass / Fail, Skip, Stop, reason), reference labelling (`GenerationModelPicker` + n) and per-judge
  cards per tier. Bell: deep link to the settings anchor, summary "labelled N of M answers · correct · partial · wrong · K skipped
  · by <labeller>". Tests `tests/test_eval_judge_calibration.py`, `EvalJudgeCalibrationCard.test.tsx`, `JudgeModelCard.test.tsx`,
  `EvalSummaryCard.test.tsx`, `NotificationBell.test.tsx`. Not done: external-benchmark / pipeline evals do not write the file.

## Compare: row-paired evidence (Epic W1)

- `compare_experiments` (Results → Compare) still picks each side's newest EvalResult with a pass_rate; it now also returns
  `row_evidence` and `a.provenance` / `b.provenance` (additive; built in `compare_row_evidence_service`, best-effort: an
  exception leaves `row_evidence: null`). Pairing is `paired_comparison_stats.pair_row_scores(before_rows, after_rows,
  score_key)` → `{before, after, only_before, only_after, unscored, same_order}` (same keys + order pair positionally, else
  first occurrence by key; None scores drop the row and count as `unscored`); `sft_lift_summary_service._paired_row_scores`
  now wraps it (unchanged behaviour). Metrics: every metric both results report that is in `_ROW_SCORE_FOR_METRIC`,
  headline-first in `_PREFERRED_HEADLINE_METRICS` order, one per score key (exact_match/accuracy share `correct`); the
  headline = first that pairs. Verdict = `paired_difference_evidence` (min 5 rows, paired t; + exact McNemar on 0/1 scores) + `not_comparable`.
  `reason`: `missing_eval` / `not_comparable` / `no_row_scores` ("no per-row scores (evaluated before per-row capture)") /
  `no_row_metric` (e.g. macro-F1 / perplexity only).
- **Comparability** (`compare_provenance`): not comparable when any differs — answer key (dataset type; else file sha256;
  else split version on the same dataset), judge (`metrics.judge.judge`, a skipped judge = none; `metrics.rubric` judge +
  `sha256` when `status == "judged"`; `metrics.code_example_judge.judge`), scoring (`eval_type` FAMILY — `exact_match` and
  `f1` are one reference-matching scorer: the lift check labels a run `exact_match` via `eval_type_for_project` where a
  manual held-out eval of the same file may say `f1`, pairing only uses metrics both report; `llm_judge` / `safety` /
  `perplexity` stay distinct — plus `metrics.scoring_mode` and the remediation / structured-plan / tool-call fingerprints
  when both carry one). Unknown answer key = a note. `certain: false` when every difference rests on an inferred answer key:
  still no verdict, worded "answer key may differ: split v3 vs v5 (v3 inferred from eval time)" and "may have been scored
  differently". Decoding (`metrics.inference.decoding`, else `details.inference.decoding`) never blocks: different settings,
  or one side predating recorded decoding, add a `caveats` line.
- **Answer-key identity**: `run_heldout_evaluation` stamps `details.dataset.sha256` (file read) + `details.dataset.version`
  (newest `DatasetVersion` of that dataset whose manifest `sha256` matches; None for an edited file / gold sets) via
  `answer_key_stamp`, before inference. Older results: version inferred = newest DatasetVersion of `details.dataset.id`
  created at or before the eval (`version_source: "inferred"`); its manifest hash goes to `inferred_sha256`, never `sha256`.
- **Content hash (B1, 2026-11-12)**: the stamp also records `details.dataset.content_sha256`
  (`dataset_service.split_content_hash`: rows re-read from the file, only `_source_document_id` / `_source_document`
  stripped, canonical JSON, order kept; kept for `scoring_mode == "remediation"`, whose rows pick passages by source
  document) and `eval_input_sha256` (content + the canonical `EVAL_INPUT_MANIFEST_KEYS` subset of `prepared/manifest.json`:
  only the fields the eval reads — task_profile, output_schema, labels, subtask, tgt_lang / target_language; the adapter
  wiring is already applied to the rows and the chat template comes from the tokenizer; pinned by
  `test_eval_input_keys_are_exactly_the_manifest_fields_the_eval_reads`). Undecodable lines hash as raw bytes, lone
  surrogates via `surrogatepass`; `_write_prepared_splits` hashes in `asyncio.to_thread`. The version match falls back
  to the manifest `content_sha256`. `_answer_key_difference`, BEFORE the "split vA vs vB" branch: file shas differ but
  eval-input hashes (both recorded) or content hashes match → same answer key + the note "test file was re-stamped (metadata
  only)" (`RESTAMPED_NOTE`; flows through `comparability.notes`); same content, different eval input → "different eval
  input"; same file sha with both eval-input hashes recorded and different → "different eval input" too. Older results:
  a recorded file sha without content is looked up among the live file + retained `prepared/versions/*/` snapshots by
  file sha (`identity_for_file_sha`: file sha first, content hash only for the match, LRU-memoised in memory by
  path/mtime/size, never written back; pruned snapshot → today's file-sha rule) and its eval-input hash is taken with
  the `manifest.json` beside the matching file (the snapshot's own); the inferred path takes the inferred version's
  manifest `content_sha256` (+ `inferred_eval_input_sha256` from the snapshot). When no content hash can be had for a
  side, identical `row_scores.keys` (`row_keys_sha256`: the keys IN ORDER, None when any key repeats, since a key is
  prompt + reference only) mean the same rows evaluated (`SAME_ROWS_NOTE`: "Same rows evaluated …; the split files
  differ" — it doesn't claim the same answer key, a `max_samples` eval may have read only a shared prefix) — never for
  remediation. Other file-sha readers are unchanged in B1:
  `documents_qa_flow_service` flow summary `test_sha256` / `val_sha256`, the data-studio lineage `manifest_file_hash` check,
  `answer_key_label`. Tests: `tests/test_split_content_hash.py`.
- **Re-score** (`rescore`): for not-comparable the side with the older eval; for missing row scores the side lacking them.
  `available` = `auto_lift_skip_reason(manual=True)` is None and the other side was scored on the test examples
  (`AUTO_LIFT_DATASET`); the button (after a GPU confirm) posts the existing `POST /evaluation/summary/lift-check` (base +
  run on the CURRENT test split and judge). Otherwise `instruction` text (baseline / simulated / not completed / no weights /
  other answer key). `in_progress` + `job_id` when a `post_training_lift_eval` Job for that run is QUEUED / RUNNING
  (`post_training_eval_service.active_lift_job_id`; the UI shows a disabled "Re-score running"). `start_post_training_lift_job`
  returns `{started: false, skipped_reason: "already_running", job_id}` then (the training watcher just records it) and the
  lift-check endpoint answers 409.
- UI: `components/evaluation/CompareRowEvidence.tsx` at the top of `ProjectEvalComparePage` (headline "B is within noise of
  A: 2 better, 7 worse, 69 same of 78 rows; change −6.4 pts (95% CI −14 to +1)", labels as `liftEvidence`, other metrics,
  differences, re-score, provenance per side). The pass-rate winner / "Fix the gap" banner is unchanged (W4 gates it).
  Tests: `tests/test_compare_row_evidence.py`, `CompareRowEvidence.test.tsx`, `ProjectEvalComparePage.test.tsx`.

## Paired verdict rule: interval + exact McNemar on 0/1 scores (2026-11-13)

- `paired_comparison_stats.paired_difference_evidence` is the one verdict for every paired before/after comparison (lift
  check, Compare row evidence, Auto-RAG / passages / reader checks, the retrieval sweep). Rule: `better` / `worse` when
  the paired-t 95% interval on the mean per-row change excludes 0 **and**, when every paired score is 0 or 1 (correct /
  incorrect, pass / fail), the exact two-sided McNemar p (`exact_mcnemar_p(better, worse)`: binomial(b + w, 0.5) on the
  changed rows, integer arithmetic) is < 0.05 (`MCNEMAR_ALPHA`). Otherwise `within_noise`. Why: the t-interval is liberal
  with few changed rows — 4 better / 0 worse on 50 rows gives +0.2 to +15.8 while p = 0.125; the smallest lift it calls is 6
  changed rows all one way (p = 0.031); split counts need more (8 / 1 p = 0.039, 12 / 3 p = 0.035).
- Continuous scores (F1, judge 0 / 0.5 / 1 with any partial, rubric means) keep the interval-only rule; one non-0/1
  score anywhere makes the pairing continuous. `seed_spread_evidence` passes `binary_guard=False` (seed values are
  rates, not per-row outcomes).
- Payload: `mcnemar_p` (float, or None when not binary; recorded even for `too_few_rows`) is added to every evidence
  dict, to Compare's top-level `row_evidence`, to the lift Job result's `evidence`, and to the passages / reader
  verdict evidence subsets. The reader model card prints it and says when it held a lift back.
- UI: `liftEvidence.evidenceNote` (EvalSummaryCard, Auto-RAG cards) explains a held-back lift ("only 4 rows changed …
  exact McNemar p = 0.125 … at least 6 changed rows (6 better, 0 worse is the smallest that counts), and more when they split both ways") and adds "(exact McNemar p = …)" to a gain / drop;
  `shortEvidenceText` (SftLiftPanel rows) and the bell's lift line add "McNemar p = …"; `CompareRowEvidence` adds it to
  the headline CI. Tests: `tests/test_paired_comparison_stats.py::McNemarGuardTests`, `liftEvidence.test.ts`.
- Speed: `exact_mcnemar_p` sums incrementally (each C(n, i+1) from C(n, i), exact ints) up to
  `_MCNEMAR_EXACT_SUM_MAX_N` = 2,000 changed rows; above that it multiplies the exact peak term C(n, k) / 2**n by a float
  sum of shrinking ratios (agrees with the exact sum to ~1e-15 relative). 40,000 changed rows take ~15 ms (the first
  version summed `math.comb` per term: ~2 min). `McNemarSpeedTests` pins identity on the 0..60 grid, 1e-12 above 2,000,
  and a loose time bound.
- The retrieval sweep's `vs_current` subset (`scripts/auto_rag_ab.pick_best_retrieval`) carries `mcnemar_p`;
  `liftEvidence.vsCurrentText` (EvalSummaryCard sweep line, AutoRagComparisonPanel table) shows "McNemar p = …" and,
  when McNemar held a one-way change back, ": too few changed rows to count".
- **Stored verdicts recorded before the guard** (lift Job results in the bell, passages / reader records, sweep arms) keep
  their interval-only better / worse. They are recognisable: no `mcnemar_p` key at all (newer evidence always has it,
  None for continuous scores). `liftEvidence.preGuardNote` appends "recorded before the McNemar check" to such a better /
  worse in the bell (lift, passages, Auto-RAG lines) and the sweep text; the reader model card adds a sentence asking
  for a re-run. Not recomputed: the stored counts can't tell whether the scores were 0 / 1. Live surfaces (eval
  summary, Did-SFT-help, Compare) recompute and need nothing.

## Experiment scorecard verdict follows the gates (2026-11-13)

- `GET /evaluation/scorecard/{id}` (only consumer: `ScorecardPanel`; export blocking is separate, in `export_service`)
  used to start from SHIP and flip only when the project's `gate_policy` (`must_pass` / `blocked_if_missing`) enforced
  it. The project default is advisory, so a simulated run with every rule "Not measured" read **SHIP** beside a FAIL
  pass/fail card. Now `api/evaluation.scorecard_decision(report, policy)`: `SHIP` only when at least one check has a
  measured `actual`, `report.passed` and no `missing_required_metrics`; `NOT EVALUATED` when no check is measured (or
  there are none); else `NO-SHIP` with the reasons. Optional rules that read "Low" don't block (same as `report.passed`).
  The policy only sets `enforced`; advisory projects get `policy_note` "Advisory policy: this verdict does not block
  export." Additive fields: `enforced`, `policy_note`; `decision` gains `NOT EVALUATED` (amber badge,
  "Not evaluated yet:" heading). Tests: `tests/test_experiment_scorecard_verdict.py`, `ScorecardPanel.test.tsx`.

## Detectable difference, every task shape (Epic W2)

- `services/eval_sizing_service.py` (pure) generalises the span-F1 sizing (`span_eval_sizing_service`, unchanged —
  its own 95%-CI-only `8·p(1−p)/Δ²` rule stays) to any per-row metric in `_ROW_SCORE_FOR_METRIC`, at 95% confidence
  AND 80% power: `Z = 1.959964 + 0.841621 = 2.801585`. `detectable_gap` / `rows_needed` always mean a gap BETWEEN TWO
  RUNS: `Z·s_d/√n`, `⌈(Z·s_d/Δ)²⌉`. **Paired** (`paired_sizing(before, after)`; lift, A-vs-B, Auto-RAG): `s_d` = SD
  of per-row differences. **Single** (`sizing_from_values`, `eval_result_sizing(er, metric_id)`): conservative
  unpaired bound `s_d² = 2·s²` (binary `s² = p(1−p)`, continuous = sample variance); the one-sample precision
  (`Z·s/√n`) is API-only as `one_sample_gap` / `one_sample_rows_needed`. `TARGET_GAP = 0.05` (`gap=` overrides).
  Degenerate: n < 5 → `too_few_rows` (no gap; worst case single `2·0.25` → 1570 rows, paired `d ∈ [−1,1]` → 3140);
  constant single side → `all_pass` / `all_fail` / `zero_variance` via Agresti–Coull `p̃ = (Σx+2)/(n+4)` (a heuristic
  for constant continuous scores); zero spread of differences → `identical_scores` or `constant_shift`, figure = the
  labelled unpaired bound `s_before² + s_after²`. `line` rounds the gap UP to whole pts and rows UP (to 10 < 1,000,
  else to 50); "enough" lines name only the target. 78 rows at 23/78 pass → single: gap 0.2046 (~21 pts), 1306 rows
  (~1,350); paired 6 better / 8 worse → gap 0.1350 (~14 pts), 569 rows (~570).
- Wired (all additive keys): `sft_lift_summary_service._attach_row_evidence` → `evidence.sizing` on every lift row;
  `compare_row_evidence_service.build_row_evidence` → `metrics[].sizing` + top-level `sizing` (headline); `auto_rag.
  _lift_evidence` → `sizing`; `eval_summary_service` → summary `sizing` (single, the headline metric of the latest eval;
  seed group = the median seed's eval; head eval = accuracy; span eval / passages = none — span keeps `SpanSizingTable`).
- UI (`liftEvidence.sizingToShow`): with paired evidence, the paired line ONLY when within_noise / too_few_rows (never
  beside a significant verdict), merged into the evidence sentence on `EvalSummaryCard` (`eval-summary-sizing` span,
  `.eval-summary__sizing-inline`); without paired evidence, the single line as its own muted `<p class=
  "eval-summary__sizing">`. Compare (`CompareRowEvidence`) shows `ev.sizing.line` only when within_noise /
  too_few_rows with no `reason` (`eval-compare-evidence-sizing`). Tests: `tests/test_eval_sizing.py` (hand-computed),
  `test_eval_summary.py`, `test_compare_row_evidence.py`, `EvalSummaryCard.test.tsx`, `CompareRowEvidence.test.tsx`.

## Checked decoding: verify-and-retry at eval time (Epic Y1 + Y4)

- **Opt-in, off by default** (`services/checked_decoding_service.py`). Setting `runtime_config["inference"]
  ["checked_decoding"] = {enabled: false, mode: "resample"|"repair", max_attempts: 3 (1..5), verifiers: null (auto) |
  [ids], sample_temperature: 0.7 (0 < t <= 2, resample), seed: 0}`; `normalize_policy` validates (`PolicyError`, a
  ValueError). `GET/PUT /projects/{id}/evaluation/checked-decoding` reads / saves it; PUT is a partial update merged over
  the saved setting (`merged_project_policy`; explicit `null` resets a field, `verifiers: null` = auto; the merged result
  is validated, 422 on a bad value, nothing saved). Per-eval
  override: `checked_decoding` on `POST .../evaluation/run-heldout` (and its async Job params) and on `POST
  .../evaluation/summary/lift-check` (both 422 via a pydantic validator); an override REPLACES the project setting
  (`{"enabled": false}` turns it off for that eval). `run_heldout_evaluation(checked_decoding=...)` resolves it
  (`resolve_policy`); perplexity evals ignore it.
- **Loop** (`evaluation_service._run_transformers_inference(checked=...)` + `_checked_retry`): attempt 1 is exactly the
  normal generation (same kwargs, same code path). Only when a verifier fails: `resample` = the same formatted prompt,
  `generation_kwargs(temperature=sample_temperature)` (neutral sampling), `torch.manual_seed(attempt_seed(seed, row_index,
  attempt))`; `repair` = history (or the prompt) + `{assistant: attempt N-1}` + `{user: "Your previous answer failed these
  automatic checks:\n- <verifier>: <message>\nReturn a corrected answer in the same format."}` through the chat template,
  same decoding as attempt 1 — without an applied chat template a repair attempt resamples (`modes_used`).
  `resample_on_repeat` (default false): a retry whose `.strip()`ped text equals any earlier attempt's is a repeat; with
  the flag on, the NEXT attempt calls `run_checked_row(resample=...)` (eval passes `_resample`) instead of `regenerate`.
  Record `repeats` and summary `rows_with_repeats` exist when > 0 or the flag is on; `policy_record` adds
  `resample_on_repeat: true` + `sample_temperature` / `seed` only when on AND mode is repair (resample mode ignores the
  flag, so its key never changes). Repair-mode records also carry `repair_messages: 2` (`REPAIR_MESSAGES_REVISION`, the
  clearer code-shape / fence messages): repair-mode results cached before it are not reused as a bar. UI: checkbox
  "If a retry repeats the same answer, sample a new one" in `CheckedDecodingSettings` (repair mode only). First passing
  attempt wins; none pass → attempt 1 kept. A retry generation that raises stops that row's retries and keeps attempt 1
  (`attempts_detail` gets `{attempt, error: <type name>}`, record `retry_error`); a CUDA OOM also runs
  `torch.cuda.empty_cache()` (guarded). The torch CPU + CUDA RNG state is saved / restored around every retry generation,
  so retries never shift later rows' sampling. Replayed dialogue turns are never checked. Only the plain causal-LM path runs
  it: GGUF / classifier head / token classification / seq2seq / multimodal → `metrics.checked_decoding.skipped =
  "engine_unsupported"` and no policy in the decoding record.
- **Verifier interface = the safety rule.** A verifier gets `(VerifierInput, answer)`. `build_verifier_input(pair,
  eval_spec)` reads ONLY `prompt` / `history`, `code_example_row.language` + `.passages`, `plan_row.passages` (never
  `acceptable`), `classification_candidates` (kept only when the prompt carries the classification handler's explicit
  enumeration, `ClassificationHandler.candidate_list_line` — never when the labels merely appear in the input text or
  there are more than `LABEL_LIST_PROMPT_CAP` candidates) and the eval's DECLARED
  output schema (`output_json_schema(manifest)`; field_match only, never derived from references). `VerifierInput` has no
  field for a reference / answer key / judge output. Test: `test_reference_never_reaches_a_verifier_end_to_end` (sentinel in
  the reference + reference-like fields, spy verifier). Verifiers (generic, by answer shape): `code_checks` (code-example rows
  with a declared language: `code_check_service.checks_for_language` runnable checks on the JSON `code` field, context
  `{"path": None, "grounding_text": <prompt the model saw>}`; never requests `context_grounding` itself; messages only:
  a dict/list `code` says it must be a string holding the <language> source, and when the checks FAILED on a `code`
  starting with ```` ``` ```` a fence hint is the last message, kept inside `MAX_MESSAGES_PER_FAILURE`), `context_grounding`
  (opt-in even in auto mode: `run_heldout_evaluation` stamps `grounding_opt_in` on the resolved policy via
  `with_grounding_opt_in` — true only when the id is in `CODE_CHECK_ALLOWED`, in the project's code-check selection
  (`runtime_config.remediation.checks` / the manifest's `output_schema.remediation.checks`), or named in the policy's
  `verifiers`; then the policy record carries `auto_includes_grounding: true` so the lift cache keys on it; skipped
  `ok: None` is not a failure), `json_schema` (declared field types, null always allowed, only the schema's own `required`
  keys — none when it declares none), `label_set` (the WHOLE answer, trimmed + casefolded + surrounding quotes /
  punctuation stripped, equals one label: "Alpha." passes, "alpha or beta" fails), `citations` (every cited id is a
  prompt passage id; `[]` passes). A row with none → `skipped: "no_verifiers"`,
  no retry.
- **Records.** Per row (prediction dict, `predictions_preview`, `eval_predictions/<id>.jsonl`): `checked_decoding: {attempts,
  chosen_attempt, attempts_detail: [{attempt, failed: [{verifier_id, messages (<=3, <=300 chars)}]}], verifiers, passed,
  extra_generation_seconds, extra_generated_tokens, verify_seconds}` (+ `chosen_stats: {generated_tokens, latency_ms}` of the
  chosen attempt when it is not attempt 1; the row's top-level `generated_tokens` / `latency_ms` stay attempt 1's). Eval: `metrics.checked_decoding` (= `metrics.inference.
  checked_decoding`) `{policy, label, rows, rows_checked, rows_without_verifiers, rows_failed_first_try, rows_retried,
  rows_fixed_by_retry, rows_still_failing, mean_attempts, extra_generation_seconds, extra_generated_tokens, verify_seconds,
  generation_seconds_total}`. **Latency / throughput / `generation_seconds` / per-row `latency_ms` + `generated_tokens` stay
  attempt-1 numbers**; `generation_seconds_total` includes retries and checks. The policy (`policy_record`: mode,
  max_attempts, verifiers or "auto", + temperature / seed for resample) is `metrics.inference.decoding.checked_decoding`.
  With the setting off none of these keys exist.
- **Lift check**: `run_post_training_lift_eval(checked_decoding=...)` resolves the policy ONCE (`lift_policy_override`, always
  an explicit dict) and passes it to the base AND fine-tuned (and every seed) eval. The base-model cache
  (`_baseline_decoding_matches(result, decoding, checked)`) requires the same `decoding.checked_decoding` record (None = off):
  an unchecked base result is never the bar for a checked run and vice versa. The span and head-only branches get the
  same resolved policy (their engines record `engine_unsupported`). Passage-trained runs are scored by the passages harness,
  which has no checked decoding: `POST .../summary/lift-check` answers 422 to an explicit enabled override for such a run
  (`checked_decoding_unsupported_reason`); with only the project setting on, the Job result carries `checked_decoding:
  {policy, label, skipped: "run_type_unsupported"}` (one resolution for both arms) and the bell says "checked decoding not
  applied (passages check)".
- **Compare (Y4)**: `compare_row_evidence_service._scoring` splits `checked_decoding` out of `decoding`; different policies →
  difference `kind: "decoding_policy"`, "different decoding policy: checked decoding off vs up to 3 tries · resample" →
  `not_comparable` + the usual re-score offer. Provenance carries `checked_decoding` (`checked_decoding_summary`).
  `sft_lift_summary_service` adds `baseline/trained.checked_decoding` (only when present) and `checked_decoding_differs`
  (a side with any `skipped`, e.g. `engine_unsupported`, counts as policy None — it ran unchecked);
  `eval_summary_service` adds `checked_decoding` / `checked_decoding_differs` (only when present).
- **UI**: `CheckedDecodingSettings` (Results eval settings in `EvalPanel`, under the per-turn judge toggle: "Checked decoding"
  toggle, "When a check fails" = try again / ask the model to fix the listed problems, "Max tries per row" 1..5; saves the
  project setting). `CheckedDecodingBadge` ("Checked decoding · up to N tries · mode" + "N rows fixed by a retry" + source
  eval) on `EvalSummaryCard`, `SftLiftPanel` and each Compare provenance row; a "not like for like" line when the lift's two
  sides differ. Tests: `tests/test_checked_decoding.py`, `tests/test_post_training_lift_eval.py`
  (`test_checked_decoding_one_policy_for_both_evals_and_the_cache_keys_on_it`), `CheckedDecoding.test.tsx`. Not done (Y5):
  Playground / local chat / export parity.

## Eval model memory release

- Held-out / lift / probe / perplexity evals load the model fresh on every call (no cache across calls or Jobs); the
  sync paths that load one — `_run_local_inference` (every engine), `_run_transformers_inference` (also called directly
  by the generative probe `predict_fn`) and `_compute_perplexity` — carry `@_releases_inference_memory`. When the
  OUTERMOST of them returns or raises (thread-local depth, so the dispatcher → engine nesting releases once) it calls
  `release_inference_memory()`: `gc.collect()`, then `torch.cuda.empty_cache()` + `ipc_collect()` when a CUDA context already exists (`is_initialized()`; never creates one),
  then glibc `malloc_trim(0)` (best-effort, never raises). On an error it first runs `traceback.clear_frames` on the
  exception, so the exception the Job runner records doesn't pin the model through its frames' locals.
- Why: loading a checkpoint can import modules that keep a frame in a self-cycle (`inspect.currentframe()` held in a
  local at import time — e.g. the first load of an architecture in the process). Its `f_back` chain reaches the eval's
  frame, so the model + tokenizer survived the return and were freed only by the cyclic GC, whose object-count
  thresholds never notice a 50+ GB model — a second eval in the same uvicorn / celery process stacked a second copy
  (OOM-kill on a unified-memory GPU box).
- `_run_transformers_inference` also empties its `model` / `tokenizer` cells in a `finally`: the per-row
  `_generate_once` closure shares them, and a traceback frame keeps its function (and closure) alive even after
  `frame.clear()`. Results, defaults and in-call reuse (checked-decoding retries, replayed dialogue turns) are unchanged.
- Cancel: the row loop has no cancel check and `to_thread` work isn't interrupted, so a cancelled eval frees its
  model only when its generation loop ends (every row still runs). An eval started right after a cancel can briefly
  hold two copies.
- CUDA context before the load: every generation / perplexity loader (`_run_transformers_inference`, the PEFT causal,
  seq2seq and sequence-classification adapter loaders, `_compute_perplexity`) calls `_init_cuda_context_before_load`
  right after `use_cuda` is resolved, before `from_pretrained`. On unified-memory GPUs a ~57 GB CPU load plus the
  weight files' page cache (not reclaimed by the driver) left too little free memory, and the context created by the
  later `.to("cuda")` failed with `CUDA error: out of memory`. No-op without CUDA; a failure is left to `.to("cuda")`.
  Same fix in `train.py` (`agent-notes/training.md`). Tests: `tests/test_eval_cuda_context_early.py`.
- Tests: `tests/test_eval_model_release.py` (automatic GC off; a fake loader reproduces the frame cycle; weakref to the
  fake model is dead after a call, after a failing call, after a `heldout_evaluation` Job that succeeds or fails, and a
  second Job's load never sees the first model alive).

## Judge resilience: unjudged rows, circuit breaker (A1)

- **Guard** (`services/judge_call_guard.py`): `classify_judge_error` -> transient (`SynthBackendTransientError`,
  `CloudLlmError.transient` / `status_code` 408/425/429/5xx (409 is permanent), httpx `TransportError` / 5xx `HTTPStatusError`, walking
  `__cause__` through `SynthBackendError` wrappers) or permanent (everything else). `JudgeCallGuard` (one per judge loop)
  retries transient failures `JUDGE_TRANSIENT_RETRIES` times (backoff 10 s x 3^n +-20%; 0 for a backend with
  `retries_internally`, i.e. the command backend, carried on `ResolvedJudge.retries_internally`); its `JudgeCircuit` is
  **per run** (each guard owns one; no registry, nothing carries over, so every new eval / judge loop starts closed and its
  first calls get the normal retries -- a stale outage can't write off a new eval on one blip) and opens after
  `JUDGE_BREAKER_CONSECUTIVE` (3) ROWS in a row failed transiently (after their retries; a permanent failure resets the count)
  or > `JUDGE_BREAKER_ERROR_RATE` of the last `JUDGE_BREAKER_WINDOW` calls. Mode `stop` (default): the run's remaining
  uncached rows -> `judge_unreachable`, no call; cache hits still apply. Mode `wait` (opt-in, per project; only when the guard
  has a Job handle on its own loop, else stop -- a celery workflow eval has none): probe every `JUDGE_OUTAGE_PROBE_SECONDS`,
  Job message "judge unreachable (...) — retrying ~HH:MM ET", resume on success, else stop once the budget is spent. The
  budget is WALL time (sleeps + probe calls, `clock`), ONE per Job (`JobProgressHandle.judge_wait_spent_seconds`, shared by
  every judge loop and both lift legs; a probe under way finishes, so at most one call over), capped at
  `max_wait_cap_seconds()` = half `CELERY_VISIBILITY_TIMEOUT_SECONDS` (6 h by default; the API's `le`, the UI's choices).
  An open circuit is half-open after `JUDGE_BREAKER_COOLDOWN_SECONDS` (matters in wait mode). Permanent errors and
  unparseable replies never count. `LlmBudgetExceeded` / cancel propagate. `JUDGE_BREAKER_ENABLED=false` (or project
  `breaker: false`) = the march before the guard: one call per row, **no guard retries** (`resolve_resilience` sets
  `retries=0`), never stopped; the command backend's own CLI retries still apply, as before; unjudged rows still get
  their reason and D2 scoring.
- **Opt-in per call site** (`resilience=` kwarg on `judge_predictions` / `judge_retrieval` / `judge_dialogues`, `guard=` on
  `judge_rows_with_rubric`; default None = unchanged). ON: held-out eval only — `_safe_judge_answers(resilience=True)` (answer
  + remediation + retrieval), `_judge_code_example_predictions`, the dialogue judge, the structured-plan rubric. OFF: pipeline
  eval (`_run_heldout_judges`), Auto-RAG (`auto_rag_ab`, thread path; it only copies `status` / `unjudged_reasons` /
  `coverage`), the teacher-answer filter (unjudged answers pass through, so a breaker skip would be a silent pass — it may
  adopt the guard only once it fails loudly below a coverage floor), probe / review-assist / calibration / relabel judges.
- **Threads**: the rubric grades in `asyncio.to_thread`; it gets a `threaded=True` guard: stop mode only, never touches the Job
  handle (whose methods open sessions on the main loop's engine), cancel via a `threading.Event` that `_watch_cancel` sets from
  the main loop (`JudgeCancelled` -> `CancelledError`). The async guard uses the Job handle only when running on the loop that
  created it (`JobProgressHandle.loop`); so does the command backend (`command._job_handle_here`: the cancel poll during a
  CLI call and the usage-window pause messages are skipped on a worker thread's loop).
- **Per-row state, only on unjudged rows** (a run without failures is byte-identical): `row_judge_status: "unjudged"`,
  `row_judge_unjudged_reason` (`judge_unreachable` / `transient_error` / `unparseable` / `permanent_error`; the rubric also
  `no_judge`),
  `row_judge_error` (<=200 chars); retrieval `row_retrieval_*` likewise; rubric `row_rubric_unjudged_reason` /
  `row_rubric_error`; code examples `_code_example_judge.unjudged_reason`. Written by every `judge_predictions` caller (the
  judge was asked) -- so the unguarded callers changed too: the external-benchmark failures list drops unjudged rows, and
  Auto-RAG / pipeline-eval rows carry `row_judge_*` keys; only the guard (breaker / retries) is opt-in. **Snapshot, only when unjudged > 0**: `status`
  (`partial` | `paused` = breaker stopped it), `unjudged_reasons`, `coverage`, `breaker {tripped, tripped_at_row (0-based),
  consecutive, last_error_kind, last_error, mode, waited_seconds}`.
- **Scoring (D2, new evals only; stored results untouched)**: `_prediction_unscored` (new) = a scored code example with
  `row_code_example_pass` None for ANY reason (judge off, `disabled_for_project`, no judge, no verdict — was `correct=1`), or
  `row_judge_status == "unjudged"` where `judge_decides_row`. `_prediction_failed` returns False for the latter (was the
  exact-match fallback). `row_scores.correct` is then None (pairing skips it), the row is out of `failed_count` /
  `failures_preview`, and `eval_predictions` records omit `correct`. `EvalResult.pass_rate` leaves them out too
  (`_pass_rate_without_unjudged`: the same row mean the pass rate falls back to -- `row_exact_match`, else `row_f1` -- over the
  other rows; None when every row was left out; `metrics.pass_rate_unjudged_excluded` = N; only when such rows exist, and not
  for a handler's own `pass_rate`). The `exact_match` / `f1` METRICS stay over all rows. Rows the judge was never asked about
  keep exact match (`test_answer_judge.py` pins both). Pipeline eval: a `correct` None row -> `unscored_reason:
  "row_unscored"`, except a code example with `row_code_example_pass` None while an answer judge ran, which `_needs_judge`
  labels `needs_judge` first (still unscored, never correct).
- **Re-judge inputs** (for A2, written now): `eval_predictions` records get `judge_pending {status, reason, task, label,
  content_key, question?, reference?, prediction?}` for unjudged rows and `checks_passed` (parses + checks + citations) for
  scored code examples.
- **Lift**: `_baseline_judge_matches` refuses a base result only past a threshold
  (`judge_call_guard.snapshot_blocks_base_reuse`: breaker stopped it (`paused` / `breaker.tripped`), judged 0 with unjudged > 0,
  or more than `BASE_REUSE_MAX_OUTAGE_SHARE` = 5 % of its rows lost to an outage) -> the base is re-run (A2 replaces this with
  a no-GPU re-judge). A blip below the floor or unparseable-only gaps still reuse (those rows just don't pair). With no judge
  resolving now the check is skipped on purpose (the fine-tuned eval isn't judged). Lift rows: `evidence.excluded_unscored`
  (rows both evals ran that one left unscored, from `pair_row_scores.unscored`) and, for judge-based metrics
  (`judge_correct` / `code_example_pass_rate` / `structured_plan_pass_rate` / `rubric_*`), `judged_subset {baseline?,
  trained?: {judged, unjudged, outage}}` when that side's block (written with reasons) left rows unjudged. The Results
  headline (`EvalSummaryCard`, `SftLiftPanel`) says "on judged rows only — fine-tuned 40 of 688 judged (judge unreachable)";
  evidence lines add "N excluded (no score on one side)".
- **Job results** (`job_result_fields`, only for a judge OUTAGE on a block written with reasons -- `judge_outage`: paused /
  tripped or any `judge_unreachable` / `transient_error` row; never for results stored before reasons existed, the rubric's
  own `status: "judged"`, or unparseable / refused-only gaps): `judge_status` ("paused" / "partial"), `judge_unjudged`,
  `judge_judged`, `judge_unjudged_reasons`, `judge_label`, `judge_kind` ("answer judge" / "code-example judge" / "rubric
  judge" / "dialogue judge"), `judge_eval_result_id` — on `heldout_evaluation` (`_summarize_eval_result`), the lift check
  (both legs, seeds, head-only) and workflow eval steps. Bell `judgeWarningText` is APPENDED to the outcome, never replacing
  it: "<outcome> · with a warning: judge incomplete by X — N of M rows unjudged (...) (eval #id)" (+ " — re-run the eval
  before relying on this lift" on a lift check, whose verdict stays); "Done, with a warning: …" when there is no outcome.
- **Summary**: `_judge_block` now emits for `score: null` when the judge was asked about rows (every row unjudged in an outage)
  and passes `status` / `unjudged_reasons` / `coverage` / `breaker`. `EvalSummaryCard` + `unjudgedNotice.ts`: counts by reason,
  where it stopped, judge label + eval id, "re-run the eval when the judge is reachable again" (no re-judge promise); no notice
  for results stored before reasons existed (no `status` / `unjudged_reasons`). The summary's `unjudged_notice`
  (`judge_call_guard.unjudged_notice_block`) covers every guarded judge -- answer, code-example, rubric, dialogue (an outage
  block first) -- with `kind`; the card words it per judge (what the rows are, what they are left out of: for the rubric,
  plans pass or fail on the other checks alone).
- **Settings**: `config.py` `JUDGE_*`; project `runtime_config["eval_judge"]["resilience"] = {breaker, mode, max_wait_seconds}`
  (`normalize_project_resilience`, clamped, max wait <= `max_wait_cap_seconds()`). `PUT /eval-judge` MERGES the block (a judge
  change keeps `resilience`; `resilience: {}` resets it); GET returns `resilience {stored, breaker, mode, max_wait_seconds,
  consecutive, max_wait_cap_seconds}` (the UI offers only waits up to the cap).
  UI: `JudgeModelCard` "When the judge is unreachable" (collapsed `<details>`).
- **Left for A2 / A3**: the `eval_rejudge` Job + endpoint + `UnjudgedRowsPanel` (+ `unjudged_only` predictions filter), the
  lift check re-judging an incomplete base without the GPU, Auto-RAG / passages verdicts saying "judge incomplete", the guard
  on Auto-RAG (thread path) and on the teacher filter (needs a loud failure first); A3 the pre-judge checkpoint (a cancel /
  crash during judging still loses the answers today). Tests: `test_judge_call_guard.py`, `test_eval_unjudged_scoring.py`,
  additions in `test_answer_judge.py`, `test_command_backend.py`, `test_rubric_service.py`, `test_dialogue_eval.py`,
  `test_post_training_lift_eval.py`, `test_code_example_scoring.py`, `test_eval_summary.py`; frontend `unjudgedNotice.test.ts`,
  `EvalSummaryCard`, `NotificationBell`, `JudgeModelCard`, `SftLiftPanel` tests. Known limits: a local CLI timeout counts as
  transient (a prompt that always exceeds the timeout can stop a healthy judge after 3 such rows).

## span_set scoring version (2026-11-12)

- `eval_task_handler_service.SPAN_SET_SCORING_VERSION` (2) is stamped as `metrics.span_set_scoring_version` on every
  span_set result. Bump it whenever span_set scoring or its eval prompt changes. `_baseline_scoring_matches` rejects
  a cached span_set base result without the current version, so the lift check re-runs the base model instead of
  pairing an old-scorer bar with a new tuned eval. The baseline *experiment* (`find_or_create_baseline_experiment`)
  is still reused; only its cached results are skipped. Version 2 = bare-list references/predictions read as
  `{"entities": [...]}` (v1 read a list reference as its first span: gold empty, so old numbers on bare-list
  splits were vacuous) + list references no longer name prompt fields. Older results stay as recorded and are not
  comparable to v2 ones. Tests: `tests/test_span_set_scoring_version.py`.
- Outside the lift Job the pairing is flagged, not blocked: `sft_lift_summary_service.span_set_version_mismatch`
  (raw eval_result metrics; a missing stamp = v1; both missing = same scorer) puts `scoring_version_differs
  {base, tuned, message}` on the lift summary, passed through by the eval summary and shown in `EvalSummaryCard`
  / `SftLiftPanel` (it names which side to re-score). Compare's `_scoring` carries `span_set_scoring_version` and
  `compare_provenance` adds a *note* (pair stays comparable) naming the side to evaluate again. The stamp is in
  `_HIDDEN_METRICS`, so it never shows as a metric/lift row. Tests: `tests/test_span_set_version_pairing.py`.

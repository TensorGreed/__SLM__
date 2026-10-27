# Evaluation: summary, lift, probes, eval gaps, benchmarks, playground

Detail notes split out of CLAUDE.md. Read when working on this area.

- **Evaluation** — `evaluation_service` runs eval packs; results land in
  `EvalResult` rows. Failure clusters, remediation plans, post-eval
  decision engine for reroute recommendations.
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
  `onResolved` auto-selects the server's run). All other eval surfaces sit
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
  are re-run once. Tests: `test_decoding_settings.py` (a tiny GPT-2 with a
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

- `services/code_check_service.py` is the registry (`register_code_check`, `list_code_checks`, `checks_for_language`, `run_code_checks`); built-ins in `services/code_checks/builtin.py`, sandboxed externals in `code_checks/external.py`. Externals run only when listed in `CODE_CHECK_ALLOWED` and the binary is on PATH; plugins via `CODE_CHECK_PLUGIN_MODULES`. API: `api/code_checks.py`. Docs: `slm-docs/docs/extensions/code-checks.md`.
- `services/sandbox_exec.py` (`run_sandboxed`) is the only place external code-check commands run; tests patch `sandbox_exec.run_sandboxed`. It also owns the minimal-env/kill helpers that `synth_backends/command.py` imports. Network isolation uses `unshare -rn` when it works (probe cached; `CODE_CHECK_NETWORK_ISOLATION` auto/off/require); it does NOT work on the dev host (user namespaces refused), so checks there report `network_isolated: false`. `auto` FALLS BACK to no isolation when `unshare -rn` fails; only `require` refuses to run (docs say so, and the Train > Remediation checks card shows "network not isolated" per ready external check). Real built-ins are trusted by identity (`code_check_service._CORE_BUILTINS`), never by a self-declared `kind == "builtin"`; plugins always need `CODE_CHECK_ALLOWED`. Java: a fragment is wrapped (member, then statement) with a sentinel member after the code and a brace-balance lexer; `java_is_toplevel` only accepts a column-0 declaration with no leading statements.
- Remediation scoring (fix lane): `code_checkable_count` / `code_parses_rate` depend on the ORIGINAL chunk only (malformed = checkable but not parsing), so base and fine-tuned share denominators. `issue_resolved_rate` / `new_findings_rate` are over rows whose patch applied (informational, hidden from the lift table); the lift metric is `issue_resolved_overall_rate` = resolved / (scored rows - detector-blind - detector-error rows). A finding the detector did not report on the ORIGINAL chunk is `detector_blind` (`row_detector_note`, `detector_blind_count`): `row_issue_resolved` None, excluded from both rates, ignored by the pass rule. Metrics carry `remediation_fingerprint` (detector identity incl. run id, effective check ids, `CODE_CHECK_ALLOWED`, retrieval settings, extra-field schema hash); a cached base result is reused only when it matches (`post_training_eval_service._baseline_scoring_matches`). Detector cache keys include the findings-prompt builder + decoding fingerprint and, for command detectors, the binary realpath/mtime/size. Retrieval for rows without passages runs in worker threads (`combine_datasets`, `_load_heldout_pairs`); own-chunk exclusion matches by normalised path suffix, source document id, or text containment (>= 80% of the row's lines).
- `services/unified_diff.py`: `apply_patch_to_chunk` (absolute line numbers, exact match ± `max_offset`, reasons `empty_patch/not_a_diff/other_file/multiple_files/context_mismatch/overlapping_hunks/out_of_range/no_op`), `make_unified_diff` round-trips with it. `offsets[i]` = applied position minus stated position.

## `remediation` scoring + the code-remediation recipe (G5, lane B2)

- Recipe `code-remediation` (`structured_extraction`, scoring mode `remediation`). Prompt + payload helpers live in
  `services/code_remediation_prompt.py` (`build_code_remediation_prompt(_from_row)`, `build_code_example_prompt(_from_row)`,
  `canonical_remediation_json`, `retrieve_passages_for_row`, `effective_remediation_config`); code is rendered by
  `code_findings_prompt.render_numbered_code`. One builder serves the adapter (`data_adapter_service._map_remediation`),
  `StructuredExtractionHandler._build_remediation_prompt`, the playbooks and the judge; chat template for train AND eval
  (`wraps_own_prompt` false for `finding_set` and `remediation`).
- Manifest stamp (`dataset_service._remediation_schema_stamp`): `required` = the three base fields + declared extras,
  `properties` = declared EXTRAS only (base names stripped; `description` is kept — `OutputProperty.description`),
  `remediation {checks, retrieve_passages, top_k, retrieval}`, `label_parents`, `label_titles`. `combine_datasets` puts
  `adapter_config["remediation"]` (extra fields, project id, retrieval settings) on the structured-extraction adapter.
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
- API: `api/remediation_config.py` (`/projects/{id}/remediation-config`; `api/remediation.py` is the older failure-plan API).
  Probe packs: `_RECIPES_WITHOUT_PROBE_PACK`. Not wired: `coach_service._PER_CLASS_RECIPE_IDS` (no `per_class` is emitted).
- Tests: `tests/test_code_remediation_recipe.py`. Docs: `slm-docs/docs/workflows/code-remediation.md`.

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

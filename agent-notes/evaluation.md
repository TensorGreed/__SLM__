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
  falling back to the metric id). **Phase 16 (cross-run memory)** —
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

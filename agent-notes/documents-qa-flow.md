# Documents → Q&A flow + training with passages (RAFT)

Detail notes split out of CLAUDE.md. Read when working on this area.

- **Documents → Q&A flow** (`services/documents_qa_flow_service.py`,
  `api/flows.py`, Job kind `documents_qa_flow`): cleaned document passages →
  per-passage prompt to a `synth_backends` backend → `pairs_per_passage`
  training pairs (synthetic, **accepted**, `synth_source="documents_qa_flow"`)
  + one different eval question (GOLD_DEV, `source="documents_qa_flow"`) →
  `apply_recipe_to_project("qa-sft")` (keeps the user's base model) →
  `split_dataset(DEFAULT_TRAINING_SOURCE_TYPES)` → `create_experiment` +
  `api.training.start` (watcher → lift check). `reuse_existing=True` skips
  generation (same pairs/answer key, re-split + train the project's current
  base model — card button "Train again on the current base model"). **Cloud generation model**:
  `backend="cloud:<provider>:<model>"` on this flow and on seed-documents resolves through
  `services/cloud_backend_pins.resolve_cloud_backend` (the project's saved key; `CloudPinError` →
  400 `BAD_BACKEND_PIN` / 402 `CLOUD_KEY_MISSING`; the span routes' `_resolve_cloud_backend_override`
  wraps the same helper) → `backend_override`; the preview takes `?backend=` (a cloud pin with a
  saved key is eligible without a local model). UI: `components/shared/GenerationModelPicker`
  on `DocumentsQaFlowCard` + `SeedDocumentsCard`. `train=False` stops after
  the split. Stops before training under `MIN_TRAINING_PAIRS` (40). Coach
  training stage: `training:documents-to-qa` with action kind
  `start_flow` (frontend `CoachSuggestion` posts the flow; it leads even
  when no task type is set); card `DocumentsQaFlowCard` above the Training
  tab's panel; bell deep-links `#documents-qa-flow`. Tests use
  `backend_override` with a fake backend.
  **Sentence-boundary chunking**: `cleaning_service.chunk_text` packs
  `split_sentences` units (sentence ends before a capital / bracket, or line
  breaks; a digit is not a sentence start so "s. 12" stays whole) up to
  `chunk_size`; overlap = trailing whole sentences ≤ `overlap` chars; only an
  oversized single sentence is hard-cut. Same function builds the sweep's
  chunk-size variants. Tests: `test_phase9_cleaning`.
  **Quotable answer key**: `build_passage_prompt` holds the eval question to a
  stricter standard (self-contained, answer = the passage's own complete
  statement, never on a cut-off chunk fragment); `eval_pair_rejection(pair,
  passage_text)` → `"deictic"` (`_DEICTIC_RE`: "this subsection", "the passage
  mentions", "mentioned above"…) / `"unsupported"` (`answer_support` < 
  `EVAL_ANSWER_SUPPORT_MIN` 0.7 of content words in the passage) / None;
  dropped keys → `FlowSummary.eval_rejected` {reason: n} + a warning; the
  passage keeps its training pairs. **Every question stands on its own**
  (prompt rule for pairs + eval): `repair_deictic_questions` runs after parsing,
  before any row is kept — `strip_passage_reference` cuts a pure attribution
  (`_ATTRIBUTION_TAIL_RE` / `_HEAD_RE`: "…, according to the passage?"), the rest
  of `is_deictic` questions go back to the generation model, one
  `build_rewrite_prompt` call per passage (via `_complete_calls`, the shared
  batch/concurrency/checkpoint runner `_generate_replies` now uses), parsed by
  `parse_rewrite_output` (wrong count → all None). Still deictic → training pair
  dropped (`pairs_rejected`), eval → the gate. Summary `questions_repaired`
  {stripped, rewritten}; card "N questions rewritten…". The template backend's
  questions are worded "What is said about …" (not "the passage") so the
  CI documents gate keeps its pairs. gpt-oss:120b on Pub 17: 280/3190 train +
  292/1064 eval questions pointed at the passage. Test fakes must return quotable eval
  answers (the flow suite's FakeBackend quotes the passage).
  **Sizing** (`size_for_decision(available_passages, pairs_per_passage)`):
  `target_test_rows()` = ceil((2·`DIFF_SD` 0.5 / `DECISION_GAP` 0.15)²) = 45
  paired rows; passages = ceil(45 / `TEST_RATIO` 0.1 / pairs) = 150 (taken by `spread_passages`:
  evenly spaced over the corpus in document order, never the first N), capped
  by what's available; when capped the test share rises to ≤ `MAX_TEST_RATIO`
  0.3 (`reaches_target` False when even that is short). `max_passages=None`
  (API default) → sized; the preview's `plan.sized_for` explains it; the
  flow passes `test_ratio` to `split_dataset` and records `split.sized_for`;
  a `reuse_existing` run keeps the active manifest's test ratio
  (`_active_split_ratios`) so re-trains score the same rows.
  **Pre-training gate** (`passages_check=True` default,
  `train_if_retrieval_ready`): after the split the flow runs the judged
  base + documents comparison on the test split AS A RETRIEVAL SWEEP
  (`_run_passages_gate` → harness `run_project_comparison(base_only,
  corpus="documents", split="test", sweep_retrieval=True)` on a thread;
  the winner is written to `runtime_config.auto_rag_retrieval` via
  `apply_retrieval_choice`, so the gate judges the best retrieval the
  project can serve and a RAG sibling inherits it; `passages_check_fn`
  injects it — tests MUST inject or pass `passages_check=False`, the
  default loads a model; the gate dict carries `retrieval` +
  `retrieval_sweep`). `read_passages_gate` → `passages_gate(...)` (pure,
  three tiers, never F1): `retrieval_ready` = evidence `better` AND score ≥
  `PASSAGES_GATE_READY_MIN_SCORE` 0.65 AND correct share ≥ 0.5 AND wrong
  share ≤ 0.25 (a system wrong a third of the time is not "already
  answering"); `retrieval_promising` = evidence better AND score ≥ 0.4;
  else `retrieval_weak`; `not_judged` (no judge) / `not_run` / `error`.
  Calibration (gemma4:12b judge): legal plain top-3 8/5/6 (0.55) →
  promising; legal top-3+reranker 12/4/3 (0.74) on project 21's split →
  ready — but on project 22's split the sweep kept top-3 (0.55 vs 0.53),
  so sweep wins on 19 rows are noise-level; support-faq best retrieval
  0/9/12 (0.21) → weak; a clean re-run 2026-10-13 (questions by
  gemma4:latest, new split) gated 5/3/11 (0.34) → weak → trained → judged
  1/4/14 (0.16) → Coach reroute (0.34 vs 0.16, same rows); the SIZED run
  (150 passages, 46 test rows) switched to top-5+reranker beyond noise
  (+0.15, CI +0.02..+0.29), gated 20/9/17 (0.53) → promising → fine-tune
  4/3/39 (0.12) → reroute. Fine-tunes scored ≤ 4 correct every time, so
  "ready" stops the flow
  (`stopped_reason` "Retrieval already answers…"; the card's "Train anyway" re-posts
  `reuse_existing` + `train_if_retrieval_ready` + `passages_check: false` — same split, no second
  sweep), "promising" trains with
  a warning and lets the judged lift compare. Coach training stage
  `training:retrieval-ready` (warning, reroute action) /
  `training:retrieval-promising` (info, "train, then compare"; same
  action) via `_retrieval_ready_nudge` (quiet once a run exists / rag_first);
  `build_eval_summary` for an untrained non-RAG project returns the
  passages summary with `pretraining_gate` + a "nothing trained yet" message.
- **Train with retrieved passages (RAFT)** (`services/passage_prompts_service.py`,
  `TrainingConfig.passage_prompts` + `passage_prompts_withheld_fraction` 0.2; flow
  `train_with_passages`, card checkbox "Train the model to answer from your passages"):
  `training_service._maybe_apply_passage_prompts` (after curriculum) rewrites the run's
  train/val into `<output_dir>/passage_prompts/` — the project's served retrieval over the
  DOCUMENT passages (`auto_rag_service.retrieve_document_passages`, k+3 hits) becomes the row's
  `system_prompt` = `document_preamble(hits)` (the one string serving + harness use); answer
  passage (`answer_support` ≥ 0.7) cited `[n]`, put back when missed (`inserted`); withheld rows
  → `NOT_IN_PASSAGES_ANSWER`. `train.py` carries `system_prompt` through `to_text_record` (only
  when the column exists) and `_rewrap_prompt_completion_row` renders it as the system message.
  Lift check (`passage_reader_eval_service.run_passage_reader_evaluation`, routed from
  `run_post_training_lift_eval` by `is_passage_trained`): base + passages (cached
  comparison_base_documents.json reused only when `base_comparison_usable`: judged, test, same
  base model + retrieval + row count) vs harness `run_project_comparison(experiment_id=…,
  corpus="documents", split="test")` → `auto_rag/passage_reader/<run>.json` (paired judge
  evidence, both arms' retrieval splits); summary `kind="passage_reader"`; verdict `arms`
  `base_with_passages` vs `finetune_with_passages`. Seed groups refuse. Tests `test_passage_prompts.py`.
  **Duplicate project** (`project_duplicate_service`, `POST /projects/{id}/duplicate`
  {name?, base_model_name?}, Settings → "Duplicate this project"): raw/cleaned/gold/prepared/
  synthetic + datasets + documents, never runs / evals / auto_rag. Use it to try another base
  model — `passages_vs_finetune` returns `different_base_model` when the bar's base ≠ the run's
  (2026-10-19: switching 33 to 1.5B overwrote its 3B bar), and the comparison GET flags
  `base_model_mismatch`.
  **Teacher targets** (`services/teacher_passage_answers_service.py`, flow
  `answers_from_teacher` + `teacher_backend` (cloud pin or local name, e.g. `ollama:qwen3:32b`)
  + `teacher_filter` + `use_batch`, card "Use a teacher model's answers to the same passages"):
  after the gate, `write_teacher_answers` retrieves every TRAIN/VAL question with the served
  retrieval (`passage_prompts_service.served_retrieve_fn`), sends `document_preamble(hits)` +
  `TEACHER_STYLE_NOTE` as system + the question (one Message Batch with `use_batch`, else 4
  concurrent calls), caches `{answer, abstained (is_abstention), system_prompt, sources, teacher,
  filter}` in `teacher_answers/answers.jsonl` keyed by `answer_key(question, retrieval)`; the
  filter = local answer judge vs the reference on train rows, `wrong` → `dropped` (`teacher_filter_keep`: `not_wrong` default, `correct` drops partial answers too; verdicts cached, the keep rule is re-applied to every cached verdict on each run — `filter_drops`, counts `filter_by_verdict`).
  **Critique-and-revise** (`teacher_revise_rounds` 0–2, default 0 = byte-identical to before;
  `teacher_critic_backend` pin / name, None = the filter verdict's own reason, no extra call;
  needs `teacher_filter` → 400 `REVISE_NEEDS_FILTER`): `critique_and_revise(records, llm=,
  judge=, keep=, rounds=, critic=)` — the reusable helper — critiques each dropped answer
  (`build_critique_prompt` + `parse_critique`, bad JSON → the judge's reason), the teacher
  rewrites it from the SAME preamble (`revise_request`), the same filter judge re-grades (final
  authority); a kept revision becomes `answer` / `filter` (original in `original_answer`,
  rounds in `revisions`, `revised: true`), each round appended to answers.jsonl (resume: same
  critic + judge → no calls); abstentions never revised; revisions always direct calls (no
  batch); counts `counts.<split>.revise` {attempted, fixed, still_dropped, abstained, calls} + `revise`
  {rounds, critic, judge, critic_is_judge}. The filter tally runs AFTER revise over final verdicts and
  counts a judged record even when its revision became an abstention (it used to skip it as
  `abstained`: p38 showed attempted 628 vs filter_dropped 317 with 69 abstaining revisions in no
  count) → `filter_dropped == still_dropped`; `counts.abstained` is first-pass only; card checkbox "Revise answers the checker rejects"
  + `GenerationModelPicker label= / localOptionLabel=` as the critic picker. Run config
  `passage_prompts_targets: "teacher"`, withheld 0 → `apply_passage_prompts(target_mode="teacher")`
  → `teacher_rows` (cached preamble verbatim as `system_prompt`, teacher answer as target;
  `no_teacher_answer` / `filtered` counted). Generated pairs carry `generated_by`. Flow result
  `teacher` + its own `usage` list (batch calls bypass the Job meter). `train.py
  _fit_passage_seq_length`: passage-prompt runs raise `max_seq_length` to the longest row (≤
  `PASSAGE_MAX_SEQ_CAP` 4096; report `max_seq_length_fit`). **Sized + pre-registered test**:
  flow `decision_gap` (0.15 → 45, 0.08 → 157; preview `?decision_gap=`), `keep_test` →
  `grow_test_set` (answer-key rows spread over new passages) → `split_dataset(keep_test,
  test_additions=, val_rows=)` (`_keep_test_rows`: kept rows verbatim first, additions
  `_source_dataset="answer_key"`, val sized to the target, `question_leakage_filter` drops
  train/val questions exact / Jaccard ≥ 0.9 to any test question) → `split.test_sha256`.
  **Val sweep**: `sweep_split` (default `val`): `_run_passages_gate` sweeps on val, applies the
  winner, scores it once on test (no sweep) — the gate + reader-eval bar read the test file;
  `gate.sweep` = the val side. `MAX_PASSAGES_CAP` 2000. With `keep_test` the VAL set is eval-only answer-key rows too
  (`val_from` / `keep_val` on `split_dataset`, `fixed_val` in `_keep_test_rows`; `split.val_sha256`).
  Local teacher: `synth_backends/ollama.think_setting` (gpt-oss → `"low"`), `normalize_teacher_answer`
  (【n】 → [n], no `**`), `unload_loaded_models` before training when `_used_local_ollama`. Resumable: `_generate_replies(checkpoint=)` appends each reply to `flows/generation_checkpoint.jsonl` (key = backend label + call; fsync'd; unlinked after the pairs are saved), the teacher appends each answer (`_append_answer`), the filter flushes the judge cache every 25 rows (`AnswerJudgeCache.flush` is atomic) — re-posting a killed flow asks only for what's missing. Flow `concurrency` (1–8) bounds direct
  generation (`_generate_replies`) + teacher calls in flight (for `OLLAMA_NUM_PARALLEL`). Tests in `test_documents_qa_flow.py`
  (`FlowGrowthSweepTeacherTests`), `test_training_memory_defaults.py`.

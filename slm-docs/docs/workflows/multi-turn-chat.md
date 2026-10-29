---
title: Multi-turn chat
sidebar_label: Multi-turn chat
---

# Multi-turn chat

A chat dataset (`messages`, or `conversations` with `from` / `value`) can hold a system prompt and several user / assistant exchanges per row. This page covers how such rows are trained, how they are evaluated, how to generate them from your documents, and how to turn an eval's wrong answers into preference pairs. Nothing here depends on a domain or a recipe.

## Training: loss on assistant turns only

A row counts as multi-turn when it has a system message, more than one assistant turn, or more than two messages. The trainer renders it with the tokenizer's chat template and trains on **assistant tokens only**: system, user and tool tokens are masked, and the end-of-turn tokens the template emits after an assistant message are trained. A tokenizer without a chat template renders `role: content` lines instead, and the final assistant turn then gets a trained end-of-sequence token (as a single-turn row does).

- A `[system, user, assistant]` row is multi-turn and is masked this way. (It used to train on every token.) A row that is exactly one user turn and one assistant turn, with no system message, trains as before.
- A row longer than `max_seq_length` loses its **oldest exchanges** (a user turn and everything up to the next user turn) until it fits. The system message and the final exchange are kept; if even that does not fit, the row is dropped. A turn is never cut in the middle.
- Some chat templates re-render earlier assistant turns differently once a later turn exists (for example Qwen3's think-block templates, or gpt-oss channel templates), so per-turn spans cannot be found. Those rows are masked on the final assistant turn only and counted in `mask_fallback_rows`. Templates that merely append an end-of-sequence token after the last message (Phi-3 style) keep every assistant turn trained.
- The `trl_sft` training backend is not affected by any of this: it trains multi-turn rows on every token.
- The run's report carries a `multi_turn` block: `rows`, `assistant_token_share` (trained tokens over all non-pad tokens of those rows), `assistant_turns`, `truncated_rows`, `dropped_turns`, `dropped_rows`, `mask_fallback_rows` (templates whose per-turn renders do not line up, masked on the final assistant turn only). A simulated run reports the same block with `approximate: true`, counted from characters (about 4 per token) rather than tokens, and only for SFT runs.

## Evaluation

- **Last turn.** A held-out chat row is scored on its final assistant turn. The model sees everything before it, system prompt included, rendered with the chat template exactly like training. Single-turn rows are unchanged.
- **Judge every assistant turn.** Pass `dialogue_judge: true` in the body of `POST /api/projects/{id}/evaluation/run-heldout`. For each row with two or more assistant turns, every turn is generated from the **gold** history up to it and graded by the answer judge against that turn's reference. This costs one generation and one judge call per extra turn (capped at 500 extra generations per eval). The result is `metrics.dialogue_judge`:

  ```json
  {"rows": 12, "mean": 0.71, "worst_turn_mean": 0.42,
   "by_turn_index": [{"turn": 1, "mean": 0.9, "n": 12}, {"turn": 2, "mean": 0.7, "n": 12}]}
  ```

  `mean` averages each row's turns, `worst_turn_mean` averages each row's weakest turn, and `by_turn_index` shows whether the model degrades later in a conversation (turns are 1-based assistant turns). Each stored row carries its own `dialogue_judge` (`turns`, `mean`, `worst_turn`). Without a reachable judge the metric is `{"skipped": "no_judge_available"}` and nothing is generated.
- **Chat judge.** Chat rows are graded like any long-answer generative task (`chat_sft` is a judged profile; a final turn of eight or more words is judged): the judge sees the whole dialogue as the question and the final turn's reference as the key. The per-turn judge above is the alternative that grades every turn.
- **Replay cap.** At most 500 extra generations are replayed, whole rows at a time. Rows past the cap are left out of `metrics.dialogue_judge` (not scored on their last turn alone) and counted in `rows_not_replayed` with `replay_cap_hit: true`. Replayed generations do not count in the eval's latency or throughput figures.
- **Stored rows.** `details.row_predictions_file` (`{path, count, truncated}`) points at `experiments/<run>/evals/result_<id>_rows.jsonl` under the project folder. It holds only the rows a judge scored: `row_index`, `history` for chat rows (the last 40 messages; `history_truncated` when cut) or `prompt`, reference, prediction, the verdict and the row's `dialogue_judge`. At most 1000 rows are stored; the rest are `truncated`. The results list never carries row arrays.

## Generate dialogues from documents

The `documents_multi_turn` playbook (recipe `generic-sft`, the one the chat task shape uses) writes passage-grounded dialogues from the project's document passages. Run it from the synthetic playbook center or `POST /api/projects/{id}/synthetic/run-playbook` with `"mode": "documents_multi_turn"` and:

| Param | Default | Meaning |
|---|---|---|
| `turns_min` / `turns_max` | 2 / 5 | exchanges (user plus assistant) per dialogue, 2 to 8 |
| `dialogues_per_passage_group` | 1 | dialogues written about each passage group, 1 to 5 |
| `passages_per_dialogue` | 2 | neighbouring passages per group, 1 to 3 |
| `source_ids` | all sources | only draw passages from these data sources (1 to 100 ids of this project); the result's `seed_pool` shows what was used per source |
| `seed_path_include` / `seed_path_exclude` | none | glob lists on the passage's path (see Seed filters in the code-findings workflow); applied before the cap |
| `seed_relevance_terms` | none | keep only passages whose path or text mentions a term (`seed_relevance_scope`: `chunk` or `file`; `seed_relevance_match`: `path_or_text` (default) or `text` = ignore the path; `relevance_exempt_source_ids` skips it for some sources) |

The teacher is asked for later user turns that depend on earlier ones, and for `[n]` citations in every assistant turn. A dialogue is kept only when roles alternate, the turn count is in range, every assistant turn cites a passage of its group, at least 60% of its content words occur in the cited passages, no user turn repeats an earlier one, and no user turn matches a question of the answer key's test split (exactly or near-identically). An answer with no content words to check counts as supported. Rejects are grouped by reason (`bad_roles`, `turn_count`, `no_citation`, `bad_citation`, `unsupported`, `repeated_user_turn`, `test_question_leak`, ...) in the review queue and can be selected and dropped in bulk. Each row holds `messages` plus `source_passages` ids. The first message is a **system message** with the dialogue's passages, in the same cite-or-say-you-don't-know wording passage-trained rows and serving use, numbered so the `[n]` citations resolve; the trainer treats it as the system prompt (never trained on). A run that accepts nothing fails with the reason breakdown. Documents whose source is evaluation-only or retrieval-only in the licence ledger are never used.

## Preference pairs from an eval

`POST /api/projects/{id}/alignment/preference-pairs/from-eval` with `{"eval_result_id": 42, "include": ["wrong", "partial"], "dataset_name": "optional", "allow_test_split": false}` turns the rows a judge marked wrong (or partial) into pairs: `chosen` is the answer-key reference, `rejected` is the model's own answer. A row's verdict is the answer judge's, or, for a chat row it did not grade, the per-turn judge's verdict on the row's final turn. The pairs are read from the eval's stored rows, so only judged rows can become pairs.

The file is written to `prepared/alignment/eval_<eval_result_id>_pairs.jsonl` with `{prompt, chosen, rejected, source: "eval", source_split, eval_result_id, experiment_id, row_index, verdict}`. For a chat row the `prompt` is the same flat render eval uses (`role: content` lines plus a final `assistant: `) and the pair also keeps its `messages`.

**The test split is refused.** Pairs built from the held-out test split would teach the model the very rows it is scored on, so a later eval on that split would measure memorisation. An eval on the `test` (or `gold_test`) split fails with `400` `test_split_refused`; `allow_test_split: true` builds them anyway and the reply carries a `warning`. Every pair carries `source_split` either way. In the UI the "Allow test split (measures memorisation)" checkbox appears only for those evals.

The reply is `{dataset_name, dataset_path, source_split, warning, created, skipped, examples}` (`dataset_id` is `null`: pairs are a file, not a dataset row). **To train DPO or ORPO on them, set the file as the alignment dataset path in Train.** `skipped` counts rows by reason: `not_judged`, `correct`, `not_included`, `prediction_equals_reference`, `empty_prediction`, `empty_reference`, `no_prediction_stored`, `not_stored` (judged rows past the stored-row cap, or whose history was cut). If no pair can be built the call fails with `400` `no_pairs_built` and the same breakdown in `metadata.skipped`. An eval that stored no rows (run before this feature, or nothing was judged) reports `no_prediction_stored`.

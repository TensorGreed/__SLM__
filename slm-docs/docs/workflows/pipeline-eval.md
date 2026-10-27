---
title: Pipeline evaluation
sidebar_label: Pipeline evaluation
---

# Pipeline evaluation

Many real tasks take more than one model: extract a list, then summarise each entry; classify a request, then answer it; detect a problem, then fix it. A **pipeline** chains projects you already trained into stages and scores the chain on one held-out set, so you can see both how well the whole thing works and **which stage breaks it**.

Open **Pipelines** from the top bar. Each member project's Results page lists the pipelines it belongs to.

"Pipeline" here means a chain of models evaluated together. It is not the project lifecycle (data → train → evaluate → ship) that other pages sometimes call the training pipeline.

## Stages

A pipeline has 1 to 8 stages. Each stage names:

| Field | Meaning |
|---|---|
| `name` | `^[a-z][a-z0-9_]{0,31}$`, unique in the pipeline. Gold outputs are keyed by it. |
| `project_id` + `model` | Which model answers: `""` = the project's served model (as the gateway resolves it), `":base"`, or `":run-<id>"` (a completed run of that project with saved weights). |
| `prompt` | `{"mode": "recipe"}` uses the project's training prompt format (only recipes with a registered prompt builder), or `{"mode": "template", "template": "...", "system": "..."}` with `{field}` placeholders. Every `{identifier}` in the template or system text must be one of the stage's inputs (saving refuses unknown ones); other braces stay literal. |
| `inputs` | `{field: {from, pointer}}`: `from` is `row` (the held-out row's `input`), `previous` (the previous stage's parsed output) or `item` (the current fan-out item). `pointer` is a JSON pointer, `""` = the whole value. In recipe mode the inputs are the recipe's arguments. |
| `fan_out` | `{pointer: "/items/*", item_key: ["/label"]}`: run this stage once per item the pointer finds in the previous stage's output. At most one stage per pipeline fans out; every later stage runs per item too. The item key must be **unique** within one output: when two items share a key the first one is used and the rest are counted as `duplicate_item_key`. |
| `retrieval` | `true` / `false` / `null` (the model's default, as in the serving gateway). |
| `max_tokens` | 1 to 4096, default 512 (leave it empty for the default; 0 is refused, not replaced). |
| `output` | `json` (parsed; a code fence is stripped, also one closed on the last line, and a reply with prose around one JSON object or array still parses) or `text`. Use `text` for bare-label or free-text outputs. |

Stage 1 can only read the row. A `previous` pointer or a fan-out needs the previous stage to output JSON. Saving checks every field and returns all problems at once (`stages[1].inputs.name.from: 'item' needs fan_out on this stage`).

## Held-out set

Upload JSONL (or send `{rows: [...]}`), one row per line:

```json
{"id": "r1",
 "input": {"text": "Bring apples and beans."},
 "gold": {"extract": {"items": [{"label": "a", "name": "apple"}, {"label": "b", "name": "bean"}]},
          "rewrite": [{"label": "a", "output": "APPLES"}, {"label": "b", "output": "BEANS"}]}}
```

- `gold` may cover any subset of stages; a stage without gold for a row is run but not scored.
- From the fan-out stage on, gold is a **list of objects** carrying the item key fields. The reference is the object's `output` field when it has one (needed for text outputs), otherwise the whole object.
- Bad rows are rejected one by one and reported by reason with line numbers (`invalid_json`, `duplicate_id`, `missing_id`, `input_not_object`, `unknown_stage_in_gold`, `fan_out_gold_not_list`, `fan_out_gold_item_missing_key`, ...); valid rows are kept. Uploading replaces the previous set (at most 5000 rows and 50 MB; a larger upload gets HTTP 413).
- Each run re-checks the stored rows against the pipeline as it is then (a stage renamed or re-keyed since the upload): rows that no longer fit are left out and listed under `gold_mismatch` by reason.

## How a run works

**Run** starts a background job (`pipeline_eval`, in the notification bell) over up to `max_rows` rows (default 50, at most 500); cancelling the job stops it before the next model call. (The API also runs up to 5 rows inside the request, without `async_job=true`; anything larger must be a job.) For every row:

- **Stage 1** reads the row, runs, and is scored against its gold.
- **Stage k≥2, end to end**: built from stage k-1's **predicted** output.
- **Stage k≥2, isolated**: built from stage k-1's **gold** output, so you see the stage on its own. It runs only for rows with gold for this stage **and** the one before; other rows are skipped (`skipped_rows`) without calling the model.

Fan-out items are matched to gold items by `item_key`: every key pointer's value must be present and equal after trimming and lowercasing (one-to-one, in order). An item missing a key value matches nothing (`missing_item_key`). An output whose item matches no gold item is **spurious** (counted, not scored); a gold item that no predicted item matched is **missed upstream**, or **lost to error** when the upstream output it would have come from failed.

### Scoring

Each stage is scored by **its own project's evaluator, the way that project's held-out eval scores**. Every output with gold becomes a row in the project's own field names: the stage inputs under their names, the stage prompt as `source_text`, and the gold under the reference fields the project's prepared rows use (`reference` when it has none). That row goes through the project's prompt builder, and the extras the scorer needs (the code and finding behind a patch, a plan's scenario and passages, the gold tool calls, …) are carried over with the same helper held-out eval uses. The model still answers the stage's own prompt; the built row is used for scoring only. So an input named like the project's row fields (`code`, `finding`, `scenario`, …) is what a recipe scorer reads.

The judges the project's held-out eval would run also run here, per stage view: the long-answer judge for long references, the structured-plan rubric, the remediation judge. Correctness follows the held-out rule: where a judge's verdict decides pass / fail it decides it here too; where the project's own checks decide (a patch, a plan's schema, a tool call), they do.

**Nothing counts as right or wrong for want of scoring.** An output with gold is **unscored**, with a reason, when:

| Reason | When |
|---|---|
| `row_context_unavailable` | The project's evaluator cannot build a row from the stage inputs (for example a fix without the code or finding it applies to, a plan without a scenario). |
| `needs_judge` | A judge would decide this output, but none was reachable or it failed. The stage says `judge_not_run` with the reason. A plan that fails its schema is still wrong; only outputs the judge would decide are held back. |
| `scoring_error` | The evaluator raised. The stage shows `scoring_error`. |

Unscored outputs are left out of the stage's accuracy, and a final item whose path runs through one is left out of the end-to-end share and the attribution (`unscored_items`, by reason). An output that failed on its own (missing input, model error, invalid JSON, a prompt the recipe could not build) counts as wrong when it has gold.

Model calls go through the serving gateway's internal path (same model resolution, retrieval and guided-JSON check as `/v1`) without using an API key's rate limit or writing gateway request logs. Identical prompts to the same stage are generated once per run (temperature 0). A run stops after 5000 model calls, says `truncated` and names the row it stopped at. A run in which no stage produced output, or nothing could be scored, fails with the reason instead of reporting an empty result.

## Results

`runs/<run_id>.json` holds, per stage, both views (`scored`, `correct`, `accuracy`, `unscored` + `unscored_by_reason`, `spurious`, `missed_upstream`, `lost_to_error`, `missing_item_key`, `duplicate_item_key`, `errors`, and `skipped_rows` for the isolated view) with the project, model id and run id that produced them, the evaluator, `judge` (`{kind, label, ran, units, reason}`), `judge_not_run` and `scoring_error`, plus:

- **End to end**: `gold_final_items` (the last stage's gold items, or one per row), `unscored_items`, and the share of the scored ones that is **fully correct**: every stage on the item's path that has gold got it right.
- **Attribution**: for each final item that is not fully correct, the **first wrong stage** on its path. The stage that produces the fan-out items is judged **per item**: it is right for an item only if it produced an item with that key that equals the gold item at the fan-out pointer (strings compared like exact match: case, punctuation and articles ignored). Without a gold item to compare, the key match alone counts; the stage reports `item_compare` (`deep_equal`, `key_only` or `mixed`) with counts. Its whole-output accuracy on the stage card is unchanged. An item the upstream stage never produced counts against that upstream stage (`missed_upstream_at`).
- **Rows**: per-row outputs and verdicts for the first 200 rows.

A run file exists from the start (`status: running`); a failed or cancelled run keeps its `error`. The runs list reads small summary files written beside each run, and shows a run that still says `running` after its job ended or vanished as `stale`. The pipelines list flags a member project that was deleted.

## API

| Method | Path | |
|---|---|---|
| `GET` / `POST` | `/api/pipelines` | list / create |
| `GET` / `PUT` / `DELETE` | `/api/pipelines/{id}` | one pipeline |
| `PUT` / `GET` | `/api/pipelines/{id}/heldout` | upload (multipart `file` or `{rows}`, 50 MB max) / preview (`?limit=`) |
| `POST` | `/api/pipelines/{id}/runs?async_job=true` | `{max_rows}` → 202 job (without `async_job`: at most 5 rows, in the request) |
| `GET` | `/api/pipelines/{id}/runs`, `/runs/{run_id}` | runs |
| `GET` | `/api/projects/{id}/pipelines` | pipelines a project belongs to |

With authentication on, seeing a pipeline needs viewer access to **every** member project (others are hidden), and editing, uploading or running needs editor access to every member project. Pipelines are stored as files under `DATA_DIR/pipelines/`.

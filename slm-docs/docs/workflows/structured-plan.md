---
title: Structured plan
sidebar_label: Structured plan
---

# Structured plan

The **Structured plan** recipe (`structured-plan`) trains a small instruct model to turn a scenario into one JSON document that validates against a JSON Schema bundle you upload. The schema can span several files joined by `$ref`. Typical use: a deployment plan, a rollout checklist or any artifact with a strict shape plus a few fields that must be right.

## Rows

```json
{"row_kind": "structured_plan",
 "scenario": "Roll out the billing service to staging, then production.",
 "passages": [{"id": "P1", "text": "Staging is always updated before production.", "source": "rollout-guide"}],
 "plan_json": {"name": "billing rollout", "environment": "staging",
               "steps": [{"action": "update", "target": "billing", "environment": "staging"}]},
 "acceptable": {"/environment": ["staging", "dev"]},
 "grid_cell": {"region": "north", "tier": "gold"}}
```

Only `scenario` and `plan_json` are required. `passages` (ids `P1..Pk`), `acceptable` (several accepted values per JSON pointer) and `grid_cell` (the matrix cell the scenario came from) are optional. A row without a plan object is rejected with `structured_plan_missing`. The recipe suggests `Qwen/Qwen2.5-7B-Instruct` (3B and 1.5B as smaller options). Rows train and evaluate through the model's chat template with byte-identical prompts: the scenario, the optional passages, the schema as compact JSON (cut with a note past 12000 characters) and the instruction to return only a JSON object.

When a row has no passages and `retrieve_passages` is on (default: on exactly when the project has a document index), the scenario is used as the query against the project's document index (`top_k`, default 3). The retrieved passages are stored on the prepared row, so evaluation renders the identical prompt.

## Settings (Train > Structured plan)

`GET` / `PUT /api/projects/{id}/structured-plan-config` (see the [API reference](../reference/api/structured-plan.md)):

- **Schema**: up to 20 JSON files (1 MB in total) and the root file name. Every file is checked under its `$schema` dialect: JSON Schema 2020-12 (the default when `$schema` is absent) or 2019-09, and all files in a bundle must use the same one (older drafts are refused). Every `$ref` must resolve inside the bundle (by file name or `$id`) and remote references are refused; nothing is fetched. Plain-name anchors (`step.schema.json#name`) across files are not supported: reference a JSON-pointer fragment instead (`step.schema.json#/$defs/name`). A `pattern` runs under a short timeout (0.5 s); a pattern that backtracks past it reports "pattern check timed out" as a validation error instead of hanging. All problems are reported together (422 `INVALID_STRUCTURED_PLAN_CONFIG`, `metadata.errors`).
- **Rubric** (optional): YAML or JSON with weighted criteria, each with five anchors (1 to 5) and an optional `critical` flag. A row passes the rubric when no critical criterion scores 2 or less and the weighted mean is at least 3.0 (a top-level `pass_threshold` overrides it).
- **Pointers**: JSON pointers (at most 50, `*` fans out over arrays) that are field-matched against the reference.
- **Judge** and **passage retrieval** toggles.
- A valid schema also stores the dereferenced schema as the project's structured-output hint (`GET /api/projects/{id}/guided-json`, source `structured_plan`), carried into the export manifest. Clearing the schema, or switching to another task type, drops it. A code-findings hint is never overwritten.
- `POST /api/projects/{id}/structured-plan/validate` checks any pasted JSON against the saved schema.

## Scoring

Per held-out row: the output is parsed, validated against the schema, and each pointer is compared with the reference. A pointer matches when the multiset of values it resolves to equals the reference's (order does not matter, types come from the reference value, `"02134"` is not `2134`). With `acceptable[pointer]` a row matches when at least one value resolves and every resolved value is one of the acceptable ones. A pointer missing from both sides matches; missing only from the output does not.

Pointer values compare leniently by the answer key's type: the string `"1"` matches the number `1`, and strings compare case- and whitespace-insensitively. A row **passes** when it is valid JSON, valid against the schema, matches every pointer and, when a rubric runs, did not fail the rubric. Rows the judge could not grade are counted as unjudged, never as failures. Without a saved schema "valid" falls back to "a JSON object" and the metrics say `schema_missing`.

Metrics: `structured_plan_pass_rate` (the headline; also `exact_match`), `json_validity_rate`, `schema_valid_rate`, `schema_errors_top` (grouped by validator, pointer with array indices as `*`, and message), `field_match_rate`, `per_pointer` (match rate, support, missing rate, and `both_missing`: rows where neither the output nor the answer key has the field, which are not counted as missing), and with a rubric `rubric_pass_rate`, `rubric_mean_score` and a per-criterion table (mean score, low-score count, weight, critical). The fine-tuned model is paired with the base model on the pass rate. A cached base-model result is reused only under the same scoring fingerprint (schema, rubric, pointers, judge, retrieval).

The rubric judge grades every row, including rows that already failed the schema or pointer checks (so the per-criterion table covers all of them); it only changes the pass for rows whose checks passed. When the schema was saved differently from the one the split was prepared with, the eval prompts and scores with the current schema and the metrics carry `schema_changed_since_prep`. The rubric judge is chosen the same way as the answer judge (a local model first, cloud only when configured). With no judge reachable the rubric block reports `no_judge` and the pass ignores the rubric.

## Data

**Data > Generate: scenario grid** writes rows from a matrix of dimensions (for example region by tier by team): see the [synthetic data workflow](./data-ingestion.md). Every answer is validated against the saved schema before it is accepted.

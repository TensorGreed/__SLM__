---
title: Code remediation
sidebar_label: Code remediation
---

# Code remediation

The **Code remediation** recipe (`code-remediation`) trains a small coder model to fix one finding in a chunk of code. It is the follow-on to [Code findings](./code-findings.md): given the chunk, the finding (label, lines, evidence) and, when you have documents, some reference passages, the model returns

```json
{"patch": "--- a/app/db.py\n+++ b/app/db.py\n@@ -10,4 +10,4 @@\n ...", "rationale": "The id is bound as a parameter.", "citations": ["P1"]}
```

`patch` is a unified diff against the code shown (hunk headers use the absolute line numbers in the prompt), `rationale` a short reason, `citations` the ids of the reference passages it relied on (`[]` when none).

## Rows

```json
{"row_kind": "remediation", "code": "...", "language": "python", "path": "app/db.py", "start_line": 10, "end_line": 13,
 "finding": {"label": "injection.query", "start_line": 11, "end_line": 12, "evidence": "q = ...", "attrs": {}},
 "passages": [{"id": "P1", "text": "...", "source": "query-guidelines", "path": null, "start_line": null,
               "end_line": null, "symbol": null, "passage_id": "<document_id>:<chunk_id>"}],
 "remediation_json": {"patch": "...", "rationale": "...", "citations": ["P1"]}}
```

`passages` is optional and its ids are `P1..Pk` in order. A row without `finding` or `remediation_json` is rejected (`remediation_finding_missing`, `remediation_json_missing`). The recipe suggests `Qwen/Qwen2.5-Coder-7B-Instruct` (3B and 1.5B as smaller options). Rows train and evaluate through the model's chat template with byte-identical prompts.

**Code examples** (`row_kind: "code_example"`, `{task, language, passages, code_example_json: {code, rationale, citations}}`) are training-only rows the [synthetic playbooks](./data-ingestion.md) can write from your code documents: the model learns to write idiomatic code using the API in the passages. In an evaluation split they are skipped by the remediation scorer and counted in `skipped_non_remediation_rows`.

### Passages without passages

When a row has none, and `retrieve_passages` is on (default: on exactly when the project has a document index), the project's document index supplies them: the query is the finding label, its taxonomy title, the evidence and the first 400 characters of the code, through the project's served retrieval settings with `top_k` (default 3, 1 to 10). A hit that is the chunk itself is dropped. The same lookup runs when the training rows are prepared (the retrieved passages are stored on the prepared row, so held-out evaluation renders the identical prompt), in the playbooks and in evaluation for rows prepared another way. A row that already carries passages is never re-retrieved.

### Extra output fields

Declare more fields beside `patch`, `rationale` and `citations` with the output schema (`PUT /api/projects/{id}/recipe/output-schema`, `properties` with `type`, optional `enum` and `description`). The prompt names them with their descriptions, the target carries them, and held-out evaluation scores them with the same typed matching `json-extraction` uses (`extra_fields` in the metrics). They are reported, not gating.

## Scoring (`remediation`)

Per held-out row the prediction is parsed and then:

| Check | Meaning |
|---|---|
| malformed | not a JSON object with a `patch` string; every check below is false |
| `patch_applies` | the diff applies to the chunk: context and removed lines match exactly (trailing whitespace ignored) at the stated position or within 20 lines of it, hunks do not overlap, the file name matches. Failures are grouped by reason (`patch_reasons`) |
| `code_checkable` / `code_parses` | the original chunk passes at least one [code check](../extensions/code-checks.md) for its language (so the check can be trusted), and the patched chunk passes the same checks. Chunks nothing can check are left out of the rate |
| `issue_resolved` / `new_findings` | only with a [detector](#detector-for-issue_resolved): the finding's label, or any descendant label in your taxonomy, no longer appears for the patched code; labels the patched code newly shows are `new_findings` |
| `citations_valid` / `cited` | only for rows with passages: every cited id was one the model was given; at least one was cited |

A row **passes** when it applies, parses (where checkable), resolves the finding (where a detector ran) and cites only valid ids. Headline metric `remediation_pass_rate` (also stamped as `exact_match` for older gates); beside it `patch_applies_rate`, `code_parses_rate` with `code_checkable_count`, `issue_resolved_overall_rate` (resolved over every scored row; a patch that did not apply counts as not resolved; this is the one compared between runs) and, over only the rows whose patch applied, `issue_resolved_rate` and `new_findings_rate` with `detector_evaluated_count`, `citations_valid_rate` and `citation_rate`, `json_validity_rate`, `malformed_rate`, `patch_reasons`. Rates a run could not evaluate are absent, never zero. The lift summary pairs base and fine-tuned runs row by row on `remediation_pass_rate`.

The scaffolded pass/fail rules gate `remediation_pass_rate` (at least 0.40), `patch_applies_rate` (0.70) and `malformed_rate` (at most 0.05). Generic probe packs do not apply to this recipe.

### Judge

When a judge resolves (local model first, as for other long answers) every patch + rationale is graded against the reference fix with a remediation rubric: does the patch remove the finding without breaking the code, is the rationale right, are the citations relevant. The score lands in `judge_correct` beside the checks, for the base and the fine-tuned run alike. If the judge also reads retrieval verdicts, rows with passages are checked for whether the passages held the guidance (`retrieval_found_rate`). Turn it off per project with `judge: false`.

### Detector for `issue_resolved`

Set `detector` to one of:

- `{"kind": "project", "project_id": 12, "experiment_id": null}`: a [code-findings](./code-findings.md) project's completed run with weights (default: its latest completed non-baseline run). It runs on the original and on the patched chunk with that project's labels, **after** the remediation model's predictions are done; its model is loaded only then and freed again. Verdicts are cached per project by hash of the code and the detector, so a re-evaluation costs nothing.
- `{"kind": "command", "check_id": "my_detector"}`: an allow-listed [code check](../extensions/code-checks.md) registered with `kind = "detector"`. Its `run` reports `{"findings": [{"label", "start_line", "end_line"}]}` (as a `findings` list on the result, or a JSON message).

A finding the detector does not report on the *original* code cannot be shown resolved by a clean patched chunk: such rows are counted as `detector_blind_count` and left out of the resolved rates (and the pass rule). A detector that fails never fails the evaluation: the affected rows are left unevaluated (`issue_resolved` absent) and the metrics carry a `detector_error` note. The detector is always named `<project> · run <id>` in the interface and the metrics.

## Settings API

`GET /api/projects/{id}/remediation-config` returns `{config, effective, detector_label, available_detectors}`: what is stored, the defaults applied, the detector's name, and the code-findings projects you can use (completed weighted runs). `PUT` with `{detector, checks, judge, retrieve_passages, top_k}` replaces the settings; it answers 422 `INVALID_REMEDIATION_CONFIG` (nothing saved) for a detector project that is not a code-findings project with a completed weighted run, an unknown check id, or a `top_k` outside 1 to 10. `checks` limits which parse checks run (`null`: every runnable check for the language).

## Synthetic data

Two playbooks write rows for this recipe: **remediation from findings** (seeds are findings or fix-pair rows of one of your datasets and the answer key's dev set; the teacher writes patch, rationale and citations, and the patch is checked against the code it claims to fix) and **code examples** (from your code documents). Like the findings playbooks they never use the answer key's test set or sources marked evaluation-only, and report every rejected row grouped by reason.

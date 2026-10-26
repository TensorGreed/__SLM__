---
title: Code findings
sidebar_label: Code findings
---

# Code findings

The **Code findings** recipe (`code-findings`) trains a small coder model to read a chunk of code and list the findings in it: for each one a **label** from your [label taxonomy](./label-taxonomy.md), an inclusive **line range** (absolute file line numbers), the **evidence** text and optional **attributes**. A chunk with nothing to report answers `{"findings": []}`, and those clean chunks matter as much as the flagged ones.

## Rows

```json
{"row_kind": "findings", "code": "...", "language": "python", "path": "app/db.py", "start_line": 10,
 "findings_json": {"findings": [{"label": "injection.query", "start_line": 11, "end_line": 12,
                                  "evidence": "q = ...", "attrs": {}}]}}
```

Import them with a [code-dataset mapper](./data-ingestion.md) (labelled code, before / after fix pairs, or good / bad function pairs). The recipe suggests `Qwen/Qwen2.5-Coder-7B-Instruct` (3B and 1.5B as smaller options). Rows train and evaluate through the model's chat template, with byte-identical prompts; every code line is shown with its absolute line number, and the taxonomy's label ids are listed in the prompt (up to 80). A row without `findings_json` is rejected (`findings_json_missing`), never given a field-match prompt.

## Scoring (`finding_set`)

A predicted finding is correct when its label is equal and its line range overlaps the reference range with IoU of at least 0.5 (inclusive lines, one-to-one matching). Held-out evaluation reports:

- `precision`, `recall`, `f1` (micro), `precision_macro` / `recall_macro` / `f1_macro`, and a per-label table (`per_class`: precision, recall, f1, support, tp, fp, fn);
- `clean_chunk_fp_rate`: the share of clean chunks where the model reported anything;
- `malformed_rate` and `json_validity_rate`: answers that are not a `{"findings": [...]}` object count every reference finding as missed;
- `attr_accuracy` per attribute, over matched findings;
- an optional hierarchical score (a parent label earns partial credit; strict metrics never change).

The scaffolded pass/fail rules for this recipe gate `f1` (at least 0.50), `clean_chunk_fp_rate` (at most 0.20) and `malformed_rate` (at most 0.05).

## Synthetic data

Three playbooks (Data > Synthetic data) write findings rows; the teacher's claim is checked against the code it returns. Seeds come from the answer key's **dev** set (clean chunks with `findings: []`, or labelled chunks for hard negatives) and from **code documents** (git repository / website sources, treated as assumed clean). The answer key's test set is never used as a seed, and neither are sources marked evaluation-only or retrieval-only.

| Mode | What it does |
|---|---|
| `seeded_defect` | rewrites a clean chunk with one defect of a chosen label (`label_id`; a free label when the project has no taxonomy), optionally involving real field names from a data column (`context_dataset_id` + `context_column`; an answer-key dataset is refused with `context_dataset_not_allowed`) |
| `hard_negatives` | from flagged chunks, writes clean look-alikes: a fixed variant, a safe look-alike, a test fixture |
| `cluster_targeted` | seeded defects in the pattern of a failure cluster |

Every run reports `rejected_count`, `reject_summary` and `rejected_examples`. Reasons: `parse_error`, `empty_code`, `unchanged_code`, `label_mismatch`, `line_out_of_range`, `evidence_not_in_lines`, `evidence_too_short`, `lines_not_changed` (the finding must sit on a line the teacher changed), `code_too_long`, `has_findings` (hard negatives) and `unknown_label`. Seeds rotate from batch to batch, so a large run draws on the whole pool.

## Structured-output hint and probes

Applying the recipe stores a JSON Schema for `{"findings": [...]}` (label enum from the taxonomy) as the project's guided-JSON hint (`GET /api/projects/{id}/guided-json`); it is refreshed when the taxonomy changes and carried into the export manifest. The recipe's probe pack checks that the model reports nothing for empty or comment-only code and that its labels do not change when a reassuring or instruction-like comment is added or variables are renamed.

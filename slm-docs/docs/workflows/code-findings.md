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

**Seed sources and the seed pool.** Playbooks that seed from documents (the code-findings and code-remediation playbooks, `documents_multi_turn`) read at most 2,000 passages per run. When more are eligible, the run takes a deterministic sample spread evenly across (data source, language) groups instead of the first 2,000, so a large source cannot crowd out the others and a newly added source can be reached (each run draws a fresh sample, so over several runs every passage can be reached; a resumed run draws the same one, and a fixed `seed` pins it). A pool within the cap is used whole, in its usual order. Pass `source_ids` (a list of 1 to 100 data-source ids of this project) to draw seeds only from those sources; an id that is not a source of this project is `400 invalid_playbook_param` (`metadata.param = "source_ids"`), and sources with no usable chunks fail the run, naming them. Sources marked evaluation-only or retrieval-only are still excluded. The result (response, dry run and Job result) carries `seed_pool`: `chunks_eligible`, `chunks_used`, `sampled`, `by_source`, `by_language` and the requested `source_ids`; the notification bell shows it as a "seeds:" line. In the playbook picker, a "Seed sources" checklist (title, source id, document count, usage) sets `source_ids`.

**Seed filters.** Large repositories hold passages that have nothing to do with the task (build files, lock files, unrelated samples). Narrow the pool with `seed_path_include` and `seed_path_exclude` (lists of up to 50 globs) and `seed_relevance_terms` (up to 100 terms). A glob is matched, case-sensitively, against the passage's repository-relative path: `*` also crosses folders, so `src/**` and `**/*.test.*` work, and a leading `**/` also matches at the top level. A passage is kept when it matches at least one include glob (when any are given) and no exclude glob. Relevance terms then keep a passage when any term, case-insensitively, appears in its path or text; `seed_relevance_scope: "file"` keeps every passage of a file when any passage of that file matches. Sources listed in `relevance_exempt_source_ids` skip the term filter (path globs still apply to them). Filters run before the 2,000-passage cap and before each run's sampling, so the sample is drawn from what is left. A glob that does not compile, an over-long list or term, or a bad scope is `400 invalid_playbook_param` (`metadata.param` names it). Filters that leave nothing fail the run with the counts (how many passages each filter dropped, per source), never an empty success. With filters set, `seed_pool` also carries `filters` and `sources`: per source `total`, `kept`, `dropped_glob`, `dropped_relevance` and `exempt`; the bell adds "filters dropped N by path, M by relevance". The picker has a "Seed filters" block under the sources checklist (one glob or term per line).

## Structured-output hint and probes

Applying the recipe stores a JSON Schema for `{"findings": [...]}` (label enum from the taxonomy) as the project's guided-JSON hint (`GET /api/projects/{id}/guided-json`); it is refreshed when the taxonomy changes and carried into the export manifest. The recipe's probe pack checks that the model reports nothing for empty or comment-only code and that its labels do not change when a reassuring or instruction-like comment is added or variables are renamed.

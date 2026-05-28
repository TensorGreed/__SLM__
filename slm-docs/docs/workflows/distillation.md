---
sidebar_position: 5
title: Distillation
---

# Knowledge distillation (teacher → student)

The biggest quality lever a small model has is learning from a strong teacher's
**soft targets**, not just the gold label. BrewSLM does *offline* distillation in
three steps: capture the teacher's distribution once, train the student against
it (no teacher loaded at train time), then measure how much quality the student
kept.

> Offline KD aligns the teacher's per-token distribution with the student's
> tokens position-by-position, which is exact when teacher and student **share a
> tokenizer** (the common KD setup — e.g. distil a larger SmolLM/Qwen into its
> smaller sibling). Teacher tokens that don't resolve to a student id are
> dropped; if most drop, the soft signal weakens.

## 1 · Capture teacher logits

Point capture at a dataset; it calls your configured teacher
(`TEACHER_MODEL_API_URL`, the same one synthetic generation uses) asking for
top-k logprobs per token, and writes them to
`data/projects/<id>/distillation/teacher_capture.jsonl`.

```bash
POST /api/projects/{id}/distillation/capture
{ "dataset_id": 12, "top_k": 10, "limit": null }
→ 202 { "task_id": "distill-…", "status": "pending", … }

# poll
GET /api/projects/{id}/distillation/tasks/{task_id}
→ { "status": "completed", "produced_count": 184, "written_path": "…", "chunk_errors": [] }
```

Capture runs as a background task (returns in ms, poll for progress). A per-row
teacher failure lands in `chunk_errors` but the rest of the batch still runs —
one bad row costs one row, not the whole capture.

## 2 · Train in distillation mode

Pick a KD recipe — `recipe.kd.classification`, `recipe.kd.qa`, or
`recipe.kd.span_extraction` — or set `training_mode="distillation"` on any
causal-LM config. The trainer reads the capture artifact and optimizes:

```
L = α · CE(student, gold) + (1 − α) · T² · KL( softmax(student/T) ‖ softmax(teacher/T) )
```

Defaults are `α = 0.5` (`distillation_alpha`) and `T = 2.0`
(`distillation_temperature`). Per-step `distill_total_loss` /
`distill_ce_loss` / `distill_kd_loss` are logged separately so you can watch the
hard vs. soft contributions.

A **pre-train gate** refuses the run with an actionable message if no capture
artifact exists yet — so you never burn GPU time on a misconfigured distillation
run.

## 3 · Measure quality retained

After both the student and a **teacher baseline run** have eval results on the
same eval set, the Eval tab's *Distillation quality retained* panel shows
`quality_retained = student / teacher` per metric (and per slice when available):

```bash
GET /api/projects/{id}/evaluation/student-teacher-comparison/{student_eid}
    [?teacher_run_id=N]
```

The teacher run is resolved from `?teacher_run_id=`, the student experiment's
`config.teacher_baseline_run_id`, or the eval pack's optional
`teacher_baseline_run_id` — in that order. This is a pure read over stored
`EvalResult` rows; no new model or judge calls. A headline like
`81% retained` tells you at a glance how much of the teacher you kept at a
fraction of the size.

## See also

- [Training workflow](training.md) — the general training primitive.
- [Evaluation & remediation](evaluation-and-remediation.md)
- [Knowledge distillation (glossary)](../reference/glossary.md#knowledge-distillation-kd)

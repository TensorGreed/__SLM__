---
sidebar_position: 5
title: Evaluation + remediation
---

# Evaluation + remediation

Stage 9 of the [pipeline](pipeline-overview.md) plus the answer-key workbench (stage 3). Evaluation isn't just a number — it's the decision engine for what to fix next. BrewSLM wires the pass/fail rules, the answer key, the failure clusters, and the remediation suggestions into one tight loop.

## What the UI calls things

The UI uses plain names for eval data; the API, CLI and file names keep the technical ones.

| In the UI | What it is | API / code name |
|---|---|---|
| **Answer key** (practice / final) | Rows you wrote or approved with the correct answer | gold set (`gold_dev` / `gold_test`) |
| **Test examples** | The prepared test split, held back from training | `test` split (`prepared/test.jsonl`) |
| **Built-in checks** | Checks BrewSLM writes (robustness, refusal, grounding) | probe pack (`probe_pass_rate`) |
| **Custom checks** | Checks you write (INV / DIR / MFT) | behavioral tests |
| **Pass/fail rules** | Metric thresholds a run must meet | gates; the rule set is the eval pack |
| **Suggested answer-key rows** | "I don't know" rows drafted after a drift check | drift traps |

See the [glossary](../reference/glossary.md#answer-key) for each term.

## The default view: one summary card

The Eval tab opens on a single card that answers three questions for the
selected run (the latest trained run is picked for you):

1. **Is it better than the base model?** — verdict: *better*, *worse*, *no
   better*, *not evaluated yet*, or *evaluated, no base-model comparison yet*.
2. **By how much?** — the headline metric, base → fine-tuned, with the
   absolute and relative change. Both numbers come from the same **test
   examples**, and the baseline is always **this run's own base model**. Perplexity
   (continued pretraining) counts as better when it drops.
3. **Where does it still fail?** — up to 5 test examples it got wrong
   (Expected / Got), plus how many failed out of how many were evaluated.

If the run hasn't been evaluated yet, **Evaluate this run** queues an
exact-match eval on the test examples (the `test` split) as a background job (the bell tells you
when it's done). Real training runs are evaluated against the base model
automatically when they finish.

Test examples always means the prepared test split from Dataset Prep. The final
answer key (`gold_test`) is used only when the project has no prepared split, so the base model and
the fine-tuned model are always scored on the same rows.

Everything else — pass/fail rules and the scorecard, built-in and custom checks, failure
clusters, remediation, and the comparison panels — sits under **Advanced
evaluation**. It is collapsed by default for beginner projects and open for
everyone else.

API: `GET /api/projects/{id}/evaluation/summary?experiment_id=` returns
`verdict`, `headline` (`metric_id`, `baseline_value`, `trained_value`,
`absolute_delta`, `relative_delta_pct`, `direction`), `failures` (≤5),
`failed_count`, `evaluated_samples` and the run/baseline ids. Omit
`experiment_id` to get the latest completed non-baseline run. Held-out eval
results now store up to 20 failing rows in `details.failures_preview`.

## Step 1 — Build the answer key {#step-1--build-the-gold-set}

The **answer key** (API name: *gold set*) is the ground-truth labelled set you trust to grade everything else. Rows go into the **practice answer key** (`gold_dev`, scored while you iterate) or the **final answer key** (`gold_test`, kept for the final grade). Neither is a default training source. Quality > quantity: 50–100 carefully-labelled rows beats 1000 sloppy ones.

### UI

Pipeline → **Answer Key** → **Sample N rows**. Pick a strategy:

| Strategy | When |
|---|---|
| `random` | First answer key ever, no priors. |
| `stratified` | You have labels / classes / intents — sample evenly across them. |
| `targeted` | You've seen failure clusters — sample rows that match a pattern. |

For each sampled row, paste the correct answer and approve. The workbench tracks `pending` → `in_review` → `approved` / `rejected`. When all rows are approved, submit the version → it locks (draft → locked).

A locked answer-key version is **immutable**. New rows make a new version.

### Answer-key diagnostics (V4 ML-native viz)

The Answer Key tab renders a `GoldSetDiagnosticsPanel` above the entry list whenever the answer key has at least a few classification rows. Two complementary views from a single `GET /api/projects/{id}/gold/diagnostics` call:

- **Class-balance bars** — one row per label sorted descending. The bar shows share-of-total; a dashed line at **15%** marks the same imbalance floor Coach Mode's `class_imbalance` signal fires below, so a bar dropping under the line previews exactly what'll be flagged at training time. Classes below the floor turn red. Header reports total rows, class count, and Shannon entropy.
- **Class-similarity heatmap** — rows × cols = labels. Each cell is the mean pairwise Jaccard between a sample of rows from each class (default 12 per class):
    - **Diagonal cells** measure intra-class redundancy. ~1.0 = rows look the same → low diversity → the model learns nothing. ~0.0 = good variety.
    - **Off-diagonal cells** measure inter-class confusability. ~1.0 = even a perfect classifier can't separate the classes from text alone. ~0.0 = the classes are easy to tell apart.
- **Cells with `n/a`** = the class doesn't have enough rows for similarity scoring; named in a footnote rather than fabricated as 0.

Non-classification answer keys (span-extraction, summarization, qa-sft) render an empty-state hint — class balance is a classification concept. Span-extraction has its own diversity signals via the trainability forecast (`entity_type_coverage_thin`, `negative_examples_missing`).

### CLI

```sh
brewslm eval gold-set sample --project 1 \
  --strategy stratified \
  --count 100 \
  --stratify-by intent

# … label rows in the UI (CLI labeling is painful for free-form data) …

brewslm eval gold-set submit --project 1 --version 1
brewslm eval gold-set list --project 1
```

### API

```sh
# Sample
curl -X POST http://localhost:8000/api/projects/1/gold-sets/sample \
  -H "Content-Type: application/json" \
  -d '{"strategy": "stratified", "count": 100, "stratify_by": "intent"}'

# Submit (lock the draft)
curl -X POST http://localhost:8000/api/projects/1/gold-sets/1/submit
```

## Step 1b — Or: build the answer key with a cloud LLM {#step-1b--or-build-the-gold-set-with-a-cloud-llm}

Pipeline → **Answer Key** (the older tab next to the workbench) carries an
LLM-assisted path that works across all four task types: **qa-sft**,
**classification**, **span-extraction**, **summarization**. Use it when:

- You're bootstrapping — no labelled data yet, but you want eval *now*.
- You need 20–50 quick examples to validate the task-type shape works
  before investing in hand-labelled rows.
- You want hallucination-trap rows alongside the normal mix.

The flow is **preview-then-save**: the LLM emits a batch, you review row-by-row
with per-row Accept checkboxes, only checked rows persist into the JSONL.

### Picking provider + model + cost

A cost-estimate badge resolves on every form change (`≈ $0.0008 estimated`).
Pricing is approximate (±25%) — calibrated against vendor list prices as of
late 2025. Providers supported:

| Provider | Wire mapping | Default model |
|---|---|---|
| OpenAI | `provider=openai` | `gpt-4o-mini` |
| Anthropic | `provider=anthropic` | `claude-haiku-4-5` |
| Deepseek | `provider=openai` + `api_url=<deepseek host>` | `deepseek-chat` |

Stored API keys: when you first paste a key, opt in to **Save this key for
future generations** — the key is encrypted server-side, only a masked hint
(`sk-…xyz`) round-trips to the UI. Switch providers and the panel re-fetches
the per-provider stored key.

### Grounding

When **Ground in this project's source material** is on (default), the
service samples a strict-budget slice of the project's cleaned chunks
into the prompt and asks the LLM to anchor each row to a passage. Caps
keep cost bounded: max 6 chunks / 1500 chars per chunk / 8000 chars total.
The post-call response surfaces an actual `reference_chunk_count` so you
know how many made it.

### Row mix (qa-sft only)

Flip **Customize row mix** on the panel to control the difficulty +
hallucination-trap distribution:

```
5 EASY    direct lookups, single fact, answer clear from one passage
3 MEDIUM  2-3 facts combined, moderate inference
2 HARD    multi-hop reasoning, edge cases, judgment calls
2 TRAPS   answer is "I don't know" — tests refusal vs fabrication
```

The four bucket counts sum to the actual row count (replaces the simple
Count field). The LLM is told the breakdown explicitly + asked to tag
each row with its `difficulty` + `is_hallucination_trap` fields. Tags
round-trip into the JSONL via `/gold/import` so you can filter your
answer key by difficulty later.

### Review & edit prompt before sending (advanced)

Opt in via the **Review & edit prompt before sending** checkbox. Clicking
Generate then fetches `/preview-prompt` and renders two editable textareas
(user prompt + system prompt) inline. Token counts update as you type. Send
fires the real generate call with the edited strings as overrides. When the
user prompt is overridden, the classification vocab filter is suspended on
parse — your edited prompt's label vocabulary wins.

### Per-task-type row shapes

Generated rows + manual-add rows + saved rows all share the task type's
canonical shape on disk:

| Task type | Required fields | Optional metadata |
|---|---|---|
| `qa-sft` | `question`, `answer` | `difficulty`, `is_hallucination_trap`, `rationale`, `source_excerpt` |
| `classification` | `text`, `label` | `rationale`, `source_excerpt` |
| `span-extraction` | `text`, `entities: [{type, start, end, text}]` | `rationale`, `source_excerpt` |
| `summarization` | `document`, `summary` | `rationale`, `source_excerpt` |

The entries list renders each task type with its own row body:

- Classification → `text` + label-as-badge
- Span-extraction → `text` (monospace) + an entity list with `[start:end]` offsets
- Summarization → collapsed `<details>` document + visible summary
- qa-sft → difficulty + trap badges + Q+A

### Manual add — per-task-type inline form

The add-row form (**+ Add Row**) below the entries list switches its fields based
on the project's task type. For classification it shows Text + a Label
combobox (`<datalist>` autocompleted from existing answer-key labels). For
span-extraction the form has a Text textarea + an **Entities JSON** editor
with live offset validation, plus a "Highlight to select" helper:

1. Type the source text + highlight a range.
2. Type the entity type (autocompleted from existing types — soft amber
   border + "New type" hint when you introduce a brand-new one).
3. Click **+ Add highlighted span** (or press Enter in the Type input).
4. The span lands as a chip (`[email "jane@example.com" 8:24 ✕]`) and
   appears in the pretty-printed JSON below. Click ✕ on a chip to remove
   without hand-editing the JSON.

The same soft-amber drift warning fires on the classification **Label** input
when you type a new value not in the existing vocabulary — useful guard
against `positive` / `Positive (with sentiment)` fragmenting eval metrics.

### Filtering + mix summary (qa-sft only)

For qa-sft projects the entries list adds:

- A **mix summary** banner: `12 entries: 5 easy / 3 medium / 2 hard · 2 hallucination traps`.
- A **filter dropdown**: All / Easy only / Medium only / Hard only / Hallucination traps only.

Other task types hide both controls (`difficulty` / `is_hallucination_trap`
metadata is QA-flavored).

### API

```sh
# Generate
curl -X POST http://localhost:8000/api/projects/1/gold/generate-via-llm \
  -H "Content-Type: application/json" \
  -d '{"provider":"openai","model":"gpt-4o-mini","count":10,
       "api_key":"sk-…","ground_in_source":true}'

# Generate with explicit mix
curl -X POST http://localhost:8000/api/projects/1/gold/generate-via-llm \
  -H "Content-Type: application/json" \
  -d '{"provider":"openai","model":"gpt-4o-mini","count":10,
       "distribution":{"easy":5,"medium":3,"hard":2,"hallucination_traps":2}}'

# Preview the prompt (no LLM call, no cost)
curl -X POST http://localhost:8000/api/projects/1/gold/generate-via-llm/preview-prompt \
  -H "Content-Type: application/json" \
  -d '{"count":10,"ground_in_source":true}'

# Manually save one classification row
curl -X POST http://localhost:8000/api/projects/1/gold/add \
  -H "Content-Type: application/json" \
  -d '{"text":"Where is my refund?","label":"billing","dataset_type":"gold_dev"}'
```

## Pass/fail rules scaffold (task-type-aware starter)

If your project doesn't yet have a preferred pass/fail rule set (eval pack), the Eval tab surfaces a **Scaffolded pass/fail rule set** panel under the pack picker. The panel auto-generates a draft tailored to the project's `selected_recipe.recipe_id`:

| Task type | Metrics | Pass/fail rules (gates) |
|---|---|---|
| `classification` | macro_f1, accuracy | `min_macro_f1` ≥ 0.65, `min_accuracy` ≥ 0.70, `min_per_class_f1` ≥ 0.50, optional `min_safety_pass_rate` ≥ 0.93 |
| `span-extraction` | span_set_f1, span_set_precision, span_set_recall | `min_span_set_f1` ≥ 0.65 + per-side precision/recall thresholds |
| `summarization` | rouge_l, groundedness | `min_rouge_l` ≥ 0.30, `min_groundedness` ≥ 0.82 |
| `qa-sft` | exact_match, f1, llm_judge_pass_rate | `min_exact_match` ≥ 0.45, `min_f1` ≥ 0.60, `min_llm_judge_pass_rate` ≥ 0.75 |
| `generic-sft`, `code-review` | exact_match/f1/llm_judge_pass_rate | Pragmatic defaults tuned per task type |

Each rule's threshold + required flag is editable inline. Click **Use scaffold** to persist — the draft is saved to `project.runtime_config["scaffolded_evaluation_pack"]` and `evaluation_preferred_pack_id` flips to `evalpack.project.scaffolded`, which the eval pack resolver routes through the new `project_scaffold` source.

API:

```sh
# Recipe-derived draft (NOT persisted).
curl http://localhost:8000/api/projects/1/evaluation/pack-scaffold

# Save the edited draft. The scaffolded pack id is forced on save
# regardless of what the client sends.
curl -X POST http://localhost:8000/api/projects/1/evaluation/pack-scaffold \
  -d '{"draft_pack": {...}}'
```

## Catch problems before training: trainability forecast

The answer key is also the input to the **trainability forecast** (Training Config page). The forecast runs the same task-type-aware signal sweep on the answer key *before* training so you can patch the data shape upfront instead of waiting for the eval to surface it. Signals overlap with the failure clusters below (per-class starvation, label-vocab drift, span-offset rot, summary/doc mismatch) but trigger at design time, not post-mortem. See [Training → Trainability forecast](training.md#trainability-forecast) for the full signal list.

## Fix-in-answer-key deep links

When you expand a failure-cluster card on the Eval tab, a **Fix in answer key** button next to the cluster-augment control deep-links into the answer-key workbench with the LLM-gen panel pre-configured: `distribution.hallucination_traps` defaults to 5 and the focus textarea is prefilled with a one-line summary of the cluster's failure pattern (reason code + classifier explanation). The destination panel shows a dismissible "Generating traps for cluster X" banner so you can verify what was prefilled before clicking Generate. Non-qa-sft recipes still get the focus hint; the trap distribution UI is qa-sft-only.

## Eval comparison + "Fix the gap" rollback

When you've run eval against two experiments in the same project, the Eval tab shows a **Compare to #\<id\>** button that deep-links into a side-by-side comparison page at `/project/{id}/eval/compare?a=<exp>&b=<exp>`. The page renders:

- Two side cards with pass-rate badges and a winner marker.
- A metric-delta table with regressions sorted to the top (direction-coded against `higher_is_better` so `eval_loss` is graded inverted).
- Failure-cluster diff: **New in B (regressions)**, **Resolved in B (fixed)**, and **Shared clusters** with per-cluster delta.
- Config diff with primary fields (`base_model`, `learning_rate`, `num_epochs`, etc.) always visible plus any other changed knobs.

When B regressed, a red **Fix the gap** banner offers a one-click "Rerun experiment #A" button that posts to the existing `rerun-from-manifest` endpoint. The rerun spawns a NEW experiment that replays A's resolved config + dataset snapshot — A and B both stay untouched. If A never produced a manifest (e.g. it failed mid-training), the click surfaces a toast explaining only completed runs can be rolled back to.

## SLM-vs-frontier benchmark report

The Eval tab also answers the production buying question — *"is my cheap local model good enough vs. just calling gpt-4o-mini?"* — as an honest report: **"Your model is X% as good as gpt-4o-mini at Y% of the cost and Z× the latency."** with a per-metric table behind it.

- **Quality** is a pure ratio over stored eval results — `slm / frontier` per shared metric — and needs a **frontier baseline run**: the frontier model evaluated on the *same* eval set. Resolve it via `?frontier_run_id=`, the experiment's `config.frontier_baseline_run_id`, or the eval pack. Without one the quality block soft-falls-back to `no_frontier_eval` (the report **never fabricates** a quality number), and the per-metric table marks where the SLM is `behind`, not only where it wins.
- **Cost + latency** are labeled by provenance. Frontier figures come from a small **published-reference table** (gpt-4o-mini / gpt-4o / claude-3.5-haiku public pricing + typical latency, `as_of`-stamped). SLM figures are `estimated` from the project's latest model-benchmark sweep (throughput → $/1M tokens at the GPU hourly rate; latency from the same sweep). No sweep yet → the SLM side is `unavailable` with a CTA, never guessed.
- Pick the comparison target with `?frontier_model_id=` (default `gpt-4o-mini`). API: `GET .../evaluation/frontier-comparison/{experiment_id}`.

> To populate the quality number, evaluate a frontier model on your answer key and point the report at that run (a one-click frontier-eval runner is planned follow-up work).

## Suggested answer-key rows: drift-triggered hallucination-trap refresh (admin opt-in)

Projects can opt into automatic hallucination-trap refresh from the **Suggested answer-key rows** panel under the Eval tab (API name: *drift traps*). When auto-refresh is off, the panel shows an opt-in banner with a one-click "Enable auto-refresh" button; when it's on, a status chip surfaces the per-refresh trap count and a "Disable" link. Either way, the panel's "Generate now" button always works — it hits the manual `POST /drift/refresh-traps` endpoint and reloads the queue.

Each pending row carries the cluster pattern (`reason_code`) that motivated it plus a task-type-shaped preview of the trap. Per-row Accept and Reject buttons triage in place: accepting appends the row to the final answer key (`gold_test.jsonl`); rejecting marks the row in the audit trail. Switch the status filter to **Accepted**, **Rejected**, or **All (audit)** to see triaged rows after the fact.

Settings persist under `runtime_config.drift_refresh_traps` (enabled + count, count clamped to [1, 20]). When opted-in, every per-deployment drift check fires the trap-refresh runner alongside its normal eval, populating the queue with fresh traps targeting the last 7 days of failure-cluster patterns.

Manual trigger:

```sh
# Generate N traps targeting recent cluster patterns. ?simulate=true
# bypasses the LLM for dev where no API key is wired.
curl -X POST "http://localhost:8000/api/projects/1/drift/refresh-traps?count=5&simulate=true"

# Triage queue — newest-first list of pending rows.
curl "http://localhost:8000/api/projects/1/drift/review-queue"

# Accept (→ appends to gold_test.jsonl) or reject one row.
curl -X POST "http://localhost:8000/api/projects/1/drift/review-queue/42/triage" \
  -d '{"accept": true, "note": "good trap"}'
```

Each queue row carries the cluster pattern (`reason_code`, `signature`) that motivated the trap so the reviewer sees *why* it was generated. Accepted rows are appended to the project's `gold_test.jsonl` and the dataset's `record_count` is bumped; rejected rows stay in the queue with the user's reason for audit. Re-triaging a row returns `409 queue_row_already_triaged` — once triaged, always triaged.

The runner falls back to deterministic placeholder traps when no LLM API key is stored on the project, so a weekly drift-check tick still populates the queue if the user opted in without wiring credentials.

## Remediation outcome tracking (admin)

Every click on a suggested-action button — from the trainability forecast panel and from the failure-cluster cards — fires a fire-and-forget `POST /api/projects/{id}/remediation/events` with `{kind, params, outcome}`. When the next eval result lands for the project, `evaluate_experiment_auto_gates` stamps every pending event in the window with `evaluation_lift_pct = (current_pass_rate - previous_pass_rate) × 100`. `GET /api/admin/remediation/outcomes?kind=<action_kind>` aggregates by kind with median + mean lift, positive-lift count, and a positive-lift rate so admins can spot suggestion sources that get clicked but don't correlate with improvements. No UI in v1 — admin reads the JSON.

## Step 2 — Pick pass/fail rules

A project's **pass/fail rule set** (API name: *eval pack*) bundles task-aware metric schemas + gate policies — the individual **pass/fail rules**. Built-in packs cover the common cases; you can scaffold custom packs via [Extensions → Scaffold](../extensions/scaffold.md).

| Pack id | Use it for |
|---|---|
| `evalpack.general.default` | Sanity-check defaults for any task. |
| `evalpack.qa.strict` | Q&A with strict exact-match + hallucination filter. |
| `evalpack.classification` | Macro-F1 + class-imbalance gates. |
| `evalpack.summarization` | ROUGE-L + length sanity. |
| `evalpack.preference` | DPO/ORPO chosen-vs-rejected pass rate. |

Each pack's task spec defines:

- `required_metric_ids` — metrics that must be present.
- `gates` — per-gate threshold + `required` flag. Failing a `required` gate fails the whole eval.
- `metric_schema` — descriptions + expected ranges for each metric.

## Automatic "did fine-tuning help?" check

After every **real** training run completes, BrewSLM queues a `post_training_lift_eval` job; watch it in the notification bell. The job does three things:

1. **Evaluates the base model.** It runs the un-fine-tuned base model on the test examples (the `test` split) (up to 100 rows, greedy decoding). The result is cached per base model, and it's reused until `prepared/test.jsonl` changes.
2. **Evaluates your run.** It runs the fine-tuned checkpoint on the same split with the same settings.
3. **Reports the lift.** It reports the headline metric, for example `exact_match: 0.10 → 0.80 (better than base)`. The **Did SFT help?** panel on the Eval tab shows the full comparison for the selected run. It always compares against the baseline for **that run's base model**, never another model's. The Eval tab's summary card also says how solid that lift is: how many test examples fine-tuning helped, hurt, or left unchanged compared with the base model, and the 95% range for the average per-row change. If the range includes zero, the card's title becomes "Ahead of the base model — but within noise" instead of "Better than the base model"; with fewer than 5 paired rows it adds "too few rows to be sure". A gain beyond row-to-row noise still carries a reminder that it is one training run. This needs per-row scores from both evals, which are kept from now on (headline metrics `f1`, `exact_match`, `accuracy`); a base-model result recorded before that is re-evaluated once, and older run results show no note until they are re-evaluated. The bell's result line and the Coach's Export suggestion follow the same verdict: a change within noise reads `f1: 0.106 → 0.136 (within noise · 14 rows better, 6 worse)` in the bell instead of "better than base", and on the Export tab the Coach says the model "isn't clearly different from its base model" rather than calling it a win or a downgrade. Under **Advanced evaluation**, each metric row in the **Did SFT help?** panel carries the same check ("within noise · 14 rows better, 6 worse"), and a difference the rows can't back up is not coloured green or red. The Auto-RAG comparison's bell line qualifies its lift the same way.

**Across seeds.** The row-level check covers which test examples were sampled; it cannot see that the same recipe trained with a different random seed gives a different model. **Check across 3 seeds** on the summary card re-trains the run's exact config as one seed group (3 runs), and when the group finishes the automatic lift check scores every seed against the base model. The card then reads `f1: 0.074 (base) → 0.194 ± 0.005 (fine-tuned, mean of 3 seeds)` with a line such as "Holds across seeds: F1 0.189, 0.197, 0.196 — every seed beat the base model", or "Seeds disagree" when the seeds land on both sides of the base model (the title then says "but the seeds disagree" instead of "Better than the base model"). The per-seed runs are listed; the five failures and the row counts come from the median seed, named on the card. The Coach's Export suggestion and the bell line follow the seed verdict. You can also set *Seeds* in the training config to launch a seed group directly.

The job is skipped, with the reason recorded on the training job, for:

- simulated runs;
- seed-group children (the leader covers the group);
- runs without saved weights;
- projects with `runtime_config.auto_lift_eval = false`.

It can also be turned off globally with `AUTO_LIFT_EVAL_ENABLED=false`.

API: `GET /api/projects/{id}/evaluation/sft-lift-summary?experiment_id=<run>` pins the comparison to one run. Leave out `experiment_id` to get the latest run.

## Step 3 — Run evaluation

### UI

Pipeline → **Eval** → **Run evaluation**. Pick the trained experiment + pass/fail rule set (eval pack). Click **Start**. The page fills with:

- **Pass/fail rule row per metric** — pass / fail + score.
- **Classification breakdown** — for classification eval results, a per-class P/R/F1 bar chart (sorted by F1 ascending — the worst class lands at the top, the row the user actually needs to look at) plus a confusion-matrix heatmap (gold rows × predicted columns, diagonal green, off-diagonal red, intensity scaled by row share). An `unparsed` column appears whenever the model emitted a label outside the candidate set — distinct visual treatment so "wrong known class" and "emitted gibberish" don't read the same. See `ClassificationChartsPanel`.
- **Failure cluster card** below — folds errors by `(reason_code, signature)`.
- **Remediation suggestions card** — per cluster, what to try next.

### CLI

```sh
brewslm eval run --project 1 \
  --experiment 42 \
  --pack evalpack.qa.strict

# Get the failure clusters
brewslm eval clusters --project 1

# Get remediation suggestions
brewslm eval remediate --project 1 --eval-result 17
```

### API

```sh
curl -X POST http://localhost:8000/api/projects/1/eval/run \
  -H "Content-Type: application/json" \
  -d '{"experiment_id": 42, "pack_id": "evalpack.qa.strict"}'

curl "http://localhost:8000/api/projects/1/eval/17/clusters"
curl "http://localhost:8000/api/projects/1/eval/17/remediation"
```

## Step 4 — Inspect failure clusters

The eval-stage failure clusters (P12) live separately from the cross-stage [failure clusters](../observability/failure-clusters.md) on the Observability page. They're scoped to a single eval result and group prediction failures by signature.

### Common cluster patterns

| Cluster | Means | Common fix |
|---|---|---|
| Coverage gap | Same wrong answer across rows with a missing concept. | Add more training data covering that concept; or augment via synthetic. |
| Formatting mismatch | Right answer, wrong format (JSON vs text, extra whitespace). | Stricter adapter; add system prompt instruction. |
| Reasoning failure | Multi-step answers go wrong mid-way. | Either move to a larger base model or add chain-of-thought style examples. |
| Safety / policy violation | Output crosses a guardrail. | Tighten the domain pack's safety hook + retrain with refusal examples. |
| Off-by-one (classification) | Confusing two adjacent labels. | More examples of the boundary case. |

Click any cluster → **Exemplars** → see the actual model output vs the answer key. The drilldown also links to the originating RunEvent + the dataset row id.

## Step 5 — Apply remediation

The remediation card surfaces suggestions per cluster. Each is one of:

- **Data op** — "Add 20 rows like X" / "Filter rows where Y" / "Augment via synthetic".
- **Training op** — "Add 1 epoch" / "Reduce LR" / "Switch recipe to safe-balanced-sft" / "Try a larger model".
- **Config op** — "Tighten adapter output_contract" / "Enable safety hook X".

Pick the top 1–2 suggestions. Apply (the card has an **Apply** button for low-risk ones) or do them manually. Then:

```sh
brewslm train rerun --experiment 42        # if no manifest change
brewslm train clone --experiment 42 \      # if config changed
  --config-overrides '{"num_epochs": 4}'
brewslm eval run --project 1 --experiment <new> --pack evalpack.qa.strict
```

The fast loop should be **2–3 hours** per iteration once you've gotten the rhythm down. If it takes longer, the fix list is probably too big — narrow it.

## Compare experiments

When you have two candidate runs, the comparison view is the right surface.

### UI

Pipeline → **Eval** → **Compare** → pick experiment A + B. The page renders metric deltas, gate pass / fail per pack, side-by-side failure clusters, and a recommendation ("A wins" / "B wins" / "too close").

### CLI

```sh
brewslm eval compare --project 1 --experiment-a 42 --experiment-b 43
```

### API

```sh
curl "http://localhost:8000/api/projects/1/eval/compare?a=42&b=43"
```

## Pass/fail rules mindset

Your pass/fail rules (the eval pack's gate policy) are your **safety rail**, not bureaucracy. A good policy answers:

- **Must pass** — `required` rules that block promote on fail.
- **Can degrade slightly** — `optional` rules with a tolerance.
- **Always unacceptable** — safety / hallucination thresholds with no tolerance.

Pass/fail rules feed directly into the [Deployability score](../deployment/rollback-and-score.md) and the deployment promote check. Don't loosen rules just to pass — that's how regressions ship.

## Reason codes you might hit

| Code | Means |
|---|---|
| `eval_runtime_error` | Generic failure inside the eval runner. Check the timeline. |
| `eval_dataset_missing` | Eval pack referenced a dataset (e.g., the answer key) that no longer exists. |
| `eval_judge_unavailable` | LLM judge call failed (provider down, quota, config). The eval will fall back to non-judge metrics; see decision log. |

## Next

- [Failure clusters (cross-stage)](../observability/failure-clusters.md) — the project-wide view.
- [Training](training.md) — when you need to rerun after a fix.
- [Export + deployment](export-and-deployment.md) — once eval is green.

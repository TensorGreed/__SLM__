---
sidebar_position: 5
title: Evaluation + remediation
---

# Evaluation + remediation

Stage 9 of the [pipeline](pipeline-overview.md) plus the gold-set workbench (stage 3). Evaluation isn't just a number — it's the decision engine for what to fix next. BrewSLM wires the eval pack, the gold set, the failure clusters, and the remediation suggestions into one tight loop.

## Step 1 — Build the gold set

The **gold set** is the ground-truth labelled set you trust to grade everything else. Quality > quantity: 50–100 carefully-labelled rows beats 1000 sloppy ones.

### UI

Pipeline → **Gold set** → **Sample N rows**. Pick a strategy:

| Strategy | When |
|---|---|
| `random` | First gold set ever, no priors. |
| `stratified` | You have labels / classes / intents — sample evenly across them. |
| `targeted` | You've seen failure clusters — sample rows that match a pattern. |

For each sampled row, paste the gold answer and approve. The workbench tracks `pending` → `in_review` → `approved` / `rejected`. When all rows are approved, submit the version → it locks (draft → locked).

A locked gold version is **immutable**. New rows make a new version.

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

## Step 2 — Pick an eval pack

An **eval pack** bundles task-aware metric schemas + gate policies. Built-in packs cover the common cases; you can scaffold custom packs via [Extensions → Scaffold](../extensions/scaffold.md).

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

## Step 3 — Run evaluation

### UI

Pipeline → **Eval** → **Run evaluation**. Pick the trained experiment + eval pack. Click **Start**. The page fills with:

- **Gate row per metric** — pass / fail + score.
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

Click any cluster → **Exemplars** → see the actual model output vs gold. The drilldown also links to the originating RunEvent + the dataset row id.

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

## Promotion gate mindset

The eval pack's gate policy is your **safety rail**, not bureaucracy. A good policy answers:

- **Must pass** — `required` gates that block promote on fail.
- **Can degrade slightly** — `optional` gates with a tolerance.
- **Always unacceptable** — safety / hallucination thresholds with no tolerance.

Eval gates feed directly into the [Deployability score](../deployment/rollback-and-score.md) and the deployment promote check. Don't loosen gates just to pass — that's how regressions ship.

## Reason codes you might hit

| Code | Means |
|---|---|
| `eval_runtime_error` | Generic failure inside the eval runner. Check the timeline. |
| `eval_dataset_missing` | Eval pack referenced a dataset (e.g., gold set) that no longer exists. |
| `eval_judge_unavailable` | LLM judge call failed (provider down, quota, config). The eval will fall back to non-judge metrics; see decision log. |

## Next

- [Failure clusters (cross-stage)](../observability/failure-clusters.md) — the project-wide view.
- [Training](training.md) — when you need to rerun after a fix.
- [Export + deployment](export-and-deployment.md) — once eval is green.

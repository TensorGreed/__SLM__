# Sentiment Classifier Pipeline — Narration Skeleton

Status: ready for the inspect-only path. ONNX-INT8 export story is
the lingering partial — narration must mark it as "the natural
target, not yet validated."

Target length: 8–10 minutes (≈1300 words).

Companion to:
`docs-demo/videos/05-sentiment-classifier-pipeline/recording-plan.md`.

---

## Cold open (0:00–0:25)

> "The sentiment classifier is the simplest of the three official
> samples. Three labels — positive, neutral, negative — applied to
> short product reviews. If you've ever trained a classifier in a
> tutorial, this is that, with BrewSLM's pipeline scaffolding around
> it. Plus one extra teaching beat at the end about mobile / ONNX
> export, since this sample's target profile is `mobile_cpu`."

## Section 1 — Seed + Data tab (0:25–1:45)

**Action**: click Sentiment classifier tile, land on Data tab.

> "I click the Sentiment classifier tile. Standard seeder behavior —
> 30 source rows, 200 gold rows, a 22-4-4 train-val-test split.
> 
> The source CSV is exactly balanced: 10 positive, 10 neutral, 10
> negative. That's the demo's deliberate setup so the model can't
> shortcut by always predicting the majority class."

**Action**: expand one row (doc id 91 from selector pass).

> "Each row is a short review and a single label. This shape is what
> the classification adapter expects."

## Section 2 — Gold Set + label distribution (1:45–3:00)

**Action**: switch to Gold Set.

> "Gold has 200 entries. The distribution is slightly skewed: 70
> positive, 65 neutral, 65 negative. That's a realistic skew —
> people are slightly more likely to write a 5-star review than a
> 1-star one, and neutral reviews are rare in the wild because most
> people who write a review feel something one way or the other.
> 
> When you build your own classifier, this is the distribution you
> want to *measure* against. Training on a perfectly balanced
> dataset and then evaluating on a real-world-skewed gold gives you
> an honest signal."

## Section 3 — Dataset Prep + classification adapter (3:00–4:30)

**Action**: switch to Dataset Prep, open Schema Profile.

> "Dataset Prep. The adapter applied here is `classification-label`
> — different from the Support FAQ's `qa-pair` and the PII
> Detector's `structured-extraction`. The adapter is what makes the
> trainer treat this as a classification problem instead of a
> sequence-generation problem.
> 
> Look at the Schema Profile panel: the label vocabulary is right
> there — `positive`, `neutral`, `negative`. The field mapping is
> `text → label`. Twenty-two rows in train, four each in val and
> test."

## Section 4 — Tokenization light tour (4:30–5:30)

**Action**: switch to Tokenization.

> "Tokenization analyzes the length distribution of your prepared
> data. For classification, sequence length matters more than for
> generation tasks — you're typically batching tens or hundreds of
> short inputs per forward pass, and your max sequence length sets
> the memory ceiling.
> 
> For mobile deployment, you'd target a max length of maybe 128 or
> 256 tokens. We won't run the analyzer in this video — it needs the
> transformers library and a tokenizer download — but this is where
> you'd check that your data actually fits the budget."

## Section 5 — Training Config + mobile target (5:30–7:00)

**Action**: Training Config Page.

> "Training Config. Look at the recipe defaults — the manifest's
> `target_profile=mobile_cpu` and `training_preferred_plan_profile=fast-iteration`
> influence the suggested settings. Smaller batch, shorter sequences,
> fewer parameters. The Hardware Auto-Tuner button will recommend a
> base model sized for your target.
> 
> Don't click Apply Recipe yet — you'd overwrite your base model
> selection. For this video we're staying with the defaults and
> walking surfaces."

**Action**: flip to Advanced → Power Tools.

> "If you flip to Advanced mode and open Power Tools, you can tune
> LoRA rank. For a three-class classifier on small data, the default
> rank 8 is usually plenty. Bumping rank 16 doesn't help much for
> classification — it's a span/structure-task lever, not a
> classification lever."

## Section 6 — Evaluation surface (7:00–8:00)

**Action**: switch to Evaluation.

> "Evaluation. Empty until we run an experiment. When we do, this
> surface emits accuracy and macro-F1 — that's the canonical
> classification eval pack — plus per-class precision and recall.
> The eval pack `evalpack.classification.default` ships with the
> repo; you can see it referenced in the prepared manifest."

## Section 7 — Compression + Export (mobile story) (8:00–9:15)

**Action**: switch to Compression, then Export.

> "Compression and Export. This is where the mobile story would land
> for this sample, but I'm going to mark it clearly as 'partial.'
> 
> The manifest declares `target_profile=mobile_cpu` and mentions
> ONNX-INT8 export as the natural endpoint. The export panel does
> support ONNX — it's in the format dropdown — and the compression
> service has a real quantization path in `backend/scripts/quantize.py`.
> 
> But we haven't actually run an ONNX-INT8 export on this sample
> end-to-end yet. Whether the local toolchain — `optimum`,
> `onnxruntime`, the rest — is installed and works on this machine is
> a question we resolve in Video 11. Watch this space."

## Wrap (9:15–9:45)

> "Sentiment classifier walkthrough done. The simplest of the three
> samples and a good shape for understanding classification
> end-to-end.
> 
> Three takeaways:
> 
> One — classification adapters force a `text → label` contract.
> 
> Two — gold distribution should look like your real-world target,
> not your perfectly-balanced training set.
> 
> Three — the mobile / ONNX export path exists in the codebase but
> isn't proven end-to-end yet. That's Video 11's problem.
> 
> Next video walks the dataset lifecycle in detail — cleaning, gold,
> synthetic — across all three samples in one shot."

---

## Things to **not** say

- Don't say "ONNX-INT8 export works end-to-end" — it has not been
  proven. Mark as partial.
- Don't claim a synthetic-data path specifically for classification —
  open Q12. The generic Q&A synthetic path works for some shapes;
  whether it works for `{text, label}` data is unverified.
- Don't say "the model fits on a phone" — that depends on the base
  model choice, which we haven't made yet.

## Optional advanced notes

- The classification eval handler dispatches via
  `backend/app/services/eval_task_handler_service.py` based on
  `task_profile=classification`.
- Quantization paths in `backend/scripts/quantize.py` cover GGUF
  and ONNX; the latter is the relevant path for mobile_cpu targets.
- Manifest preferred plan profile `fast-iteration` is one of three
  built-in profiles (`fast-iteration`, `balanced`, `quality`); see
  `backend/app/services/training_recipe_service.py`.

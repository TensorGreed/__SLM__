# Sentiment Classifier Pipeline — Narration

Status: **synced** with the actual narrated take produced by
`tts/generate_v05_narration.py` (Orpheus voice "leo") on 2026-05-20.

The **Python script** at
[tts/generate_v05_narration.py](../../../tts/generate_v05_narration.py)
is the **authoritative source** of the spoken text. This file mirrors
the same text plus stage directions / Playwright cues. Edit the
script first.

Total runtime: **2:26** (matches
`docs-demo/recordings/raw/05-sentiment-classifier-pipeline-narrated.mp4`).
Section timings come from `tts/audio/v05-durations.json`.

Companion to:
[docs-demo/videos/05-sentiment-classifier-pipeline/recording-plan.md](../../videos/05-sentiment-classifier-pipeline/recording-plan.md).

This is the third sample in the inspection arc. Two beats are
distinctive to this video: **class balance** (10/10/10 source,
70/65/65 gold) and the **mobile_cpu target** which hints at an
ONNX-INT8 export story — marked as future, not validated.

---

## Pre-roll (not narrated)

Playwright logs in as admin, clicks the **Demo · Sentiment classifier**
tile, lands on the Data tab. ~5 seconds before narration starts.

## Section 1 — Cold open (0:00–0:15)

**On screen**: Data tab loaded; 30 raw documents listed below.

> "Welcome to the Sentiment Classifier pipeline walkthrough. This is
> the simplest of the three samples — three-way classification with
> the labels positive, neutral, and negative. Thirty source rows,
> perfectly balanced ten, ten, ten. Each row is text and a single
> label."

## Section 2 — Data tab + expand a raw row (0:15–0:32)

**On screen**: click `[data-testid^="expand-doc-"]` on the first
row; the `text` and `label` payload expand inline.

> "Data tab. Thirty source reviews. Each row has two columns: the
> text, and the gold label. Expand one. You see exactly the shape
> the model has to learn — read a review, emit one of three labels.
> The balance matters: ten of each class means the model never gets
> to cheat by always predicting the majority."

## Section 3 — Gold Set tab (class distribution callout) (0:32–0:51)

**On screen**: click **Gold Set**; "Entries 200" badge visible.

> "Gold Set. Two hundred entries. The distribution is seventy
> positive, sixty-five neutral, sixty-five negative. Slightly skewed
> positive — typical of real-world reviews. The eval handler
> measures per-class precision and recall against this gold, so
> under-represented classes still get measured."

## Section 4 — Dataset Prep tab (labels) (0:51–1:07)

**On screen**: click **Dataset Prep**; Schema Profile panel shows the
three labels.

> "Dataset Prep. Schema Profile shows the three labels. The adapter
> is classification-label — it canonicalizes every prepared row to
> a text column and a label column. Splits are twenty-two train,
> four validation, four test. Small, but enough to verify the loop
> end to end."

## Section 5 — Tokenization tab (mobile angle) (1:07–1:21)

**On screen**: click **Tokenization**.

> "Tokenization. Same idea as the previous samples. The twist for
> this sample: target profile is mobile CPU, so max sequence length
> matters more than usual. Short sequences mean a smaller model
> footprint and faster inference on-device."

## Section 6 — Training tab → Training Config (mobile target) (1:21–1:37)

**On screen**: click **Training** → click **Open Training Config →** →
land on `/project/<id>/training-config` → click **Advanced**.

> "Training tab — empty, expected. Into the Training Config page.
> Flip to Advanced. The Training Config picks up the mobile CPU
> target profile from the manifest — that hints at smaller batches,
> shorter sequences, and a tighter model footprint on export.
> Defaults are tuned for mobile."

## Section 7 — Evaluation tab (classification pack) (1:37–1:50)

**On screen**: navigate to `/project/<id>/pipeline/eval`; "No
experiments to evaluate" empty state.

> "Evaluation tab. Empty until we have an experiment. For this
> sample the eval pack is the classification default — accuracy and
> macro-F1 in the headline, per-class precision and recall in the
> detail panel."

## Section 8 — Compression + Export (ONNX-INT8 future) (1:50–2:08)

**On screen**: click **Compression** → click **Export**; export
format dropdown including `onnx` is visible.

> "Compression and Export. The natural target for this sample is
> ONNX with eight-bit quantization, which would give us a fast
> on-device model. ONNX is in the export format list, but the
> end-to-end story for this sample isn't validated yet — that's
> Video Eleven. For now we're just confirming the shape of the
> export surface."

## Section 9 — Wrap (2:08–2:26)

**On screen**: hold on Export tab.

> "And that's the third sample. Three task profiles, three scoring
> contracts, one shared pipeline. Quickstart, support FAQ, PII
> detector, sentiment classifier — that's the inspection arc
> complete. Next videos pick up the runtime-heavy side: actually
> launching a training run, scoring against gold, compressing, and
> serving."

---

## Things to **not** say

- Don't claim the ONNX-INT8 export *has* been validated — it
  hasn't. Say "the natural target" or "in the export format list",
  not "it works."
- Don't speak the manifest's `target_profile = mobile_cpu`
  literal. Say "the mobile CPU target profile" in plain English.
- Don't list per-class gold counts beyond what's said (70/65/65) —
  reading more numbers aloud feels recital-style and the on-screen
  display covers the detail.
- Don't read literal tech tokens (env var names, REST paths,
  adapter literal strings) — same rule as the other videos.

## Optional technical notes (background; not spoken)

- Eval pack id: `evalpack.classification.default`. Configured via
  the manifest's `evaluation.preferred_pack_id`. Dispatches to the
  classification handler in
  [backend/app/services/eval_task_handler_service.py](../../../backend/app/services/eval_task_handler_service.py).
- ONNX export is in the export format enum at
  [backend/app/models/export.py](../../../backend/app/models/export.py)
  but no successful export run has been observed in this pass. Open
  question Q23 / Q24 — parked here, answered by Video 11.
- Source distribution is exact 10/10/10 by construction; gold
  70/65/65 reflects the real-world skew toward positive reviews
  even when curating.
- Q12 (classification-specific synthetic path) is still open —
  the synthetic tab is intentionally skipped in this video to avoid
  showing an unverified surface.

## Why no synthetic section

Videos 03 and 04 each included a Synthetic tab section. This video
skips Synthetic because Q12 — "is there a classification-specific
synthetic data generation path?" — remains unresolved. The Synthetic
tab does render for this sample but the surface looks generic; the
recording would either invent a story or have to explain a caveat
that distracts from the class-balance / mobile-target beats.
Compression and Export get the freed time instead.

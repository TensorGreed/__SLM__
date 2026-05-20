# PII Detector Pipeline — Narration

Status: **synced** with the actual narrated take produced by
`tts/generate_v04_narration.py` (Orpheus voice "leo") on 2026-05-20.

The **Python script** at
[tts/generate_v04_narration.py](../../../tts/generate_v04_narration.py)
is the **authoritative source** of the spoken text. This file mirrors
the same text plus stage directions / Playwright cues. Edit the
script first.

Total runtime: **2:26** (matches
`docs-demo/recordings/raw/04-pii-detector-pipeline-narrated.mp4`).
Section timings come from `tts/audio/v04-durations.json`.

Companion to:
[docs-demo/videos/04-pii-detector-pipeline/recording-plan.md](../../videos/04-pii-detector-pipeline/recording-plan.md).

The key narrative beat is Section 3 (Cleaning) — the disambiguation
between cleaning-time PII redaction and the PII Detector model task.
That's the value of the entire video.

---

## Pre-roll (not narrated)

Playwright logs in as admin, clicks the **Demo · PII / PCI Detector**
tile, lands on the Data tab. ~5 seconds before narration starts.

## Section 1 — Cold open (0:00–0:19)

**On screen**: Data tab loaded; 61 raw documents listed below.

> "Welcome to the PII Detector pipeline walkthrough. This sample is
> a span-level entity detector — you feed it a snippet of text, and
> it emits a structured list of every personal-information span it
> finds. Email, phone, social security, credit card — ten entity
> types in total. There's a confusion risk we'll clear up in a
> minute."

## Section 2 — Data tab + expand a raw row (0:19–0:32)

**On screen**: click `[data-testid^="expand-doc-"]` on the first
row; the `text` and `entities_json` payload expand inline.

> "Data tab. Sixty-one source rows. Each row has two columns: the
> text, and a structured list of every entity in that text. The
> entity list is the ground truth — that's the shape the model has
> to learn to produce. Expand a row to see what one looks like."

## Section 3 — Cleaning tab (disambiguation) (0:32–0:53)

**On screen**: click the **Cleaning** tab; the redaction toggles
are visible.

> "Now the confusing part. The Cleaning tab has a personal-
> information redaction option. That's a regex pre-processing step —
> it can mask personal information in source text before training.
> The PII Detector model in this sample is the opposite — it finds
> personal information and emits a structured list. Same word in
> the product name, two completely different features. We're not
> running the redaction here."

## Section 4 — Gold Set tab (0:53–1:04)

**On screen**: click **Gold Set**; "Entries 200" badge visible.

> "Gold Set. Two hundred entries. Each one has a snippet, an
> expected entity list, and a rationale. The eval handler scores
> the model's predicted entities against the gold entities, per
> entity type."

## Section 5 — Synthetic tab (1:04–1:25)

**On screen**: click **Synthetic**; span generation mode visible.

> "Synthetic. The lever to grow sixty-one source rows into two
> thousand training rows. For this sample, the generator runs in
> span mode — you'd list the ten entity types you care about, and
> the teacher model generates new text with matching entity
> annotations. Local Ollama is wired up, but we're not running
> generation here. It's runtime-heavy and lives in its own
> walkthrough."

## Section 6 — Dataset Prep tab (1:25–1:40)

**On screen**: click **Dataset Prep**; Schema Profile panel shows
the `span_set` scoring mode.

> "Dataset Prep. The schema profile shows the scoring mode — span
> set. That's the contract with eval: the model has to emit a
> structured output with an entities array, and eval scores per
> entity type. Splits are forty-five train, eight validation, eight
> test."

## Section 7 — Tokenization tab (1:40–1:49)

**On screen**: click **Tokenization**.

> "Tokenization. Same idea as the previous sample. Reports per-row
> token counts and the maximum sequence length you'd budget for.
> Surface only for this video."

## Section 8 — Training tab → Training Config (Advanced) (1:49–2:09)

**On screen**: click **Training** → click **Open Training Config →**
button → land on `/project/<id>/training-config` → click **Advanced**
on the config-mode switch.

> "Training tab — empty, expected. Into the Training Config page.
> Defaults to Essentials. Flip to Advanced. For span extraction
> tasks the docs recommend bumping low-rank adaptation from rank
> eight to rank sixteen, and targeting all four attention
> projections instead of two. The Advanced view exposes those
> controls. Defaults work if you're starting out."

## Section 9 — Evaluation tab + wrap (2:09–2:26)

**On screen**: navigate back to `/project/<id>/pipeline/eval`;
"No experiments to evaluate" empty state.

> "Evaluation tab. Empty until we have a finished experiment. For
> this sample the eval handler scores per entity type — precision
> and recall for email, phone, social security, and the other
> seven. Next video: the sentiment classifier sample. Different
> task profile, different scoring mode."

---

## Things to **not** say

- Don't conflate cleaning-time PII redaction with the detector
  model task — Section 3 exists to prevent this.
- Don't say "the demo has 60 snippets" — that's the stale manifest
  prose. The CSV has 61 rows; say 61 (open Q1).
- Don't click the Synthetic Generate button. Even though Ollama is
  configured, running span generation isn't in scope for this
  inspection video; it has its own walkthrough.
- Don't read literal tech tokens aloud (env var names, REST paths,
  field names like `scoring_mode=span_set` — say "span set" as
  plain English).

## Optional technical notes (background; not spoken)

- `output_schema.scoring_mode = span_set` is the prepared manifest's
  distinguishing field — that's what dispatches the eval handler
  to span-level precision/recall scoring.
- Recommended PEFT settings for span tasks (from
  [slm-docs/docs/demos/pii-detector.md](../../../slm-docs/docs/demos/pii-detector.md)):
  LoRA rank 16, target modules `q_proj, k_proj, v_proj, o_proj`.
  Spoken in plain English as "rank sixteen, all four attention
  projections."
- All 10 entity types: email, phone, ssn, credit_card, person_name,
  street_address, date_of_birth, ip_address, api_key, bank_account.
- Gold counts skew toward person_name (138) and email (72); api_key
  is rarest at 21 rows. Eval weights per-class recall, so the rare
  classes still get measured.

## Why no "missing teacher key" warning

The original Video 04 plan pointed at a `WARN: missing
TEACHER_MODEL_API_KEY` banner on the Synthetic tab. As of the
2026-05-19 runtime decision, `backend/.env` now wires a real Ollama
teacher; the banner no longer renders. Narration says "Local Ollama
is wired up" instead.

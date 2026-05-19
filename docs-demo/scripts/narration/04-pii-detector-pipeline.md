# PII Detector Pipeline — Narration Skeleton

Status: ready for the inspect-only path. The #1 narration risk for
this sample is **conflating cleaning-time PII redaction with the
PII Detector model task** — they share the name "PII" but they are
completely different features.

Target length: 10–12 minutes (≈1600 words).

Companion to:
`docs-demo/videos/04-pii-detector-pipeline/recording-plan.md`.

---

## Cold open (0:00–0:40)

> "This is the PII Detector pipeline walkthrough. Before we click
> anything, one thing you should know up front: the word 'PII' shows
> up twice in this product, and they're completely different
> features.
> 
> First, in the Cleaning tab, there's regex-based PII redaction. That
> means: scan source text for things that *look* like email addresses
> or SSNs, and replace them with `[REDACTED_EMAIL]`. It's a
> data-cleaning step, not a model task.
> 
> Second, this sample — the PII Detector — is a *model* task. The
> goal is to train a small model that, given text, emits a structured
> JSON list of entities with their types and character offsets. That
> output can drive a redaction pipeline, a LlamaFirewall scanner, or
> a compliance audit log.
> 
> Same word, two completely different jobs. We'll touch both in this
> video, and I'll call out the difference whenever we switch context."

## Section 1 — Seed + Data tab (0:40–2:30)

**Action**: click PII / PCI Detector tile → land on Data tab.

> "I'll click the PII / PCI Detector tile. Same seeder behavior as
> Support FAQ — 61 source rows imported, 200 gold rows, 45-8-8 split
> pre-written.
> 
> Now look at the Data tab. We have 61 documents. Let me expand
> one…"

**Action**: expand `[data-testid="expand-doc-61"]`.

> "…and you can see the row has *two* columns: a `text` field with
> the raw sentence, and an `entities_json` field with the structured
> ground truth. Each entity carries a type, character start and end
> offsets, and the matched text.
> 
> This is the contract: the model has to learn to emit this exact
> shape. We have ten entity types in this sample — email, phone,
> SSN, credit card, person name, street address, date of birth, IP
> address, API key, bank account."

## Section 2 — Cleaning tab (the disambiguation moment) (2:30–4:00)

**Action**: switch to Cleaning.

> "Here it is. The Cleaning tab. Watch the PII redaction toggle.
> 
> If we turned this on and ran a cleaning batch, the cleaning service
> would run regex patterns over every raw document and replace
> detected patterns with placeholder tags. That's useful when your
> source data shouldn't leave your perimeter unredacted — say you're
> training on customer messages and want to scrub email addresses
> from the text *before* the model ever sees them.
> 
> This is **not** the PII Detector. The detector is a downstream
> model task we're going to train. The cleaning redaction is an
> upstream data-cleaning option. Two different surfaces, two
> different jobs.
> 
> One more nuance: if we did run the cleaning redaction, the
> resulting cleaned text would *lose* the entity information,
> because the entities would be replaced with `[REDACTED_*]` tags.
> That defeats the detector model's purpose. So when training a
> detector, don't redact at cleaning time — you'd lose your training
> signal."

## Section 3 — Gold Set tab (4:00–4:45)

**Action**: switch to Gold Set.

> "Gold set. 200 entries. Locked. Same shape as the raw rows but
> hand-verified. Per the evidence map, this covers all ten entity
> types — person_name leads at 138 instances, api_key has the fewest
> at 21. That distribution matches what you'd expect for real-world
> PII: lots of names, few API keys."

## Section 4 — Synthetic span generation (4:45–6:30)

**Action**: switch to Synthetic.

> "Synthetic generation. The mode is already set to span_extraction
> because the seeded manifest carries the `span_set` scoring mode.
> The synthetic generator will produce rows in the exact `{text,
> entities: [...]}` shape the eval scores against.
> 
> The recommended teacher for span tasks is `qwen2.5:7b-instruct-q4_K_M`
> or the 14B variant locally on Ollama. Qwen2.5 was trained on
> structured output and emits cleaner JSON than llama3 for this kind
> of task. The recommendation comes from the operator docs at
> `slm-docs/docs/demos/pii-detector.md`.
> 
> See the warning banner — `TEACHER_MODEL_API_KEY` is missing. We
> won't click Generate today. The detailed synthetic walkthrough is
> in Video 04 of the series."

## Section 5 — Dataset Prep schema (6:30–7:45)

**Action**: switch to Dataset Prep, open Schema Profile.

> "Dataset Prep. This is the headline panel for this sample. Look at
> the prepared-manifest summary — specifically the output schema.
> 
> `scoring_mode: span_set`. That's the contract. The eval handler
> dispatches on this value and runs entity-level matching: for every
> predicted entity, look for a gold entity with the same type, start,
> and end. Exact-match by default. Off-by-one boundary errors count
> as a miss plus a hallucination.
> 
> Strict matching is the right contract for redaction: a 'John'
> prediction for a gold 'John Smith Jr.' span breaks redaction just
> as badly as missing the span entirely."

## Section 6 — Training Config + LoRA recommendations (7:45–9:15)

**Action**: Training Config → Essentials toggle → Advanced → Power
Tools.

> "Training Config. By default we're in Essentials mode, with just
> the launch-critical controls. Flip the toggle in the page header to
> Advanced…"

**Action**: flip toggle.

> "…and the Power Tools tab opens. This is where the LoRA controls
> live. Defaults are rank 8, alpha 16, target modules `q_proj, v_proj`.
> 
> The pii-detector operator docs recommend bumping these for span
> tasks: rank 16, alpha 32, and all four attention projections —
> `q_proj, k_proj, v_proj, o_proj`. Roughly doubles training time
> but typically lifts span-task F1 by 5–15 points. The recommendation
> is in `slm-docs/docs/demos/pii-detector.md` under 'Improving F1
> after the first training run.'"

## Section 7 — Eval surface (9:15–10:00)

**Action**: switch to Evaluation tab.

> "Evaluation. Empty for now. When we have an experiment, this
> surface emits the per-class breakdown — for the PII task that
> means precision, recall, and F1 broken out by every entity type.
> Compliance teams care way more about per-class recall than overall
> F1: a 90% F1 model that misses every SSN is unshippable. The
> per-class breakdown is the load-bearing report."

## Wrap (10:00–10:45)

> "That's the PII Detector pipeline. Three things to remember:
> 
> One — cleaning redaction and the detector model are completely
> different features that happen to share the word 'PII'. Don't
> conflate them.
> 
> Two — the output schema is the contract. `span_set` mode, strict
> entity matching, per-class metrics.
> 
> Three — for span tasks, bump LoRA rank to 16 and target all four
> attention projections. Default rank 8 is conservative; span tasks
> like rank 16.
> 
> Next: the sentiment classifier in Video 07. Then we get into
> training runs in Video 09."

---

## Things to **not** say

- Don't say "the model masks PII" — it detects, it doesn't mask. The
  output of the detector is *structured JSON about PII locations*;
  a downstream consumer (LlamaFirewall, your redaction pipeline)
  decides what to do.
- Don't say "the manifest has 60 source rows" — manifest prose is
  stale; the file has 61.
- Don't say cleaning redaction is "automatic" — it's a configurable
  option behind a toggle.

## Optional advanced notes

- The PII demo bundle includes two helper scripts (`_generate_bundle.py`
  and `kaggle_pii_to_brewslm.py`) for developers extending the
  sample. They are *not* part of this video — author appendix only.
- The structured-extraction eval handler is in
  `backend/app/services/eval_task_handler_service.py`; the span_set
  scoring is dispatched from `output_schema.scoring_mode`.
- The synthetic generator for span mode lives in
  `backend/app/services/synthetic_service.py:generate_span_extraction_rows`
  with batched-async support via Story 1.7's
  `start_span_generation_task`.

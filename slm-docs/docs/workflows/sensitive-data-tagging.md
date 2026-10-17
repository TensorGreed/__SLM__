---
title: "Sensitive-data tagging from zero"
sidebar_label: "Sensitive-data tagging"
description: "From no data to a Hugging Face model that finds, classifies and locates PCI / GDPR / HIPAA and custom sensitive data in text — every step in BrewSLM."
---

# Sensitive-data tagging from zero

You want a model that takes a text and returns every piece of sensitive
data in it — **what** it is (a card number, a date of birth, a medical
record number, your own custom types) and **exactly where** it is
(character start and end) — and you have **no labelled data**, only a
local Ollama and, optionally, an Anthropic key. This page is the whole
path through BrewSLM. Nothing here needs a script; every step is a card
or a button, and each one is a step any documents-first project can use.

The shape of the task in BrewSLM is **span extraction** (`span-extraction`
recipe, `structured_extraction` task profile, `span_set` scoring): a row
is `{text, entities: [{type, start, end, text}]}`. The two things this
page adds to that recipe are a way to get *documents* when you have none,
and a way to train for *exact offsets*.

## 1. A project with no documents → "Write sample documents"

Create a project (any name; the brief can say "detect sensitive data in
support tickets and clinical notes"). On **Data** → **Sources** the first card,
*No documents yet? Write sample documents*, asks a generation model —
your local Ollama model first; a cloud model only if you pin one under
Synthetic → backends — to write realistic, **invented** documents:

- *What the documents must contain*: the kinds of data, in your words —
  "card numbers (Visa / Mastercard / Amex formats), expiry dates, CVV,
  IBANs; names, emails, phone numbers, home addresses, dates of birth,
  national ids, passport numbers, IP addresses; medical record numbers,
  health-plan member ids, diagnoses, medications, provider names;
  employee ids, API keys, internal ticket ids".
- *Genres*, one per line: support tickets, billing emails, clinical
  visit notes, insurance claims, HR onboarding emails, data-subject
  access requests, incident reports, bank chat transcripts, travel
  bookings, pharmacy messages — the places that data shows up.
- *Documents* and *words each* (defaults 60 × 160).

It runs as a Job the bell tracks ("Writing document 12/60: clinical visit
note (ollama:gemma4:12b)"). The documents are ingested as **one text
file** (`seed-documents-<time>.txt`, source `seed_documents`) and cleaned
into passages, so everything after this is identical to a project you
uploaded documents into. Every person, number and identifier in them is
made up; the model is told so, and the file is marked by its source.

## 2. Label them → the span synthesiser

On **Data** → **Generate**, with the project's task shape set to span
extraction (the `span-extraction` recipe — pick it from the recipe picker
or let the task-shape card confirm it), the span generator takes your
passages and a list of **entity types** — the canonical ones plus your
custom ones, e.g. `credit_card, card_expiry, cvv, iban, bank_account,
person_name, email, phone, address, date_of_birth, national_id, passport,
ip_address, medical_record_number, health_plan_id, diagnosis, medication,
provider_name, employee_id, api_key, ticket_id` — and asks the teacher
model to rewrite each passage with *different* synthetic values and
return every entity with its character offsets. Each row is validated
before it is kept: `text[start:end]` must equal the entity's text, types
outside your list are dropped, and the built-in regexes add any obvious
email / phone / card / IP the teacher missed. Rows land in the review
queue (source `span_generation`), grouped by source; accept them in bulk
or reject the ones you don't like.

The teacher is an OpenAI-compatible endpoint: Ollama's
`http://localhost:11434/v1` with any local model is free; an Anthropic or
OpenAI model costs per call. A reasonable split is: **local model for the
bulk** (hundreds of rows), **a cloud model for the answer key** (the
held-out rows the evaluation scores against — a few hundred calls with a
small model is cents), because the key's quality bounds what the
evaluation can tell you.

## 3. Split, then train for exact offsets

**Data prep → split** as usual (the default 80 / 10 / 10; the test
examples are what the evaluation scores). Then on **Training**, set the
*task type* to **Token classification (spans with exact offsets)** and a
base model that is an *encoder* — `microsoft/deberta-v3-base` is the
default choice (184M parameters, fast tokenizer, strong on NER);
`bert-base-cased` works too.

What this mode does differently from the generative span recipe: instead
of a language model *writing* the entities as JSON (and counting
characters wrong), it trains a token-classification head on BIO labels
derived from each row's character spans, so at inference the offsets come
from the tokenizer's own offset mapping — exact by construction. The run
writes a plain Hugging Face model (`model/` with the tokenizer and a
`label_space.json`), per-epoch **span precision / recall / F1** (strict
match on type + start + end, the same match the `span_set` scorer uses),
and a per-type breakdown in `training_report.json`. Epochs auto-size to
the row count (≈1500 example-passes, 3–20 epochs) unless you set them.

## 4. Evaluate → refine → re-train

**Evaluate this run** on the Results page (it also runs by itself after
training) scores a token-classification run *on its own* — there is no
base-model comparison, because its base is an encoder with an untrained
head. The card reads "Span F1 0.85 against the answer key": the answer
key written by a second model first (with who wrote it), the test
examples second (labelled by the training-data teacher), the weakest
types, and the failing texts; **Re-evaluate** re-scores both. The
held-out evaluation recognises a token-classification checkpoint and
tags the test rows with it (no generation), so the **span_set** report is
the usual one: precision / recall / F1 overall, **per type** (where the
model is weak), and per row the entities *matched*, *missed* and
*hallucinated* — with the model's output beside the key. Weak types are
the refine loop, and it is one button: **Refine weak types** on the
**Data** → **Generate** step (span mode) reads the latest span evaluation, names every
type under F1 0.8 with its score ("Weak on run #55: passport (0.00),
medication (0.60), …"), and starts the span generator restricted to
those types, sampling from all passages, with a row count sized to how
many types are weak (`GET /synthetic/weak-types?threshold=`). Accept the
rows, re-split, re-train; the Eval summary compares the runs.

### An answer key the teacher didn't write

Every number above scores the model against labels the *same* local
teacher wrote. **Write answer key with this model** (**Data** → **Generate** step, span
mode) hands the test split's texts — unchanged — to the provider and
model selected in the panel (a cloud pin such as
`cloud:anthropic:claude-haiku-4-5-20251001` with the project's saved key,
or another local model), asks only for the entities, verifies every
offset from the entity text, and keeps the result as the project's
**gold_test** set (`POST /synthetic/relabel-answer-key`, Job
`span_relabel`). The bell reports rows, entities and the **agreement
between the old labels and the new key** (strict span F1, per type) —
the share of "error" that was ever the student's. Then evaluate the run
against `gold_test` on the Results page: the per-type scores are now judged
by a second opinion. Cost: one call per six texts; a 44-row test split
is eight Haiku calls.

What it said on project 31 (run #56, 2026-10-14): Haiku 4.5 labelled
all 44 test texts in 45 s (283 entities; the first attempt lost two
batches to replies cut mid-JSON — the generator now asks for compact
type + text only and splits a failed batch), and the two labellers
**agree at F1 0.74** — 193 spans in common, 48 only in the local
teacher's labels, 90 only in Haiku's. Haiku's key is denser: 14 phones,
12 emails and 13 names the local labels don't have, and 11 of the local
`date_of_birth` spans it reads as another kind of date. Scored against
it the run reads **P 0.77 / R 0.71 / F1 0.74** (was 0.85 / 0.92 / 0.88
against the local key): recall falls on exactly the types the local
teacher under-labelled (phone R 0.44, email R 0.59, person_name R
0.72), and the types both labellers agree on stay perfect (IBAN, IP
address, MRN, ticket id; CVV 0.95, card expiry 0.93). About a third of
the "error" the first key measured was the key, not the model.

**Why the local labels lacked them — and the fix.** Every one of the 26
phones and emails the Haiku key had and the local labels didn't matches
BrewSLM's own regex. They were missed because the *refine* passes
restricted the entity-type list to the weak types, and both the teacher
("use ONLY these") and the regex merge honoured it — so refine rows
carried unlabelled phones, emails and names, and after the re-split those
rows taught the model that such spans are nothing. Two changes: a refine
now passes **focus types** (what each snippet must contain) separately
from the **entity-type list** (the project's whole vocabulary, every type
of which is labelled — the Refine button does this for you), and
**Regex pass over saved rows** (`POST /synthetic/spans/regex-augment`,
idempotent) repairs rows already generated under a restricted list: on
project 31 it added 438 entities (210 phones, 208 emails) to 254 of 874
rows. Retrained on the repaired labels (run #57, same 44 test texts), the
run reads **P 0.79 / R 0.79 / F1 0.79** against the Haiku key (was 0.74):
phone 0.61 → **0.96**, email 0.71 → **1.00**. What remains against the
second opinion is labelling *convention*, not missing labels —
`date_of_birth` 0.60 (Haiku reads the local teacher's DOBs as other
dates), `bank_account` 0.32, `national_id` 0.59, `provider_name` 0.52
(the two labellers disagree on those too), and names inside greetings
(person_name recall 0.72).

### Check the answer key yourself

Every key above was written by a model, so a score against it inherits
that model's mistakes — and the training labels came from the same
family, so the two can agree on an error. The **Data** → **Answer key** step (test set)
now carries **Check the answer key yourself**: the key's rows, one by one,
with the passage and its spans highlighted. Rows come **unreviewed
first**, and among those the ones where the project's latest trained run
**disagrees with the key most** (strict span F1 between the two — one of
them is wrong, and a person decides which); green = in the key, orange =
the model missed it, dashed = the model found something the key doesn't
have. Three verdicts per row: **Accept** (the labels stand), **Fix**
(edit the spans inline — select text in the passage to add one, or start
from the model's spans — the original is kept so a cleared verdict
restores it) and **Reject** (not a fair test: the row leaves every eval).
Verdicts live on the gold rows themselves (`review` block in
`gold/gold_test.jsonl`), so every reader scores the corrected labels, and a
rejected row never reaches the scorer.

The payoff is on the Results page: the answer-key line now says how many rows
a person checked and re-totals the run's score **on the checked rows
apart from the unchecked ones** — the checked score is the one that
cannot be the key's own mistake. A correction or rejection newer than the
eval flags it stale ("re-evaluate to score the corrected key"). The
exported model card carries the same line under the answer-key
evaluation ("Human review: 12 of 43 rows checked by a person (3
corrected, 1 rejected). On the checked rows: …"). API: `GET
/projects/{id}/gold/review` (queue; `?experiment_id=` picks the run to
compare with, `compare=false` skips the model), `PUT
/projects/{id}/gold/review` `{index, key, verdict, corrected?, note?}`
(`verdict: null` clears). The model's spans are cached per run in
`gold/review_predictions.json`, so reopening the queue is instant. The
model's spans are decoded **as served** — under the run's tuned
thresholds and recall bias, the same decoding the eval scores — so a
"model disagrees" row is a disagreement the score will count (plain
decoding showed spans the served model drops, which sent the first
version of this queue chasing a difference the eval could never see).

What it found on project 31 (2026-10-16, run #117, the 43-row clean key
written by Sonnet): the model disagreed with the key on **12 rows**, and
a person went through them. **Eleven were the model's mistakes** — a
transaction date tagged as `date_of_birth` (twice), a referral date as
`visit_date`, a login handle as `person_name`, `mild steroid cream` as a
`medication` (a class, not a drug), and five boundary differences on
drug names (`Tamiflu` vs `Tamiflu, dosage 75mg`) where the key's reading
stands. **One was the key's**: row 32 labels `4242424242424242` a
`bank_account` where the convention says a card number is `credit_card`
— corrected. Three undisputed rows were accepted on the way (15 verdicts
in all). Re-evaluated, run #117 reads **F1 0.905 on the 15 checked rows
and 1.00 on the 28 unchecked**, overall 0.968 — unchanged by the
correction, because the served model doesn't tag that card number
either (the error only moved from a false positive to a miss). Read the
split for what it is: the checked rows were *chosen* because the model
disagreed there, so the unchecked 1.00 is the agreement the queue
started from, not a random-sample estimate; the number to trust is the
0.905 on rows a person signed off, and the finding is that the key was
right 11 times out of 12 — at least on the rows a different model could
flag. Rows where both models are wrong the same way still pass unseen,
which is what the independent benchmarks are for.

### Can the answer key tell two runs apart?

A per-type F1 on a handful of reference spans moves by tenths between
seeds, so "credit_card went 0.57 → 0.74" means little unless the key has
the spans to back it. The span Eval card now sizes each type the way the
documents→Q&A flow sizes its test split: for two runs scored on the
*same* reference spans, a gap Δ in a type's F1 clears a 95% interval when
the type has about `8·p·(1−p)/Δ²` reference spans (p = the type's F1 as
the share of spans a run gets right, clamped to 0.5–0.95 so a perfect
score on three spans claims nothing). At Δ = 0.05 that is **152 spans
for a type at F1 0.95, 512 at 0.8, 800 at 0.5**. The card's line
("*Can the answer key tell runs apart?*") says how many types have
enough, and the per-type table shows each type's support, what it needs,
the shortfall, and — the more useful number — the smallest difference
its current support **can** show (`2·sqrt(2·p·(1−p)/n)`). The same
sentence goes on the exported model card under each evaluation.

On project 31's 43-row key (2026-10-16, run #117): **none of the 22 types
can show a 0.05 difference**, and together they are 4,692 reference
spans short. The best-supported types can show about 0.10 (person_name,
37 spans; email 26; phone 25), medication 0.25 (20 spans), bank_account
0.56 (5 spans), and api_key — 2 spans — 0.94. So every per-type movement
reported in the rounds above sits inside what the key can see, except
where it was confirmed on the Gretel or CoNLL benchmarks (thousands of
spans); the overall F1, on 279 spans across all types, resolves about 0.04.

**Refine weak types** on the **Data** → **Generate** step now targets the under-sampled
types as well as the weak ones (`GET /synthetic/weak-types` →
`under_sampled`, `types_to_refine`): the generated rows must contain
them. Two things to know: they are *training* rows, and the project's
split keeps the test rows, so new rows only reach the answer key through
a re-split **without** "keep test" followed by a re-key — and at a 0.1
test share, closing a 500-span shortfall on one type means thousands of
rows containing it. The honest reading of the table is usually not
"generate more" but "stop reading per-type deltas smaller than the gap
it can show, and trust the benchmarks for those".

### One labelling convention for both labellers

Two models can both be right and still disagree — is a date in a
clinical note a date of birth or a visit date, is `GB29 NWBK …` a bank
account or an IBAN, is "Dr. Chen" a person or a provider? **Labelling
conventions** (**Data** → **Generate** step, span mode) is one line per type — what
counts and what doesn't — stored on the project
(`GET/PUT /synthetic/type-definitions`, with suggested starting wording
for common type names). Both the model that writes the training labels
and the model that writes the answer key get the lines for the types
they label, so they draw the boundaries in the same place. **Relabel
training rows with *model*** (`POST /synthetic/relabel-answer-key`
with `target: "training"`) rewrites the saved rows' labels in place
under the conventions — texts unchanged, then the regex pass — so rows
generated before the conventions existed follow them too.

What the conventions did on project 31 (2026-10-14), measured in three
steps on the same 44 test texts:

| step | result |
|---|---|
| suggested wording accepted for all 23 types | one line each; both labellers get the lines for the types they label |
| Haiku re-keys the test split under them | 44/44 in 60 s, 288 entities. Run #57 (trained on the *old* labels) scores the same against it — F1 0.79 — so the gap is in the training labels, not the key |
| local teacher relabels the 874 training rows under them | 39 min on gemma4, 0 failures; old vs new training labels agree at F1 0.82 — the conventions moved about one label in six |
| labeller agreement, local vs Haiku | **0.74 → 0.85** (provider_name 0.93, national_id 0.92, visit_date 0.89, date_of_birth 0.78) |
| re-train → run #58, scored against the Haiku key | **P 0.83 / R 0.85 / F1 0.84** (run #57: 0.79). date_of_birth 0.52 → **0.78**, national_id 0.69 → **0.92**, visit_date 0.69 → **0.87**, medication 0.70 → **0.88**, diagnosis 0.74 → **0.96**, provider_name 0.80 → **0.85**; phone and email 1.00 |

Two types didn't move, and both are about the span's *edges*, not its
type: all 12 remaining person_name disagreements are a title one
labeller includes and the other doesn't ("Ms. Olivia Reed" vs "Olivia
Reed"), and bank_account (0.21) splits "Account XXXX-3344, routing
number 99001122" into one span or two. The suggested wording now says
where those edges are (a person's name without the title; each account
or routing number its own span, digits only) — a definition has to fix
the boundaries, not just the category. Run #58 is export 7.

### Compliance bundles: start from a regime, score per regime

A compliance bundle is a starting set of entity types for one regime,
each with a one-line labelling definition. Four ship built in —
**PCI-DSS** (credit_card, card_expiry, cvv, bank_account, iban),
**GDPR personal data** (person_name, email, phone, address,
date_of_birth, national_id, passport, ip_address, device_id,
geolocation), **HIPAA PHI** (medical_record_number, health_plan_id,
diagnosis, medication, provider_name, visit_date plus the identifiers
HIPAA counts) and **CCPA personal information**. They are plain data
(`compliance_bundles_service.BUILTIN_BUNDLES`), not special handlers:
any type can carry any tag.

**Compliance bundles** (**Data** → **Generate** step, span mode, next to *Labelling
conventions*) — tick bundles, **Apply bundles**:

- the bundle's types are added to the *Entity types* list;
- each type you haven't defined yet gets the bundle's wording as its
  labelling convention — a definition you wrote is **never**
  overwritten (the toast says how many it kept);
- each type is tagged with the regime. A type in two bundles
  (person_name is GDPR, HIPAA and CCPA) carries all their tags.
  Unticking a bundle removes its tag; the definitions stay.

Below the bundles, the *regime tags* box is one line per type —
`credit_card: pci_dss, ccpa` — and accepts your own tags
(`employee_id: internal_tier_1`), so you can group types your own way.
State lives in `runtime_config["compliance"]`
(`{bundles, custom, type_regimes}`). API: `GET
/projects/{id}/compliance/bundles`, `POST .../compliance/apply`
`{bundle_ids, remove_bundle_ids}`, `PUT .../compliance/type-regimes`
`{type_regimes}`.

**By regime** — once types are tagged, the span evaluation (Results page,
and the Job result after training) adds a table: per regime, micro
precision / **recall** / F1 summed from each type's true positives,
false positives and misses, weakest recall first, with "*N* of *M*
missed". Recall is the column to read for redaction: every missed span
is data that leaked. A regime type the evaluation never saw is listed
as "not in this eval" rather than counted as perfect.

### Your own identifier types (custom patterns)

The built-in regex pass knows the well-known formats — email, phone,
SSN, card, IP, API key, ISO / US dates. Your organisation's own
identifiers (an employee id like `EMP-48213`, a ticket reference, an
internal account format) are just as regular, and a regex catches them
with exact offsets where a labelling model may miss one or misplace its
edges. **Add a type** (**Data** → **Generate** step, span mode, under the labelling
conventions) does it in one place:

1. a **name** (normalised like the conventions: lower-case, spaces → `_`),
2. a **one-line definition** — saved with the labelling conventions, so
   every labelling model gets it,
3. an optional **pattern** (a Python regex, optional *ignore case*) —
   **Test pattern** shows how many matches it finds in your saved span
   rows and cleaned document passages, with the first ten in context
   and their character offsets, before anything is saved,

and adds the type to the entity-type list. Saved patterns are listed
under the form with a Remove button.

From then on **every regex pass** for the project uses the built-ins
**plus** your patterns, under the same entity-type filter: span
generation (the teacher's prompt also names your types among the ones
the regex pass catches), **Regex pass over saved rows**, and relabelling
training rows / labelling passages / labelling whole documents. A custom
pattern adds matches for its type and overrides nothing — a pattern for
`phone` in a local format catches those numbers *as well as* the
built-in ones. Run the regex pass over saved rows after adding a type to
label rows saved before it existed (idempotent). The second-model answer
key is left alone: it stays the other model's independent reading.

Patterns are checked when saved (`PUT /synthetic/type-patterns`; nothing
is saved if any entry fails, the error names the entry and why):

- it must compile and be at most 300 characters;
- it must not match the empty string (`\d*`, a bare lookahead) — it
  would tag zero-length spans everywhere;
- no **catastrophic backtracking**: an unbounded repeat nested in
  another (`(a+)+`, `(\w+\s?)*`) is rejected outright, and the rest is
  timed over ~2,000-character repetitive strings in a separate process
  that is killed after 3 s — so `(a|a)+b` is refused instead of hanging
  the server. Rewrite such a pattern without the nesting
  (`\bEMP-\d{5}\b`, not `(EMP-(\d+)+)`).

API: `GET /synthetic/type-patterns` (patterns, any stored entry that no
longer validates, the built-in types, the limits),
`PUT /synthetic/type-patterns` `{patterns: [{type, pattern, description,
flags}]}` (flags: letters `i` / `m` / `s`), and
`POST /synthetic/type-patterns/test` `{pattern, type, flags}` →
`{valid, error, texts_scanned, texts_matched, count, matches: [{source,
text_index, start, end, match, before, after}]}`. The patterns live in
the project's `runtime_config["span_custom_patterns"]`.

## 5. Export → Hugging Face

**Export → Hugging Face** packages the run's `model/` directory as-is
(there is no adapter to merge). Use it anywhere Transformers runs:

```python
from transformers import pipeline

tagger = pipeline("token-classification", model="path/or/hub-id", aggregation_strategy="simple", stride=128)
text = "Charge card 4111 1111 1111 1111 for Ada (ada@example.test) by 09/27."
for ent in tagger(text):
    start, end = ent["start"], ent["end"]
    while start < end and text[start].isspace():  # DeBERTa's tokenizer counts the leading space
        start += 1
    print(ent["entity_group"], start, end, text[start:end], round(ent["score"], 3))
```

`start` / `end` are character offsets into the string you passed; the
`entity_group` is your entity type, custom ones included. DeBERTa-v3's
tokenizer attaches the space before a word to the word, so the pipeline's
`start` can sit one character early — the two-line trim above (which
BrewSLM's own evaluation applies) gives the exact span.

### What's in the exported `model/`

Besides the weights and tokenizer, a span export writes two files:

- **`README.md` — the model card.** Built only from what BrewSLM
  recorded for the run: base model, entity types (from
  `label_space.json`), training rows and where they came from (the run's
  `training_report.json` and the prepared train file it read; the source
  tally is left out if the split has changed since), and per-type
  precision / recall / F1 from the run's latest span evaluations —
  **answer key first** (with the model that wrote it), test examples
  second. It carries the usage snippet above with the right `stride`
  for the run's window, and a limitations list that says how many texts
  the scores rest on, which types have too few reference spans to trust,
  whether there is an independent answer key, and whether the training
  data is synthetic. A section with no data (no evaluation yet, no
  training report) is left out — the card never fills a gap with a
  number. Uploading the folder to the Hub publishes it as the model page.
- **`thresholds.json`** — per-type minimum span scores and a
  `recall_bias`, all `0.0` by default (= exactly the argmax spans).
  BrewSLM's redact endpoint reads it; edit it to tune the shipped
  behaviour.

### Confidence and high-recall mode

`predict_spans` (the decoder evaluation uses) can also score each span:
`score` is the mean over the span's tokens of P(B-type) + P(I-type),
`min_score` the weakest token. Two knobs trade precision for recall:

- **Per-type thresholds** drop spans of one type below a score — e.g.
  `{"person_name": 0.8}` keeps only confident names and leaves every
  other type alone.
- **Recall bias** (0–1) is subtracted from each token's "O" probability
  before the label is picked, so a token the model thought *probably not*
  an entity, but only by a small margin, gets tagged. For redaction a
  missed account number costs more than a masked word, so a bias of
  0.1–0.3 is the usual "high-recall mode". Scores stay unbiased, so a
  borderline span shows a low score.

With no thresholds and bias 0 the output is identical to the argmax
decoder (evaluation is unchanged).

### Tuning thresholds, with recall floors

**Tune thresholds** on the span Eval card picks the thresholds for you on
the **validation split**. It never uses the test examples or a benchmark,
because those are what the thresholds are then measured on. The split is
run through the model once. Every threshold and recall bias after that
is a cheap re-decode of the same scores. The result goes into the run's
`thresholds.json`, which evaluation, the redact endpoint and the export
all read. Re-evaluate afterwards to see the test score under the new
thresholds.

Without floors, each type gets the threshold that maximises its F1, and
it stays at 0 unless the gain is real. For redaction that is the wrong
target: a missed account number is the leak, while an extra masked word
is only noise. So you can give a type a **recall floor**, which is the
minimum validation recall it must keep. Set floors in the tune panel,
per type plus an optional default for every other type, and they are
saved on the project (`GET/PUT /api/projects/{id}/span-recall-floors`).
A floored type is not tuned for F1. Instead it gets the
**best-precision threshold whose recall still meets the floor**. When
two thresholds tie, the higher one wins.

Sometimes even threshold 0, which keeps every predicted span, misses the
floor. Thresholds only remove spans, so they cannot fix that. The panel
then says **not reachable at threshold 0**. It also tries recall biases
(0.05 to 0.5) on the same scores and reports the smallest one that meets
the floor, with its precision cost for that type and overall. The bias
is global, so every type gets more spans. That is why it is only written
to `thresholds.json` when you tick *apply the suggested recall bias*. The
thresholds are then re-tuned under that bias.

If you applied a compliance bundle (PCI-DSS, HIPAA: 0.95; GDPR, CCPA:
0.9), the panel shows each regime's floor beside the types it covers.
**Use compliance suggestions** copies them into the editor. Nothing is
set until you save or tune with them.

`thresholds.json` records the floors (`recall_floors`), the chosen
`per_type` thresholds and `recall_bias`, and under `tuned.report` each
type's recall and precision before → after along with its
`floor_status` (`met` / `not_reachable` / `no_support`). The exported
model card has a **Recall floors** table that shows whether each floor
was met on validation.

**A recall-floor tune, measured on the benchmarks (2026-10-16, run
#117).** Floors of 0.9 on `person_name` and the compliance bundles' floors
on everything else (PCI / HIPAA 0.95, GDPR / CCPA 0.9), with "apply the
suggested recall bias" on. On the 156-row validation split nine floors
were already met at threshold 0 and **ten were not reachable at any
threshold or any bias in the grid** (address, bank_account, credit_card,
diagnosis, health_plan_id, iban, medication, passport, phone, visit_date
— their validation recall tops out at 0.82–0.94), so the tuner applied
**no** recall bias and, for the floored types, dropped the per-type
thresholds to 0 (validation micro F1 0.934 → 0.939, precision up on the
eight types that kept a threshold). On the benchmarks that is a **loss**:
CoNLL people unchanged (P 0.88 / R 0.59 — `person_name` never had a
threshold), Gretel **F1 0.708 → 0.668, precision 0.680 → 0.615, recall
0.740 → 0.732**. Recall rose by at most 0.02 on any type (credit_card
0.81 → 0.83, phone 0.85 → 0.87) while precision fell by 0.2–0.3 on the
same types (credit_card 0.67 → 0.40, phone 0.77 → 0.56, bank_account
0.40 → 0.20), and medical_record_number lost recall (0.56 → 0.48) to its
raised threshold. So no: recall did not rise more than precision fell —
the missed spans on real text are not borderline tokens a lower bar
would admit; the model doesn't see them at all. The validation-tuned
thresholds were restored and the benchmarks re-scored under them. The
lever for recall on these types is training text with those shapes, not
decoding.

**Identifier-shapes round, run as lanes (2026-10-16).** Forty seeded
documents (local model) asking for real-world address layouts, hospital
MRN formats, licence plates as they appear in notices, and bank account
numbers next to routing / reference numbers and same-shaped customer ids;
72 passages, 409 Sonnet labels ($0.24: address 84, MRN 69, plate 57,
bank_account 24, person_name 90). Re-split keeping the test rows, three
seeds of #117's config, thresholds tuned on validation, seed picked by
validation span F1 (#120, 0.923 vs 0.911 / 0.912). Scored on the
benchmarks — a **null round with a regression**, confirmed by a fresh
reviewer:

| run | Gretel F1 | CoNLL F1 (R) | plate | bank | address | MRN |
|---|---|---|---|---|---|---|
| #117 (shipped) | 0.708 | 0.708 (0.59) | 0.54 | 0.58 | 0.63 | 0.66 |
| #119 / #120 / #121 | 0.730 / 0.707 / 0.701 | 0.657 / 0.691 / 0.659 (0.52 / 0.57 / 0.54) | 0.59 / 0.57 / 0.57 | 0.42 / 0.28 / 0.35 | 0.64 / 0.63 / 0.63 | 0.69 / 0.67 / 0.66 |

Gretel overall moved inside seed noise and not in one direction; the
validation pick is flat (−0.001). CoNLL recall fell on all three seeds
(0.59 → 0.52–0.57): the new documents added 90 names in letter and form
contexts plus 27 clinicians (provider_name, deliberately *not*
person_name) and diluted the news share, pushing the model toward names
cued by a label word. Addresses and MRNs did not move (136 and 232
reference spans resolve ~0.12 and ~0.09). Plates rose ~0.04 on all three
seeds, but 55 spans cannot resolve that. bank_account got **worse the
way the catch-all always does**: recall stayed 1.0 while the prediction
count went from ~42 (#117) to 63–106 per seed — the seed asked for "the
same digit shapes used as customer ids or order numbers" but delivered
only 24 bank_account labels and no look-alike labels, so the model learned
a broader net, and employee_id precision fell with it (0.92 → 0.61–0.83).
#117 stays shipped; the 72 rows are accepted in `synthetic.jsonl`
(`passage_labels:doc=752`) and can be retired from the review queue.
Lesson: a 40-document mixed-type batch judged on 17- and 55-span
benchmark types cannot resolve the effect it is after; build a ≥ 300-span
reference for plates and accounts first, and seed negatives *with
labels* for the look-alikes, or stop.

**Look-alike round (2026-10-16).** Reading #120's Gretel false positives
against Gretel's own labels showed where the catch-all came from: 89
wrong bank_account spans were MRNs 20, card numbers 19, device
identifiers 15, SSNs 8, customer ids 7, routing numbers 4 — and most of
those kinds had **no project type**, so the model filed them under the
nearest digit-string type. The round gave every look-alike its own type
(device_id, customer_id, routing_number, transaction_id, username,
tax_id, licence_number — 30 types in all) with a definition that says
what it is *not*, seeded 40 documents where each kind carries its label
word and the same digit shapes recur across kinds, Sonnet-labelled them
(62 passages, 589 spans, $0.26; device_id 67, customer_id 55,
routing_number 40, username 35, transaction_id 93 — but license_plate 2,
address 2), relabelled the test rows and both keys under the grown
vocabulary ($0.25 + $0.13), and trained three seeds of #117's config.

| run | Gretel P / R / F1 | CoNLL R / F1 | credit_card F1 | MRN F1 | employee_id P | plate R |
|---|---|---|---|---|---|---|
| #117 shipped | 0.680 / 0.740 / 0.708 | 0.59 / 0.708 | 0.74 | 0.66 | 0.92 | 0.65 |
| #123 (validation pick) | 0.717 / 0.741 / 0.729 | 0.54 / 0.672 | 0.81 | 0.75 | 0.89 | 0.27 |
| #124 | 0.708 / 0.763 / 0.735 | 0.58 / 0.714 | 0.79 | 0.74 | 0.70 | 0.38 |
| #125 | 0.705 / 0.763 / 0.733 | 0.60 / 0.725 | 0.79 | 0.78 | 0.68 | 0.29 |

**Mixed, not a win** (fresh reviewer's verdict, which the numbers bear
out). The mechanism works where the look-alikes got labels: credit_card
precision 0.67 → 0.82–0.90 and MRN F1 0.66 → 0.74–0.78 on all three
seeds, Gretel overall +0.02–0.03 on all three. It fails where they
didn't: **license_plate** collapsed (recall 0.65 → 0.27–0.38; 26 of 55
plates now get nothing, 11 are read as national_id after a "License
Plate:" label) because the seed carried 2 plate labels against 16
`licence_number` labels defined as "not a vehicle plate"; **employee_id**
precision fell (0.92 → 0.68–0.89) as customer_id / username / device_id
absorbed ids Gretel calls employee ids. bank_account moved inside its
17-span noise (P 0.38–0.57). Of the criteria set before the round only
one seed met one of them, and the validation pick (#123) is the weakest
seed on CoNLL. **#117 stays shipped**; the 30-type vocabulary stays (the
credit_card / MRN gains show it works) but is unfinished: plates,
addresses and licence numbers are below the 20-label floor. Two product
gaps this exposed, both fixed the same day: the seed pick read the
*pre-tune* validation F1 (#125 was best after tuning) — it now picks on
the validation F1 under each seed's tuned thresholds when every seed was
tuned (`span_f1_tuned`); and rewriting the answer key dropped the 15
human verdicts — a relabel now carries them: corrections win over the
new labels, accepts carry when the labels are unchanged, rejects carry,
and an accept whose labels changed is kept on the row as
`previous_review` (the bell reports carried / dropped counts).

**Look-alike round 2: plates and staff-id contrasts (2026-10-16).** A
second seeded document (40 documents, 69 passages, 590 Sonnet spans,
$0.27) with at least two licence plates per document in the formats real
notices use (122 plate labels) and employee-id / customer-id / username
triples sharing letter-digit shapes (78 / 89 / 68), plus passports and
national ids as the dashed look-alikes. Three seeds of #117's config,
picked by **tuned** validation F1 this time (#127, 0.945 vs 0.943 / 0.944
— a near tie; the pre-tune pick would have been #129).

| run | Gretel P / R / F1 | CoNLL R / F1 | plate P / R | employee_id P | credit_card F1 | MRN F1 |
|---|---|---|---|---|---|---|
| #117 shipped | 0.680 / 0.740 / 0.708 | 0.59 / 0.708 | 0.46 / 0.65 | 0.92 | 0.74 | 0.66 |
| #127 (tuned pick) | 0.744 / 0.774 / 0.759 | 0.55 / 0.681 | 0.64 / 0.42 | 1.00 | 0.84 | 0.76 |
| #128 | 0.725 / 0.757 / 0.741 | 0.52 / 0.669 | 0.75 / 0.38 | 0.75 | 0.86 | 0.76 |
| #129 | 0.732 / 0.769 / 0.750 | 0.52 / 0.649 | 0.65 / 0.64 | 0.79 | 0.84 | 0.71 |

**Still mixed; #117 stays.** Gretel is now clearly better on all three
seeds (+0.03 to +0.05 F1, precision +0.05 to +0.06; credit_card +0.10 to
+0.13, MRN +0.05 to +0.10, national_id +0.06 to +0.15), employee_id
precision is partly back (0.75–1.00) and plate precision is up (0.46 →
0.64–0.75). But plate **recall** came back in only one seed (0.64; the
others 0.38–0.42 — after 122 labelled plates, so the remaining misses are
Gretel's plate formats, not volume), and **CoNLL recall fell for the
third round running** (0.59 → 0.52–0.55, all three seeds, below the 0.57
floor set before the round): each look-alike document adds ~100 names in
forms and notices, so bare headline capitals and "Name ( Country ) time"
results lines — the same CoNLL failures #117 has — get rarer in training.
For a PII tagger, missing 6% more people in prose is the worse failure,
so the trade is refused. The fresh reviewer's two notes: pick on per-type
validation stability rather than the macro mean when seeds tie (the
0.002 gap decided nothing, and #127 is the worst plate-recall seed); and
stop seeding plate / staff-id rounds — restore prose-name recall first
with a document of unlabelled-prose names (headlines, results tables,
bylines), then re-evaluate.

**Prose-names round (2026-10-16): the one that worked.** The look-alike
rounds had cost CoNLL person recall (0.59 → 0.52–0.55) because every
seeded document put names inside labelled fields. This round seeded 40
documents of names with **no label word** — all-caps headlines, wire
results tables ("5. Korneilus Hole ( Norway ) 23.87"), bylines and
datelines, surname-only quotes — with teams, companies and places as the
non-name contrast (73 passages, 483 person_name spans, $0.26). One
labelling error caught before training: Sonnet had tagged 36 dateline
dates as date_of_birth against the convention; removed. Three seeds of
#117's config, picked on tuned validation F1 (#131, 0.951 vs 0.944 /
0.947 — a spread inside seed noise, so the rule picks but cannot rank).

| run | Gretel P / R / F1 | CoNLL P / R / F1 | credit_card F1 | MRN F1 | national_id F1 | plate F1 | bank_account P |
|---|---|---|---|---|---|---|---|
| #117 (shipped until 2026-10-17) | 0.680 / 0.740 / 0.708 | 0.883 / 0.590 / 0.708 | 0.74 | 0.66 | 0.67 | 0.54 | 0.40 |
| #131 (tuned pick, shipped until the contrast round below) | 0.717 / 0.784 / 0.749 | 0.866 / 0.620 / 0.723 | 0.80 | 0.76 | 0.81 | 0.64 | 0.30 |
| #132 | 0.732 / 0.775 / 0.753 | 0.927 / 0.687 / 0.789 | 0.85 | 0.73 | 0.80 | 0.65 | 0.35 |
| #133 | 0.759 / 0.767 / 0.763 | 0.906 / 0.699 / 0.789 | 0.87 | 0.72 | 0.81 | 0.57 | 0.55 |

**Every criterion set before the round passes on all three seeds**:
CoNLL recall 0.62–0.70 (≥ 0.59), Gretel F1 0.749–0.763 (≥ 0.74), and the
look-alike gains kept (credit_card, MRN, national_id, phone, ip, email
all up three-for-three). The prose-names document restored exactly the
recall it was meant to, and undid nothing. Still unresolved, and not
measurable on their supports: bank_account (17 spans, precision 0.30–0.55
— #131 over-fires on account-like numbers), license_plate (55 spans,
0.57–0.65), health_plan_id (2), api_key (7). The fresh reviewer's call:
**replace #117 with #131** — Gretel +0.04, CoNLL recall +0.03, the
identifier types up, at the cost of CoNLL precision −0.017 and
bank_account precision on 17 spans. On the pick: #131 is the weakest of
the three on the benchmarks (CoNLL F1 0.723 vs 0.789 / 0.789), but
choosing #132 or #133 on that basis would be picking on the test set and
make every reported number optimistic; the honest ship is the rule's
pick, with the note that the rule's metric did not separate the seeds.

**Shipped (2026-10-17): #131, export 35.** One call,
`POST /api/projects/31/seed-groups/130/ship {"export_format":"huggingface"}`,
records the pick on the group's leader and runs the Hugging Face export of
the picked child. The first attempt (export 34) came out with the seed-
selection line and the benchmark table but **no Evaluation section and no
"Human review" line**: a seed child is never lift-checked by itself (the
group's leader is), so #131 had no span evaluation on record, and the model
card only states evidence that exists. *Evaluate this run* on #131 (the
lift-check Job: test examples + answer key, both 44 texts) and a second ship
produced export 35, whose card now reads, in order: *Seed selection: Picked
seed 45 (run #131) of 3 by validation span F1 (tuned thresholds) …*; the
answer-key evaluation with *Human review: 11 of 44 rows checked by a person
(1 corrected). On the checked rows: precision 0.90 · recall 0.94 · F1 0.92
(11 texts)*; and the External benchmarks table (CoNLL 0.87 / 0.62 / 0.72,
Gretel 0.72 / 0.78 / 0.75). Lesson for the product: ship after the lift
check, or the card is thinner than the model deserves. (Fixed the same day:
*Ship this seed* now runs the lift check on a seed with no evaluation first.)

**bank_account / employee_id contrast round (2026-10-17).** The loudest
remaining gap was bank_account precision 0.30 on Gretel (17 spans, recall
1.0: 16 right, 41 wrong). Measuring what #131 actually called a bank
account (`eval_tools/bench_confusions.py`, which lists a run's false
positives on a benchmark by what they really were) showed the pattern: 20
unlabelled ids (routing / ABA / SWIFT numbers, a tracking number, a payment
id, device identifiers, Unix timestamps, customer ids), 10 card numbers
with no "card" cue ("Payment: 3588…", "Shipper:", "Order ID:"), 8 MRNs. A
long bare number was a bank account by default — the training rows had 445
bank_account spans, a third of them in one "Account Numbers: A99999999999,"
shape, and only 44 routing numbers. employee_id (P 0.89 / R 0.66) missed
bare ids in Gretel's shapes (G-82727-I, Mr-44381, Q4896704) even after
"Employee ID:". So: one document of **contrasts, each with its own cue** —
40 statements, payroll forms, wire confirmations, HR / IT inventories,
incident reports, logistics notices, billing, loyalty letters, JSON / XML
logs, loan letters, rotas; every account next to a routing number, a card,
a tracking / order / device / batch / timestamp / customer / MRN id, and
staff ids in the Gretel shapes beside the same shapes as incident, audit,
part and flight ids. 67 passages, Sonnet-labelled (840 spans, $0.45): 143
bank_account, 115 routing_number, 107 transaction_id, 155 employee_id, 54
ticket_id, 50 customer_id. Hand audit before training fixed 36 labels (17
batch numbers labelled transaction_id, 7 dateline dates as date_of_birth, 8
"account ending in 4471" as credit_card, …). Three seeds of #131's config,
picked on tuned validation F1 (0.9496 / 0.9468 / 0.9486 → #135):

| run | Gretel P / R / F1 | CoNLL P / R / F1 | bank_account P / R (FPs) | employee_id P / R | credit_card | MRN | national_id | plate | scan / clean key |
|---|---|---|---|---|---|---|---|---|---|
| #131 (shipped until this round) | 0.717 / 0.784 / 0.749 | 0.866 / 0.620 / 0.723 | 0.30 / 1.00 (41) | 0.89 / 0.66 | 0.80 | 0.76 | 0.81 | 0.64 | 0.866 / 0.967 |
| #135 (tuned pick, shipped as export 36 until the scrub below) | 0.743 / 0.797 / 0.769 | 0.899 / 0.645 / 0.751 | 0.71 / 1.00 (7) | 0.80 / 0.76 | 0.89 | 0.77 | 0.81 | 0.59 | 0.859 / 0.965 |
| #136 | 0.755 / 0.777 / 0.766 | 0.850 / 0.614 / 0.713 | 0.62 / 0.88 (9) | 0.85 / 0.73 | 0.84 | 0.73 | 0.76 | 0.63 | 0.863 / 0.970 |
| #137 | 0.750 / 0.803 / 0.776 | 0.814 / 0.578 / 0.676 | 0.49 / 1.00 (18) | 0.81 / 0.76 | 0.87 | 0.78 | 0.79 | 0.60 | 0.867 / 0.965 |

The targeted effect is real on every seed: bank_account false positives 41
→ 7 / 9 / 18 (a count over 500 texts, beyond noise even though it varies
2.5× by seed), the ten cue-less card numbers gone everywhere (credit_card
precision 0.78 → 0.94–0.97), employee_id recall 0.66 → 0.73–0.76, Gretel F1
+0.02–0.03 on all three. What it cost, also on every seed: employee_id
precision 0.89 → 0.80–0.85 (new *fragment* predictions — the first subword
of logins and street numbers next to staff ids; the 14 username rows in
the document are the likely cause), license_plate 0.64 → 0.59–0.63 (≈ 3 of
55 spans, same-signed everywhere), national_id recall 0.79 → 0.71–0.75.
CoNLL: #135's 0.751 is seed luck — the three-seed mean (0.713) equals #131's
0.723 within a spread of 0.075. Own keys unchanged. Of the pick's 7
remaining bank_account errors, 3–4 are the benchmark's own convention (an
"Account Number: 3833…" it calls a card, an account after a SWIFT it leaves
unlabelled) — under our convention the ceiling on this benchmark is ~0.85,
and #135 is near it. The fresh reviewer's call: **ship #135 over #131** —
only the pick clears every bar set before the round, and the regressions
it carries are the data's (shared by its siblings), not the seed's.
**Shipped the same day: #135 = export 36** (superseded by #143 below), via
`POST /api/projects/31/seed-groups/134/ship {"export_format":"huggingface"}`.
This time one call was enough: #135 had no evaluation on record, so ship
ran the span lift check itself first (test examples F1 0.963, answer key
0.965, both 44 texts; the response reports it as `evaluated_before_ship`)
and the card came out complete on the first export — *Seed selection:
Picked seed 45 (run #135) of 3 by validation span F1 (tuned thresholds)
(… seed 45 0.9496, seed 46 0.9468, seed 47 0.9486)*, *Human review: 11 of
44 rows checked by a person (1 corrected). On the checked rows: precision
0.89 · recall 0.92 · F1 0.91 (11 texts)*, and the External benchmarks table
(CoNLL 0.90 / 0.64 / 0.75, Gretel 0.74 / 0.80 / 0.77). If another round: a
small negative-example document for employee_id (logins, house numbers and
customer ids beside staff ids) plus plates in prose; bank_account needs no
more data.

**Benchmark contamination, found and scrubbed (2026-10-17).** While reading
the next seed document, the lane noticed the local seeder had copied the
prompt's example values verbatim — and many of those examples had been lifted
from Gretel's own rows when the prompts were written from the confusion lists.
A scan of every training row against Gretel's gold values found **32 distinct
values in 591 places across 248 rows** (the employee id `G-82727-I` 57 times,
`KS40540825` 93, the plate shapes `X077-0102-227-0` / `B19-0947-371-68`, the
street names of three Gretel addresses …), from the look-alike rounds, the
contrast round and the new document. Every occurrence was replaced with an
invented value of the same length and character classes (offsets unchanged,
the 44 test rows untouched), and the shipped config was retrained on the
scrubbed data as a control:

| run | Gretel P / R / F1 | CoNLL-400 F1 | CoNLL-large F1 | employee_id P / R | plate F1 | bank_account P |
|---|---|---|---|---|---|---|
| #135 shipped (export 36, before the scrub) | 0.743 / 0.797 / 0.769 | 0.751 | 0.816 | 0.80 / 0.76 | 0.59 | 0.71 |
| #139 control | 0.731 / 0.781 / 0.755 | 0.724 | 0.787 | 0.54 / 0.77 | 0.64 | 0.65 |
| #140 control | 0.748 / 0.792 / 0.770 | 0.741 | 0.776 | 0.82 / 0.72 | 0.68 | 0.59 |
| #141 control (tuned pick) | 0.736 / 0.797 / 0.766 | 0.700 | 0.795 | 0.70 / 0.80 | 0.70 | 0.52 |

The leak bought #135 little overall (Gretel F1 within 0.01 of the clean
seeds, inside seed spread) but its employee_id figure was partly memorised
(F1 0.78 vs 0.63–0.77 clean, mostly precision), and the clean seeds read
0.02–0.04 lower on the large CoNLL slice. Lesson for the product: seed
prompts must never carry benchmark values, and a benchmark import should
scan the training rows for its gold values the way the leakage filter
already scans for whole rows.

**A larger CoNLL slice (2026-10-17).** The 400-sentence CoNLL test slice
resolves ~0.07 F1, so most CoNLL moves above were noise. Benchmark 198 =
the CoNLL-2003 *validation* split, 2,000 sentences (the import cap), 1,169
person spans, 0 leakage; it resolves ~0.03: #117 0.772, #131 0.823, #135
0.816 — #117 → #131 was real (+0.05), #131 → #135 a tie, and #135's +0.03
on the small slice was seed luck, as the three-seed mean had said.

**employee_id / license_plate negative-example round (2026-10-17).** Doc
757: 40 documents (fleet inspections, towing notices, rental agreements,
insurance claims, helpdesk logs, access-request forms, chat transcripts,
JSON audit logs, incident reports, courier notes, accident reports, crew
rotas) with staff ids beside the same shapes as customer ids, logins,
incident / audit / case / work-order ids, part numbers and flight numbers;
plates in prose in many formats beside VINs, National Insurance numbers,
SSNs, postcodes, phone and tracking numbers, driving-licence numbers. 75
passages, Sonnet-labelled (885 spans, $0.58); the hand audit fixed 40
(19 VINs typed device_id deleted, 6 dateline dates typed date_of_birth, 9
staff ids with no staff cue, `user_id` values retyped username, …), then the
scrub above. Three seeds of #135's config on the scrubbed data with the
document, tuned, pick by tuned validation F1 → **#143**:

| run | Gretel P / R / F1 | CoNLL-400 P / R / F1 | CoNLL-large P / R / F1 | employee_id P / R | plate F1 | bank_account P | credit_card | MRN | national_id | scan / clean key |
|---|---|---|---|---|---|---|---|---|---|---|
| #141 control pick | 0.736 / 0.797 / 0.766 | 0.874 / 0.584 / 0.700 | 0.934 / 0.692 / 0.795 | 0.70 / 0.80 | 0.70 | 0.52 | 0.89 | 0.76 | 0.77 | 0.867 / 0.970 |
| **#143 (tuned pick, shipped, export 37)** | 0.756 / 0.800 / 0.778 | 0.929 / 0.705 / 0.801 | 0.931 / 0.741 / 0.825 | 0.95 / 0.70 | 0.69 | 0.50 | 0.90 | 0.78 | 0.86 | 0.863 / 0.970 |
| #144 | 0.745 / 0.775 / 0.760 | 0.876 / 0.681 / 0.766 | 0.931 / 0.724 / 0.814 | 0.94 / 0.70 | 0.66 | 0.68 | 0.87 | 0.77 | 0.79 | 0.869 / 0.965 |
| #145 | 0.773 / 0.791 / 0.782 | 0.915 / 0.645 / 0.756 | 0.951 / 0.723 / 0.821 | 0.69 / 0.80 | 0.77 | 0.80 | 0.86 | 0.75 | 0.84 | 0.869 / 0.963 |

Against the clean control, seed by seed: employee_id precision 0.54 / 0.82
/ 0.70 → 0.95 / 0.94 / 0.69 (two seeds up ~0.2, none down — the data's
gain; recall slipped 0.77 / 0.72 / 0.80 → 0.70 / 0.70 / 0.80), plates
+0.05 / −0.02 / +0.07 (mixed sign — noise; the pick's plate false positives
are still NI numbers, MRNs and postcodes), and an unplanned real gain on the
large CoNLL slice, 0.787 / 0.776 / 0.795 → 0.825 / 0.814 / 0.821 (same sign
on all three, mean +0.034 ≥ the 0.03 the slice resolves — the document's
114 person names in forms and prose). Keys flat; bank_account precision
0.65 / 0.59 / 0.52 → 0.50 / 0.68 / 0.80 is 17-span noise. The fresh
reviewer's call: **ship #143** — clean, ties or beats #135 on every
resolvable benchmark, loses nothing above noise; and no further round —
seed-to-seed swings on identical data (employee_id F1 0.63–0.80) exceed
anything 103 / 55 / 17 spans can resolve.

**Shipped and published (2026-10-17): #143, export 37 →
[huggingface.co/TensorGreed/brewslm-sensitive-data-tagger](https://huggingface.co/TensorGreed/brewslm-sensitive-data-tagger).**
`POST /api/projects/31/seed-groups/142/ship` lift-checked #143 itself (test
examples F1 0.969, answer key 0.970, 44 texts) and wrote the card with the
seed-selection line, *Human review: 11 of 44 rows checked (1 corrected);
on the checked rows F1 0.92*, and all three benchmarks (CoNLL-400 0.93 /
0.70 / 0.80, CoNLL-large 0.93 / 0.74 / 0.83, Gretel 0.76 / 0.80 / 0.78).
Two lines were added to the card's Limitations by hand before publishing:
the Gretel per-type picture against the previous export (employee_id
recall −0.06, bank_account precision −0.21, precision +0.15, plates +0.10,
what the plate false positives still are) and the contamination note. Then
`POST /api/projects/31/export/37/publish-hub {"repo_id":
"TensorGreed/brewslm-sensitive-data-tagger"}` ([how it works](export-and-deployment.md#publishing-to-the-hugging-face-hub))
uploaded `model/` as the repo root and the manifest + release notes under
`brewslm/`; the card's front matter gained `license: mit` and `language:
en`. Checked from the Hub with the plain `transformers` pipeline
(`stride=128`): `employee_id` EMP204711 0.99, `credit_card` 1.00,
`license_plate` KA 05 MJ 4567 0.73, `phone` 1.00, `email` 0.99.

### Try it: the redact endpoint

```bash
curl -X POST localhost:8000/api/projects/31/redact \
  -H 'Content-Type: application/json' \
  -d '{"text": "Call Ada on 555-0101", "mode": "mask", "recall_bias": 0.2}'
```

`mode` is `mask` (`[PHONE]`), `tag` (`<phone>555-0101</phone>`) or
`spans` (text unchanged). The response carries every span with `score`
/ `min_score`, the redacted `text`, and `offsets` mapping each span in
the input to its replacement in the output. `experiment_id` picks a run
(default: the newest completed span run); `min_score` / `recall_bias`
override the run's `thresholds.json` for that call. The Results page's span
summary has a **Try redaction** box over the same endpoint, with a
recall-bias slider. The model stays loaded between calls (one model in
memory at a time).

### Long documents: pass `stride`

The encoder reads 512 tokens at a time. BrewSLM trains, validates,
predicts and evaluates a longer text in **overlapping windows** (128
tokens shared between neighbours; each token's label comes from the
window where it sits furthest from an edge), so a document of any length
is tagged end to end. The exported tokenizer's `model_max_length` is the
training window, so `stride=128` in the pipeline windows the model the
same way. **Without `stride` the pipeline cuts the text at 512 tokens.**
Exports of runs trained before windowing landed get the window written
into `tokenizer_config.json` at export time (the release notes say so) —
re-export an older run rather than reusing its old export.

Measured on project 31 (2026-10-14): the 44 test texts joined into four
multi-page documents of 780–915 tokens, run #58's weights, scored
against the Haiku answer key:

| read through | F1 | entities past token 512 found |
|---|---|---|
| each text alone (the reference) | 0.84 | — |
| BrewSLM before windowing (cut at 512) | 0.59 | 0 / 126 |
| old export, HF pipeline (whole document at once) | 0.59 (precision 0.51) | 110 / 126 |
| **BrewSLM, windowed** | **0.82** | 108 / 126 |
| **re-export, HF pipeline with `stride=128`** | **0.83** | 108 / 126 |

The windows cost almost nothing against reading each text alone (0.82
vs 0.84). What they don't fix: with a "Page n of 11" footer between the
texts, precision falls to 0.65 (F1 0.74) — the model tags page
furniture it never saw in training. That's a data gap, not a windowing
one: train on documents that carry headers and footers.

### Page headers and footers: train on real layout

Three pieces, all in the UI:

1. **Write sample documents → "Multi-page, with headers and footers"**
   (`layout: "paged"`). Each document is split into pages, and header
   and footer lines are inserted by code — letterheads, "CONFIDENTIAL",
   "Page 2 of 5", "- 2 -", print dates — varied per document.
2. **Label document passages with *model*** (**Data** → **Generate** step, span mode;
   `POST /synthetic/relabel-answer-key {target: "passages",
   document_id}`). The span generator *rewrites* passages into new
   snippets, which drops the layout; this labels the passages **as
   they are**, under your conventions, and adds them as training rows.
   They land in the review queue (source `passage_labels:doc=<id>`):
   accept them there — pending rows never train.
3. **Split with "Keep the current N test examples"** (`keep_test`). The
   test rows stay exactly as they were (rows with the same text never
   move into training), and only train / validation are re-divided — so
   the new run is scored on the same rows as the old one.

Project 31 (2026-10-14): 40 paged documents (7 min) → 129 passages, 83
of them with page lines → 128 labelled by gemma4 in 5 min (708
entities) → accepted → run #62. Scored on the same multi-page documents,
plus a footer style the paged layout never generates (`[3/11]` and a
hospital letterhead), against the Haiku key. Run #60 is the control:
the old training data, only re-split.

| documents | run #59 | run #60 (control) | **run #62 (+ passages)** |
|---|---|---|---|
| texts joined by blank lines | 0.82 | 0.84 | 0.84 |
| "Page n of 11" footers | 0.74 (precision 0.65) | 0.73 | **0.84 (precision 0.82)** |
| held-out footer + letterhead | 0.69 (precision 0.57) | 0.65 | **0.83 (precision 0.80)** |
| the 44 answer-key texts, one by one | 0.84 | 0.83 | 0.85 |

The footer gap closed, including for a style the model never saw; short
texts didn't regress. (All the measuring text was invented, as were the
footers.)

### Whole documents as training rows, and OCR noise

**Label whole documents with *model*** (**Data** → **Generate** step; `target:
"documents"`, `max_chars` default 4000) makes each training row a run of
whole paragraphs of one document — on paged sample documents, a whole
multi-page document with its page lines, usually longer than one
512-token window — so training runs in windows on real long rows. The
labeller sees each row in **passage-sized pieces** (~1,000 chars, cut at
line / sentence ends) and the spans are shifted back onto the row: asked
to label a whole document at once, gemma4 put ~4.1 entities per 1,000
characters on it against ~6.8 on passages, and a model trained on those
thin labels got worse. Every labelling path now pieces long texts.
**Relabel training rows** takes an optional source group
(`document_labels:doc=740`) to redo one batch.

Project 31 (2026-10-14): 40 more paged documents → 36 long rows (29 over
one window; training made 517 windows from 486 rows) → whole-document
labels 399 entities → in pieces 610. On the multi-page documents, F1 vs
the Haiku key:

| | run #62 (passages) | run #63 (whole, labelled whole) | **run #64 (whole, labelled in pieces)** |
|---|---|---|---|
| blank lines / footers / held-out footer | 0.84 / 0.84 / 0.83 | 0.79 / 0.80 / 0.79 | 0.85 / 0.85 / 0.85 |
| with 5% OCR-style noise | 0.76 / 0.76 / 0.77 | 0.73 / 0.74 / 0.75 | 0.76 / 0.78 / 0.75 |
| with 10% | 0.69 / 0.66 / 0.70 | 0.68 / 0.67 / 0.70 | 0.69 / 0.67 / 0.71 |
| the 44 answer-key texts one by one | 0.85 | 0.84 | 0.83 |

Long rows train correctly but add little over passages here (0.84 →
0.85 on 288 spans is within noise; 0.85 → 0.83 on the 44 texts too).
**OCR-style noise** — length-preserving character confusions (O↔0, l↔1,
S↔5, e→c, rn-style swaps, . ↔ ,) on that share of the confusable
characters, so a noisy entity is still the right span: the scores hold
at 2% (−0.01), lose ~0.07 at 5%, ~0.15 at 10% and ~0.30 at 20%. Noise
wasn't in the training data; that is the next lever.

### Training for OCR'd text: noisy copies

**OCR-noise copies** (Training config → Advanced → Power Tools, shown for
token classification; `ocr_noise_augment_rate` 0.05 / 0.1 / 0.2 and
`ocr_noise_augment_copies` 1–3) adds copies of every training row with
look-alike characters swapped at a rate drawn up to that level. The swaps
keep the text's length, so every span keeps its offsets; validation and
test rows are never noised, and auto-epochs counts the copies so the
optimizer budget stays the same. Training uses a subset of the
look-alikes (`TRAIN_CONFUSIONS`); `h g u Z 9 2 @ -` are kept out so the
measurement can include noise the model never saw.

Project 31 (2026-10-14), run #65 = run #64's data + one noisy copy per row
at up to 10% (455 vs 429 optimizer steps), on the multi-page documents
(average of the three layouts), F1 vs the Haiku key:

| noise | clean | 2% | 5% | 10% | 20% |
|---|---|---|---|---|---|
| run #64 | 0.85 | 0.83 | 0.76 | 0.69 | 0.51 |
| **run #65 (+ noisy copies)** | 0.83 | 0.82 | **0.81** | **0.79** | **0.75** |

At 10–20% noise the copies recover 0.10–0.25 F1, for 0.01–0.02 on clean
text (within noise; the 44 answer-key texts read 0.83 for both runs).
The look-alikes kept out of training barely hurt either run (0.82–0.85
at every level), so this doesn't show whether the robustness carries to
other kinds of OCR damage — broken words and lines aren't simulated.
Turn it on (Medium) when your documents come from scans; leave it off
for born-digital text.

### A scanned PDF, OCR'd by BrewSLM

Scanned PDFs are now read: pages without a text layer are rendered
(`pdftoppm`) and OCR'd (`tesseract`) during ingestion when those tools
are installed (`apt install tesseract-ocr poppler-utils`); without them
the upload says what to install. `BREWSLM_OCR=0` turns it off;
`BREWSLM_OCR_LANG` (e.g. `eng+deu`) and `BREWSLM_OCR_DPI` tune it.

Project 31 (2026-10-14): the 44 answer-key texts typeset on six A4 pages
with a letterhead and page footers, then degraded like a scan (skew,
blur, speckle, JPEG) into an image-only PDF → upload → BrewSLM OCR'd it
in 24 s (real OCR errors: `securehealth net`, `HPL- 112233`, stray `‘ ~ _`
at line starts, lines broken mid-sentence) → cleaned into 18 passages →
Haiku 4.5 labelled them as they are (279 spans, 50 s; rows rejected from
training with a reason — evaluation only). F1 against that key:

| run | precision | recall | F1 | on the clean texts (Haiku key) |
|---|---|---|---|---|
| #62 (passages) | 0.59 | 0.73 | 0.65 | 0.85 |
| #64 (whole documents) | 0.65 | 0.72 | 0.68 | 0.83 |
| #65 (+ OCR-noise copies) | 0.70 | 0.72 | **0.71** | 0.83 |

Real OCR costs ~0.15 F1 against clean text — more than the
character-swap simulation predicted, because OCR also breaks lines,
splits tokens and adds stray marks. The noise copies help a little,
mostly precision (+0.05 over #64); recall doesn't move. Weakest on the
scan: passport (0 of 6), bank account, person name (0.47 on 36 — names
broken across lines). 18 passages is a small test: treat ±0.03 as
noise.

**Training on OCR'd text.** *Write sample documents* has a third layout,
**Scanned (OCR'd) multi-page**: BrewSLM typesets the paged documents,
degrades the pages like a scan, saves an image-only PDF and OCRs it on
ingestion, so the passages carry real OCR errors (needs tesseract on the
server). Project 31: 40 scanned documents (13 min incl. ~150 pages of
OCR) → 127 passages → 125 labelled locally (630 entities) → accepted →
run #67 (+ OCR-noise copies, like #65):

| | run #65 | run #67 (+ OCR'd passages) |
|---|---|---|
| the real scan (Haiku key) | 0.71 | 0.72 |
| footer documents, clean / 10% noise | 0.84 / 0.79 | 0.84 / 0.80 |
| the 44 answer-key texts | 0.83 | 0.85 |

OCR'd training text didn't move the scan score. The weak types on the
scan are digit strings OCR mangles (passport, bank account, card
number) — and the local labeller, labelling the same mangled text, may
skip them, so the training labels are thinnest exactly where the model
needs help.

**A bigger scan test, and better labels on OCR'd text** (2026-10-14).
60 more scanned documents → 194 OCR'd passages → Haiku 4.5 labelled 192
(1,011 entities; rejected from training — evaluation only). Haiku also
relabelled the 125 OCR'd *training* passages (Relabel training rows,
source group `passage_labels:doc=742`): the local labels agreed with
Haiku's at only F1 0.65 (282 entities Haiku found that the local model
missed), so the local labels on OCR'd text were thin. Run #69 trained
on Haiku's labels:

| run | 18-passage scan | 192-passage scan | all 210 (1,290 spans) | clean answer key |
|---|---|---|---|---|
| #65 (passages + noise copies) | 0.71 | 0.59 | 0.62 | 0.83 |
| #67 (+ OCR'd passages, local labels) | 0.72 | 0.60 | 0.63 | 0.85 |
| #69 (+ OCR'd passages, Haiku labels) | 0.72 | **0.62** | **0.64** | **0.86** |

The 18-passage test flattered every run by ~0.1. On the larger one,
better labels on OCR'd text help a little (0.59 → 0.62, mostly recall)
and the clean score is the best yet, but scanned text stays ~0.22 below
clean. The weakest types on scans are now `provider_name` (0.29 on 82
spans), `visit_date` and `credit_card` — names and dates OCR splits
across lines, and card numbers whose digits it confuses.

**Scanned-text mode** (**Data** → **Generate** step → *Labelling conventions* →
"Texts come from scanned documents (OCR)"; the API is
`PUT /synthetic/type-definitions {"scan_tolerant": true}`). Two things change:

- **Labelling** — every model that labels your spans (the training-label
  generator, the relabeller, the answer key) is told that a value broken
  by stray spaces or marks (`2024-02- 10`, `US- Z1Y2X3W4`, `Sarah -
  Jenkins`) is one span, without the marks or a label word at its edges.
- **Decoding** — evaluation and the redact endpoint tidy predicted spans:
  stray marks are trimmed off span edges, and neighbouring spans of the
  same type separated only by whitespace and stray marks (never a comma,
  letter or digit) are joined — so a span can cross where an OCR line
  break was.

What it did on project 31's 210-passage scan test (2026-10-14). Cleaning
had already flattened every line break into a space, so the "break" is
the debris it leaves; categorising run #69's 410 missed spans showed most
are not debris at all — 189 entities missed outright (short fragments,
letterheads), ~80 a title convention ("Mr. Kevin Brown" vs "Kevin
Brown"), 44 a type mix-up (visit date read as a date of birth); debris
and broken values account for under a quarter.

| | old scan key | scan-tolerant key (re-keyed by Haiku) |
|---|---|---|
| run #69, plain decoding | 0.640 | 0.651 |
| run #69, OCR-tolerant decoding | 0.652 | **0.664** |
| run #70 (labels re-done under the convention), plain | 0.631 | 0.643 |
| run #70, OCR-tolerant decoding | 0.642 | 0.657 |

OCR-tolerant decoding helps every time (+0.012–0.014) and changes
nothing on clean text. Retraining on scan-tolerant labels did *not*
help (#70 sits ~0.01 below #69 — within noise; clean 0.855 vs 0.862).
The convention mainly made the answer key more consistent (17% of the
192-passage key's spans changed). The scan gap is now mostly entities
the model never finds, not their edges.

**Where the remaining misses came from — and two fixes that worked**
(2026-10-14, run #71). Two conventions were doing the damage, not OCR:

- **Titles.** The project's person_name definition predated the "name
  only, without a title" wording, so 52 names in the scan key and 12 in
  the clean key were labelled "Mr. Kevin Brown" while the model learned
  "Kevin Brown". Re-wording the definition and re-keying fixed the clean
  key (person_name 0.68 → 1.00 for every run — it was a key error). Haiku
  still kept a title on 21 scan names.
- **A card's last four digits.** "card ending in 4242" was labelled
  credit_card about half the time in training (89 of 178 left blank) and
  almost always in the keys — so the model found 14 of 26 on the scans.
  The fix: say so in the definition, and add a custom pattern
  (**Data** → **Generate** step → *Add a type* → pattern
  `(?:(?<=ending in )|(?<=last four digits )|…)\d{4}\b` for credit_card,
  flag i), whose regex pass labelled 175 more in the saved rows.
  Run #71 then finds 25 of 26 on the scans and 14 of 15 on the clean key.
  The shipped suggestions now say this for credit_card, and that cvv is
  the digits only.

| run | scan key (re-keyed, 1327 spans) | clean answer key (re-keyed, 44) |
|---|---|---|
| #69 | 0.675 | 0.910 |
| #70 (scan-tolerant labels) | 0.665 | 0.900 |
| #71 (+ no-title names, last-four labels) | **0.680** | **0.919** |

On the scans the gain is small (+0.005, noise-level). The short
fragments it missed were mostly last-four digits (fixed) and CVVs,
which on scans are often a bare "the CVV 123" after debris. What's left
is provider names (45 misses) and visit dates (26) the model never
tags — a coverage gap in the training text, not a decoding one.

**Provider names and visit dates (run #72, 2026-10-14).** Looking at
those misses first changed the plan: about half were the *key* breaking
the project's own conventions — Haiku tagged companies ("TechSphere
India Pvt Ltd") as provider_name and order, ticket and "document
generated" dates as visit_date. So the definitions now say what those
types are *not* (shipped suggestions updated too): visit_date is a date
the patient was or will be seen for care, never an order, transaction,
ticket, billing, print or "document generated" date; provider_name is
never a shop's, bank's or insurer's name. Then 40 more scanned documents
from the seed card (layout *scanned*, genres clinical plus order and
support emails as contrast) were labelled by Haiku as training rows:
114 passages, 205 provider names, 223 visit dates.

| run | scan key (v3, 1272 spans) | clean key (v3, 44) | scan provider_name | scan visit_date |
|---|---|---|---|---|
| #71 | 0.678 | 0.911 | 0.45 | 0.58 |
| #72 (+114 scanned clinical passages) | **0.684** | **0.924** | **0.52** | 0.53 |

provider_name improved; visit_date did not. Run #72 finds the same
visit dates (35 vs 34 exact) but adds false ones, mostly footer print
dates. That traced back to the training rows: 47 accepted rows still
label a "Printed …" / "Document generated …" date as a birth or visit
date, mostly in groups labelled before the new wording. Relabel the
older groups after changing a definition — a convention only reaches
the rows labelled under it.

**Groups labelled under older conventions.** Every labeller now stamps
the rows it writes with the version of the conventions it was given (a
short hash of the definitions and the scanned-text flag). The **Data** → **Generate**
step's review queue compares each source group with the project's current
version and lists the groups labelled under an older one, or before
versions were recorded, at the top of the queue (an "older conventions"
badge marks them in the accepted list too). Each has **Relabel under
current conventions with &lt;labeller&gt;**, which relabels that group in
place with the model that labelled it, and, for unstamped groups,
**Mark as current**: use that when you know the group was labelled under
today's wording; it stamps the rows without relabelling and records that
it was an attestation.

On project 31 the queue flagged every group at first, since nothing was
stamped yet. Docs 742 and 744 had been labelled by Haiku under the
current wording, so they were marked current. Docs 739 and 740 were
relabelled with their own labeller (the local model):

| run | scan key (v3) | scan visit_date | visit dates predicted | dates on print footers | clean key (v3) |
|---|---|---|---|---|---|
| #71 | 0.678 | 0.58 | 51 | 19 | 0.911 |
| #72 | 0.684 | 0.53 | 65 | 17 | 0.924 |
| #73 (739/740 relabelled) | **0.697** | **0.62** | 62 | 14 | 0.920 |

Visit dates recovered past run #71, and the scan score is the best so
far. The local model still labels some footer print dates despite the
wording (27 of the 31 such labels in 739/740 remained). The clean key
has about 19 visit dates, so its visit_date swings (0.92 → 0.85) are
within noise.

**Relabelling everything with one stronger labeller, and retiring what
hurts (run #74, 2026-10-14).** The strip has a **Relabel with** picker
(each group's own labeller, or any model that labelled rows in this
project) — so the five legacy generator groups and docs 739 / 740 were
relabelled with Haiku. Accepted groups also have **Retire group**: the
rows leave training (kept on disk as rejected, reason "retired") and a
"Retired groups" list offers **Restore**. To decide whether any group
should go, each legacy group was retired in turn, the model retrained and
scored, and the group restored (leave-one-out):

| run | scan key | scan visit_date | clean key |
|---|---|---|---|
| #73 | 0.697 | 0.62 | 0.920 |
| #74 (all relabelled by Haiku) | **0.702** | **0.68** | 0.906 |
| #75–#79 (each legacy group left out once) | 0.686–0.706 | 0.65–0.73 | 0.897–0.922 |

No group helped by its absence beyond noise (the five leave-one-out runs
spread about ±0.01 — a useful measure of what one training run's noise
looks like), so none was retired. Haiku also tags some footer print
dates despite the convention, so that error is the labeller's, not the
old groups'.

**A stronger answer key, and the first external check (2026-10-15).**
Both keys were re-keyed by Claude Sonnet. Sonnet agrees with Haiku's clean
key at F1 0.92 but with its scan key at only 0.76 — on OCR'd text the two
labellers disagree about a quarter of the spans, so scan scores carry a lot
of key noise. On the Sonnet keys runs #73–#79 score 0.71–0.74 (scan) and
0.85–0.88 (clean). Most of the clean-key gap is one ambiguity in the
conventions: `provider_name`'s example is "Dr. Chen" (title included) while
`person_name` says "without a title" — Haiku kept "Dr.", Sonnet dropped it.
Pick one and say it in both definitions.

The project's model was then scored on an [external benchmark](./external-benchmarks.md):
400 sentences of CoNLL-2003 news (people's names, labelled by people) — F1
0.21, recall 0.13. The model had only ever seen names in correspondence.
`person_name` was widened to any person's name ("private individuals and
public figures alike"), and 40 news-style documents (match reports, wire
stories, press releases, court reports — 107 passages, 375 names) were
labelled by Sonnet and added:

| run | CoNLL people (P / R / F1) | scan key (Sonnet) | clean key (Sonnet) |
|---|---|---|---|
| #74 | 0.52 / 0.13 / 0.21 | 0.735 | 0.861 |
| #86 (+ news names) | **0.90 / 0.33 / 0.49** | **0.757** | **0.882** |

The benchmark more than doubled and the project's own keys went up, not
down. Recall on news names is still low (0.33): sports results ("Andrei
Ivanov ( Russia ) 24.88") and surnames alone on later mentions are the
misses.

**One labeller, one wording (run #87, 2026-10-15).** `provider_name` now
excludes titles exactly like `person_name` ("Laura White" in "Dr. Laura
White"; the shipped suggestion says so too). Sonnet re-keyed both keys under
that wording, labelled 40 more news documents (surname-only later mentions,
sports results lines, initials — 101 passages), and relabelled every group
the review queue flagged as labelled under older conventions — 1,384 rows
through the strip's *Relabel with* request:

| run | scan key | clean key | CoNLL people (P / R / F1) |
|---|---|---|---|
| #74 | 0.727 | 0.837 | 0.52 / 0.13 / 0.21 |
| #86 | 0.752 | 0.858 | 0.90 / 0.33 / 0.49 |
| #87 | **0.856** | **0.949** | 0.82 / **0.56 / 0.67** |

Read the first two columns with care: the keys and the training labels now
come from the same labeller under the same conventions, so part of that jump
is the model agreeing with Sonnet, not being right. The CoNLL column is
labelled by people and owes nothing to our labellers — and it rose too
(recall 0.33 → 0.56), which is the evidence the model improved. 27 rows got
no labels back from Sonnet and kept their old ones, so five groups still show
in the stale list with a handful of rows each.

**What holds up independently (2026-10-15).** A second external benchmark,
500 texts of [Gretel's PII set](./external-benchmarks.md) (Apache-2.0,
machine-generated but independent of our labellers), maps 19 of its types
onto 14 of ours (`ssn` → national_id, `first_name` + `last_name` → one
person_name, `ipv4`/`ipv6` → ip_address …; generic `date` and the address
parts ignored). Then runs #86 and #87 were each retrained with three seeds
on their own data snapshot (the versioned split each one trained on):

| | #86 (3 seeds) | #87 (3 seeds) | holds up? |
|---|---|---|---|
| CoNLL people F1 (human labels) | 0.445 ± 0.003 | **0.666 ± 0.032** | yes — far beyond the seed spread |
| Gretel PII F1 (14 types) | 0.559 ± 0.009 | 0.562 ± 0.004 | no change |
| scan key (Sonnet) | 0.748 ± 0.005 | 0.861 ± 0.002 | yes, but partly self-agreement |
| clean key (Sonnet) | 0.852 ± 0.005 | 0.957 ± 0.004 | yes, but partly self-agreement |

The news names transferred; nothing else did. On Gretel the model finds
emails (0.85), IP addresses (0.76), names (0.72) and phones (0.68), and
misses identifiers in formats it never saw: license_plate 0.00 (55 spans),
bank_account 0.11, credit_card 0.40, address 0.46, employee_id 0.50,
medical_record_number 0.53. That is the next gap — identifier formats,
not more of the same documents.

The relabel button in the review queue now relabels only the rows that are
not labelled under the current conventions (`only_stale`), so retrying the
27 rows Sonnet skipped sent 27 rows, not the thousand around them.

**Identifier formats (P1p, 2026-10-15).** 50 more documents were generated
from a generic brief — licence plates from eight countries, sort codes /
IBANs / masked accounts, spaced / dashed / masked card numbers, employee
ids, MRNs, full addresses in several layouts (never from Gretel's
examples) — and labelled by Sonnet (90 passages: 131 plates, 113 addresses,
80 cards, 78 employee ids, 61 MRNs, 71 accounts / IBANs). Three seeds of
#87's config on the new split:

| | #87 (3 seeds) | + identifier rows (3 seeds) |
|---|---|---|
| Gretel PII F1 | 0.562 ± 0.004 | **0.619 ± 0.011** |
| CoNLL people F1 | 0.666 ± 0.032 | 0.678 ± 0.036 (no regression) |
| scan / clean key (Sonnet) | 0.861 / 0.957 | 0.866 / 0.976 |

Employee ids rose 0.45 → 0.62, email 0.81 → 0.94, IP addresses 0.75 →
0.83. License plates (0.00 → 0.08), card numbers (0.41 → 0.40) and bank
accounts (0.09 → 0.10) did not move, and the misses say why: the model
reads an identifier's *shape* and ignores the label in front of it — a
16-digit number after "Credit Card Number:" is tagged bank_account, and
Gretel's synthetic plates ("G71-6482-344-91") look like nothing but an id,
so only the word "License plate" could tell. Two misses are convention
differences, not errors: Gretel keeps an "ACCT-" prefix the scanned-text
rule strips. The next gap is context, not formats: rows where the same
shape carries different types depending on the label before it.

**Context over shape (P1q, 2026-10-15).** Two changes. The scanned-text
clause now keeps a prefix that is part of an identifier ("ACCT-3637328949",
"EMP-004521", "PT-30021" — joined by a hyphen or with no space) and strips
only label words in front of it ("Account number:", "SSN:", "MRN:"); the
clause's wording is now part of the conventions version, so changing it
flags every row labelled under the old wording. And 40 contrast documents
put the same shapes under different labels ("Credit card number: 4539 …" /
"Account number: 4539…" / "Customer ID: 4539…", id-like plates next to
employee numbers and MRNs) and list several ids on one line separated by
commas — 65 passages, 146 cards, 145 accounts, 115 plates. Sonnet re-keyed
both keys, labelled the new rows and relabelled every stale row through the
review queue:

| three seeds each | before (P1p) | after (P1q) |
|---|---|---|
| Gretel license_plate | 0.08 ± 0.04 | **0.46 ± 0.03** |
| Gretel credit_card | 0.40 ± 0.01 | **0.57 ± 0.07** |
| Gretel bank_account | 0.10 ± 0.02 | **0.23 ± 0.03** |
| Gretel PII F1 (14 types) | 0.619 ± 0.011 | **0.660 ± 0.010** |
| CoNLL people F1 | 0.678 ± 0.036 | 0.667 ± 0.012 |
| scan / clean key | 0.866 / 0.971 | 0.878 / 0.965 |

The model learned to read the label before an identifier: plates rose from
near zero, cards and accounts by half again, with nothing else moving.
Bank accounts are still the weakest mapped type (0.23).

**Shipping a seed (P1r, 2026-10-15).** The seed to ship was picked by the
project's own validation F1 (#102: 0.939 vs 0.936 / 0.929) — never by the
benchmarks it is then reported on. *Evaluate this run* now works on a single
seed of a seed group when you start it by hand (the automatic check still
evaluates the group as a whole). The export's model card has two new parts:
**Labelled by** (who last labelled the training rows — here all 1,021 by
Sonnet, though they were generated by four different models) and
**External benchmarks** (each benchmark's source, licence, scored types and
P / R / F1 for this run). Scoring the exported files directly reproduces the
run's numbers (CoNLL 0.664, Gretel 0.661).

Why bank_account stays low on Gretel: the model finds 16 of its 18 account
numbers — the problem is 98 extra spans. bank_account has become the model's
catch-all for digit strings it can't place: Gretel calls them card numbers
(20), coordinate fragments (15), routing numbers (15), device ids (14),
customer ids, plates, MRNs. The fix is negatives — numbers that are not
sensitive, left unlabelled — not more accounts.

The 44 *test examples* still carry the first teacher's labels, which predate
most of the conventions (titles, provider names, prefixes): the run scores
0.82 there against 0.96 on the Sonnet answer key, and provider_name 0.19 /
credit_card 0.35 on the test examples are convention disagreements, not
model errors. Re-key them, or read the answer key.

**Negatives, and a null result (P1s, 2026-10-15).** The bank_account
definition said "a domestic bank account, routing or sort-code number" —
so the routing numbers Gretel counts against us were our own convention.
It now says a routing number / ABA / SWIFT / sort code identifies the bank
and is not labelled (project and shipped suggestion). 40 documents full of
numbers that are not sensitive (coordinates, routing and SWIFT codes,
device / order / tracking ids, VINs, SKUs — ~370 number-like strings, 234
labelled spans) were added, and the 44 test examples were relabelled in
place by Sonnet (new **Relabel test examples** button: a test split kept
across re-splits never follows relabelled training rows; saved versions
holding the same split are updated too). Three seeds each:

| | P1q | P1s |
|---|---|---|
| Gretel precision | 0.583 ± 0.011 | 0.588 ± 0.005 |
| Gretel bank_account precision | 0.133 ± 0.021 | 0.144 ± 0.028 |
| Gretel PII F1 | 0.660 ± 0.010 | 0.661 ± 0.008 |
| CoNLL people F1 | 0.667 ± 0.012 | 0.685 ± 0.006 |

Within noise. The targeted errors did go — bank_account spans on Gretel's
coordinates 15 → 0, on routing numbers 15 → 7 — but as many appeared on
id formats the model had never seen (medical record numbers 5 → 19,
customer ids 5 → 11, SSNs 0 → 6). bank_account is still the model's
catch-all for an unfamiliar number; fixing formats one at a time moves the
error rather than removing it. 20 of the extra spans are fragments shorter
than five characters.

Relabelled by the same model as the answer key, the 44 test examples now
agree with it at 0.95 — they stopped being a second opinion.

**Larger model vs tuned thresholds (P1t, 2026-10-15).** Three seeds each of
deberta-v3-large (learning rate 2e-5, ~15 min a seed) and of the base model,
each with and without per-type thresholds tuned on the validation split
(*Tune thresholds* on the span Eval card):

| three seeds each | base | base + thresholds | large | large + thresholds |
|---|---|---|---|---|
| Gretel precision | 0.588 ± 0.005 | **0.687 ± 0.007** | 0.563 | 0.674 |
| Gretel F1 | 0.661 ± 0.008 | **0.705 ± 0.010** | 0.612 | 0.655 |
| CoNLL people F1 | 0.685 | **0.685** | 0.579 | 0.579 |
| scan / clean key | 0.878 / 0.965 | **0.889 / 0.965** | 0.857 / 0.943 | 0.876 / 0.951 |

The thresholds win: +0.10 precision and +0.04 F1 on Gretel, CoNLL untouched
(person_name's threshold stayed at 0). The large model lost everywhere at
this one configuration. The winner was shipped with *Ship this seed*:
seed 43 (run #107) of three, picked by validation span F1 — its model card
records the rule and the values, its thresholds.json the tuned thresholds,
and its benchmark table Gretel 0.71 / CoNLL 0.69.

**Final round and the shipped model (2026-10-15).** One last combined
round: 50 documents of identifier contexts for the weakest benchmark types
and 40 of news names (178 passages, Sonnet-labelled), three seeds of the
shipped config, thresholds tuned on validation for each, seed picked by
validation span F1 and exported with *Ship this seed*:

| | previous (#107) | new seeds (#115–117, mean) | shipped (#117) |
|---|---|---|---|
| Gretel PII F1 | 0.710 | 0.708 | 0.708 |
| CoNLL people F1 | 0.689 | 0.676 | 0.708 |
| scan / clean key | 0.891 / 0.966 | 0.885 / 0.966 | 0.882 / 0.968 |
| Gretel credit_card | 0.57 | 0.78 | 0.74 |
| Gretel bank_account | 0.34 | 0.61 | 0.58 |
| Gretel employee_id | 0.59 | 0.70 | 0.75 |
| Gretel license_plate | 0.68 | 0.52 | 0.54 |

Overall the round is a tie within seed noise, but it moves the types that
matter most for a redaction tool: card numbers and bank accounts (PCI) gain
~0.2, plates lose ~0.15. Run #117 (export 33) was the shipped model until
2026-10-17; #107 (export 32) remains the alternative where plates matter
more.

**The shipped model (2026-10-17): #143, export 37, published on the Hugging
Face Hub as `TensorGreed/brewslm-sensitive-data-tagger`.** Six data rounds
after #117 (identifier shapes, two look-alike rounds, prose names, the
bank_account / employee_id contrasts, the employee_id / plate negative
examples — all above), a benchmark-contamination scrub, and a larger CoNLL
slice. Three ships in one day: #131 replaced #117 in the morning, #135
replaced #131 after the contrast round, and #143 — the first model trained
on scrubbed data — replaced #135 in the afternoon:

| | #117 (export 33) | #131 (export 35) | #135 (export 36, pre-scrub) | #143 (export 37, shipped) |
|---|---|---|---|---|
| Gretel PII P / R / F1 | 0.680 / 0.740 / 0.708 | 0.717 / 0.784 / 0.749 | 0.743 / 0.797 / 0.769 | 0.756 / 0.800 / 0.778 |
| CoNLL people, 400 sentences | 0.883 / 0.590 / 0.708 | 0.866 / 0.620 / 0.723 | 0.899 / 0.645 / 0.751 | 0.929 / 0.705 / 0.801 |
| CoNLL people, 2,000 sentences (1,169 spans) | 0.936 / 0.658 / 0.772 | 0.944 / 0.730 / 0.823 | 0.936 / 0.723 / 0.816 | 0.931 / 0.741 / 0.825 |
| scan / clean key F1 (v5, 30 types) | 0.852 / 0.962 | 0.866 / 0.967 | 0.859 / 0.965 | 0.863 / 0.970 |
| answer key F1 (checked rows) | 0.968 on 43 texts (15 checked: 0.905) | 0.967 on 44 (11 checked: 0.92) | 0.965 on 44 (11 checked: 0.91) | 0.970 on 44 (11 checked: 0.92) |
| Gretel credit_card | 0.74 | 0.80 | 0.89 | 0.90 |
| Gretel medical_record_number | 0.66 | 0.76 | 0.77 | 0.78 |
| Gretel national_id | 0.67 | 0.81 | 0.81 | 0.86 |
| Gretel license_plate | 0.54 | 0.64 | 0.59 | 0.69 |
| Gretel employee_id P / R | — | 0.89 / 0.66 | 0.80 / 0.76 | 0.95 / 0.70 |
| Gretel bank_account precision (17 spans) | 0.40 | 0.30 | 0.71 | 0.50 |

Only the large CoNLL slice (~0.03) and, overall, Gretel (~0.04) resolve
anything; the per-type Gretel numbers are directional. On what resolves:
#117 → #131 real on CoNLL (+0.05) and Gretel (+0.04); #131 → #135 → #143
ties on CoNLL-large, Gretel +0.03 across the two steps. #135's employee_id
and bank_account figures were measured with Gretel values in its training
rows and are overstated; #143's are clean. The #117 answer-key figure is a
different key (re-written under the 30-type vocabulary; 11 of the 15
verdicts carried over), so only the scan / clean rows compare like for
like on the own data — and those keys resolve ~0.04 overall, not per type.
That closes the training work for this project for now (the user's call:
today's numbers are satisfactory; revisit later): from the first span
model (CoNLL F1 0.21, a scan F1 of 0.59) to Gretel 0.78 and CoNLL 0.80 /
0.83 on independent, uncontaminated data, with every pick made on
validation alone.

## What it costs to relabel

Measured on project 31 (2026-10-15), one 114-row group, list-price estimates
from each job's own token counts:

| | cost | agreement with Sonnet-only labels |
|---|---|---|
| Sonnet, no prompt cache (before) | $0.506 | — |
| Sonnet, prompt cache | **$0.429** (−15%) | — |
| tiered: Haiku, Sonnet on rows where Haiku and the model disagree | $0.617 | 0.92 |
| Haiku alone | $0.150 | 0.57 |

- **Prompt caching saves ~15%, not more**: output tokens (~1,000 a call,
  priced 5× input) dominate, and only the ~1,500-token instruction block is
  cached. Haiku did not cache at all — the block is below its minimum
  cacheable length.
- **The tiered mode did not pay here**: 113 of 114 rows disagreed with the
  trained model (any one span differing in a dense passage counts), so
  nearly every row went to Sonnet on top of Haiku's cost. On this project
  Haiku also follows the conventions poorly (0.57 against Sonnet), so a
  looser rule would still send most rows. Use it on projects where the cheap
  labeller and the model mostly agree.
- **Targeted relabels are the big saving**: after a one-type wording change
  (license_plate), only 154 of 1,927 stale rows involve the type; the rest
  are carried forward without a call — ~92% less than relabelling all.
- **Batch mode** halves whatever remains.

A full Sonnet relabel of the 1,927 rows: ≈ $8.55 before, ≈ $7.25 with the
cache, ≈ $3.60 with cache + batch; a typical one-type change ≈ $0.58, or
≈ $0.29 batched.

**Batch mode, live (2026-10-15).** A 16-row group relabelled by Haiku through
Anthropic's Message Batches API: 3 requests, all succeeded in ~3 minutes,
none fell back to the direct path, $0.009 at half price.

**Tiered mode with an agreement threshold.** A row now goes to the review
model only when its bulk labels and the trained model agree less than a
threshold (row span F1; *Review disagreements with … when a row's
agreement is below* in the review queue; default 0.8; the job reports the
distribution). Same 114-row group:

| | cost | rows sent to Sonnet | agreement with Sonnet-only |
|---|---|---|---|
| Sonnet only | $0.429 | — | 1.00 |
| tiered, any difference | $0.617 | 113 | 0.92 |
| tiered, below 0.8 | $0.533 | 104 | 0.92 |
| tiered, below 0.5 | $0.275 | 39 | 0.73 |
| Haiku only | $0.150 | — | 0.57 |

Most rows sit between 0.5 and 0.8 agreement, so a threshold either still
sends nearly everything to Sonnet or saves ~36% at a real loss of quality.
On this project the cheaper labeller is too far from the conventions for
the tiered mode to pay; it is for projects where the bulk labeller and the
model mostly agree — the job's agreement distribution shows which you have.

## What the first run produced (2026-10-13)

Everything below ran on one machine with local Ollama (gemma4) as the
only teacher — no cloud calls at all.

| step | result |
|---|---|
| Write sample documents | 80 documents across 10 genres, 8.4 min → 106 passages |
| span synthesis (23 entity types, all passages) | **295 rows, ~1,950 entities**, 42 min; the regex pass adds the obvious emails / phones / cards the teacher missed |
| split | 228 / 28 / 29 |
| train — token classification, `microsoft/deberta-v3-base` | 27 epochs / 405 steps, **46 s** on a GB10; validation span F1 0.91 |
| held-out eval, 29 test examples (`span_set`, strict type + offsets) | **P 0.88 / R 0.92 / F1 0.90**, 16 of 29 rows exactly right; 1.00 on email, phone, IBAN, IP, CVV, health-plan id, ticket id; weak: passport 0 / 3 found, medication 0.60, api_key 0.67, provider_name 0.67 (6 false positives), national_id 0.76 |
| refine — 150 more rows asked for the weak types | 87 rows kept (the teacher rarely invents API keys: 2); re-split 297 / 37 / 38; re-train 22 epochs / 418 steps, 61 s |
| held-out eval after refine, 38 test examples | F1 **0.90** overall (0.895); the targeted types moved: passport 0.00 → **0.62**, medication 0.60 → **0.79**, provider_name 0.67 → **0.83**, national_id 0.76 → **0.87**, diagnosis 0.88 → **0.96**. (A different, larger test split — the per-type numbers are the comparison, not the overall.) |
| refine, second pass — the **Refine weak types** button (one click) | read run #55's evaluation: bank_account 0.00, passport 0.62, api_key 0.67, medication 0.79 → 100 rows asked for those four → **55 rows in 5 min** (12 per call, none empty: bank_account 31, api_key 19, passport 19, medication 15) → re-split 341 / 42 / 44 → re-train 63 s → bank_account **0.00 → 0.71**; passport 0.62, api_key 0.67, medication 0.76 — within noise on 8 / 3 / 17 gold spans. Overall F1 0.88 on the new 44-row split. The type with no coverage came up from nothing; the rest are limited by how few of them the test split holds |
| export → Hugging Face | `model/` + tokenizer + `label_space.json`, 744 MB; the snippet above on a fresh UK-style message tagged email, phone, IBAN, CVV, employee id, ticket id, provider and diagnosis at the right offsets, split a Mastercard-format PAN and missed a name inside a greeting — data diversity, the next refine target |

What was wrong before this run could happen — all fixed in BrewSLM, not
worked around: cleaning redacted the "PII" in the seed documents; the
span teacher's wrong offsets dropped the entity instead of being
repaired from its text; an empty teacher reply fell back silently to
regex-tagged sentence fragments; a thinking model through the
OpenAI-compatible URL returned empty content (use the synth backend,
which turns thinking off); 50 rows per call made the model skip the
labels (a dozen per call now); the first training ran 45 steps (epochs
now size to ~400 steps); and the held-out eval measured perplexity
because the span recipe's adapter didn't declare its task profile. Each
is a sentence here and a commit in the log.

## What it costs

Local only: the seed documents (60 docs ≈ 20 min on a 12B Ollama model),
the span rows (≈3 s per passage), training (a few minutes on a GPU; the
encoder is small), the evaluation (seconds). Cloud: only what you choose
to send — the answer key, or a judge pass — and the **Data** → **Generate** step's
backend picker shows the model name on every row it wrote.

**Batch relabels at half price.** A relabel that can wait — the answer
key, Relabel training rows, the passages or whole documents, or a stale
group from the review queue's strip — can go through Anthropic's Message
Batches API: tick **Batch (half price, slower)** next to the buttons
(enabled only for a `cloud:anthropic:…` model; the API answers 400
`BATCH_NEEDS_ANTHROPIC` otherwise, `use_batch: true` on
`POST /synthetic/relabel-answer-key`). Every label call of the job is
submitted as one batch with the same prompts (the conventions still go
as the cached block), billed at half the list price; results take
minutes to hours, and the bell shows "batch submitted · N requests ·
waiting" until they're in. A request that errors or expires is redone
directly at the normal price, split in halves if its reply won't parse,
exactly as without a batch. The finished job's cost line counts the
batch tokens at half price and says "via batch". The tiered review pass
(Review disagreements with) stays synchronous.

<!-- numbers: filled in from the first full run (see ROADMAP) -->

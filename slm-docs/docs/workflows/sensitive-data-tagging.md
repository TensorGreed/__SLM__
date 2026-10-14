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
support tickets and clinical notes"). On the **Data** tab the first card,
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

On the **Synthetic** tab, with the project's task shape set to span
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

**Evaluate this run** on the Eval tab (it also runs by itself after
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
Synthetic tab (span mode) reads the latest span evaluation, names every
type under F1 0.8 with its score ("Weak on run #55: passport (0.00),
medication (0.60), …"), and starts the span generator restricted to
those types, sampling from all passages, with a row count sized to how
many types are weak (`GET /synthetic/weak-types?threshold=`). Accept the
rows, re-split, re-train; the Eval summary compares the runs.

### An answer key the teacher didn't write

Every number above scores the model against labels the *same* local
teacher wrote. **Write answer key with this model** (Synthetic tab, span
mode) hands the test split's texts — unchanged — to the provider and
model selected in the panel (a cloud pin such as
`cloud:anthropic:claude-haiku-4-5-20251001` with the project's saved key,
or another local model), asks only for the entities, verifies every
offset from the entity text, and keeps the result as the project's
**gold_test** set (`POST /synthetic/relabel-answer-key`, Job
`span_relabel`). The bell reports rows, entities and the **agreement
between the old labels and the new key** (strict span F1, per type) —
the share of "error" that was ever the student's. Then evaluate the run
against `gold_test` on the Eval tab: the per-type scores are now judged
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

### One labelling convention for both labellers

Two models can both be right and still disagree — is a date in a
clinical note a date of birth or a visit date, is `GB29 NWBK …` a bank
account or an IBAN, is "Dr. Chen" a person or a provider? **Labelling
conventions** (Synthetic tab, span mode) is one line per type — what
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
2. **Label document passages with *model*** (Synthetic tab, span mode;
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
footers; real scanned documents add OCR noise this doesn't test.)

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
to send — the answer key, or a judge pass — and the Synthetic tab's
backend picker shows the model name on every row it wrote.

<!-- numbers: filled in from the first full run (see ROADMAP) -->

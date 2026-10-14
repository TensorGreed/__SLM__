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

The held-out evaluation recognises a token-classification checkpoint and
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

## 5. Export → Hugging Face

**Export → Hugging Face** packages the run's `model/` directory as-is
(there is no adapter to merge). Use it anywhere Transformers runs:

```python
from transformers import pipeline

tagger = pipeline("token-classification", model="path/or/hub-id", aggregation_strategy="simple")
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

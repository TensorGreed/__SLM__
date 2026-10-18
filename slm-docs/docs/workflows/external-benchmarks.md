---
title: External benchmarks
sidebar_label: External benchmarks
---

# External benchmarks

Your answer key tells you how a run does on rows from the same pipeline that
made its training data. An **external benchmark** is a labelled set someone
else made — a public dataset or a file you upload — scored next to your own
key, so you can see whether the model holds up on text it never resembled.

Works for **classification**, **span (entity) tagging**, **Q&A** and
**summarization** projects.

## Add one (Results page → External benchmarks → + Add a benchmark)

1. **Source** — a Hugging Face dataset, `hf:<dataset>[:<split>[:<revision>]]`
   (e.g. `hf:Yelp/yelp_review_full:test`), or upload a CSV / JSONL / Parquet
   file. Rows are **sampled across the source** (not the first N — datasets
   are often sorted by label).
2. **Preview** — BrewSLM detects the shape:
   - *classification*: a text column + a label column (a Hugging Face
     `ClassLabel` is read by name, e.g. `1 star`, not `0`);
   - *spans*: a text column + a list of `{start, end, label}` offsets,
     `tokens` + BIO `ner_tags` (joined with spaces, offsets computed), or a
     list of entity strings without offsets (`[{"entity": "jo@x.io",
     "types": ["email"]}]`, also as a JSON / Python-literal string) — each
     value is located in the text (every occurrence, longest first; an entity
     with several types takes the first);
   - *Q&A*: a question column + an answer column, optionally a context
     column (SQuAD-style `answers` — `{"text": [...], "answer_start": [...]}`
     — read their first text). If your project doesn't answer from a context
     (anything but a grounded / RAG Q&A project), the context is put in front
     of the question (`Context: … Question: …`) so the question stays
     answerable; the benchmark says *context folded into the question*;
   - *summarization*: a document column (`article`, `document`, `dialogue`…)
     + a summary column (`summary`, `highlights`, `abstract`…).

   An "answer" column that looks like labels (a few distinct one-word values)
   is read as classification, not Q&A.
3. **Map the labels** — every benchmark label onto one of your labels, onto
   **several** (Ctrl/Cmd-click — a *group*), or onto none (**ignore**).
   Several benchmark labels can map to one of yours (`1 star`, `2 star` →
   `negative`). One benchmark label can map to a group of yours (`DATE` →
   `visit_date` + `date_of_birth`, when the benchmark labels every date one
   way and your project splits them): a prediction of **any** label in the
   group counts as the group, and the score lists the group as one label
   (`date_of_birth + visit_date`). A label of yours can be in one group only
   — the form flags an overlap and won't add the benchmark.
   Ignored classification rows are dropped; ignored entity types are removed
   and **not scored** — the model isn't penalised for types the benchmark
   never labels. For spans, *Join neighbouring spans* merges e.g. a first and
   a last name that map to the same type.

   Q&A and summarization benchmarks have **no labels to map** — the form says
   so and goes straight to the name and licence.
4. **Licence + local only** — the licence comes from the dataset card when
   there is one (edit it otherwise). *Local only* marks data under a use
   agreement: it can only be scored with a model on this machine.

Rows that match your training data (exact or near-identical) are dropped, so
the benchmark stays held out. The benchmark is locked and eval-only — it never
becomes a training source.

## Gold values in your training rows (contamination)

Dropping identical rows is not enough: a benchmark's **answers** can leak into
training text on their own. On 2026-10-17, 32 values from a PII benchmark
turned up 591 times in a project's training rows — the seed-document prompts
had used the benchmark's own ids and plates as examples, and the local seeder
copied them word for word. The model's score on those types measured
memorisation, and nothing flagged it.

So every import now scans the benchmark's **gold values** — each entity text
of a span benchmark, each answer of a Q&A or summarization one (classification
benchmarks have labels, not values, and are skipped) — against the project's
training rows (the prepared train + validation splits, else the default
training sources), whitespace- and case-insensitive, values of 4+ characters.
Each benchmark in the panel then says either *None of its N gold values appear
in your training rows* or, for example:

> 32 of 1,169 gold values (3%) appear in your training rows, 591 times
> (employee_id 20, plate 12) · notable

with *which values* listing the worst offenders, how many rows hold each and
which source they came from (`seed_documents:doc=7`, `documents_qa_flow`,
`cleaned`…). The label is a share of the benchmark's distinct values: *low*
under 2 %, *notable* from 2 %, *severe* from 10 %. A hit is evidence, not
proof — a common first name from a news benchmark will legitimately occur in
support tickets, and a single one stays *low*. A long list of ids, addresses
and plates that all occur in your text is the signature of copied examples.

Training rows change after import (seeding, synthesis, re-splits), so
*rescan* re-runs the check against the current rows
(`POST /api/projects/{id}/benchmarks/{benchmark_id}/contamination`) and
replaces the stored block. The scan is stored on the benchmark manifest
(`contamination`) and, when values were found, the exported span model's card
carries one line per affected benchmark under *External benchmarks*.

If values were found: scrub them from the training rows with invented values
of the same shape, retrain, and rescan before trusting the number. Never put
a benchmark's values into a seeding or synthesis prompt — describe the shape
instead.

## Score a run

*Score this run* (or *Re-score*) runs the selected run over the benchmark as a
background job (the bell shows the result). Classification: accuracy +
macro-F1 per class. Spans: precision / recall / F1 over the mapped types only.
Q&A and summarization are scored the way your own test examples are: exact
match and token F1, and — for long answers (a median reference of 8+ words)
— the LLM answer judge your project resolves (a local model first). The
headline is *judged correct* when the judge ran, otherwise F1. A **local-only**
benchmark is never sent to a cloud judge: if the project's judge is a cloud
model, the answers are not judged and the score line says *not judged:
local-only benchmark, cloud judge (…)*.
The Eval summary card lists every benchmark's score for the run, with its
source and licence.

Scores are kept per benchmark (`benchmarks/<name>/results.jsonl`), separate
from the project's own evaluations. Each benchmark in the panel shows its
**score history** — a small trend line with one point per scored run (hover a
point for run, date and score) and "first → last" — read from `GET
/api/projects/{id}/benchmarks/{benchmark_id}/history`. The line only joins
scores of the same metric: an F1 from before a judge was available isn't
drawn on the same line as later judged scores.

## What it showed on two projects (2026-10-15)

**Review sentiment** (a DeBERTa classifier trained on 1,500 Amazon reviews),
benchmarked on 241 Yelp reviews (1–2★ → negative, 3★ ignored, 4–5★ →
positive):

| run | its own test examples | Yelp benchmark |
|---|---|---|
| full fine-tune | 0.67 | **0.74** |
| LoRA | 0.73 | 0.61 |

The two runs swap places on the benchmark — the one that looks better on its
own split transfers worse to another site's reviews.

**Sensitive-data tagger** (project 31), benchmarked on 400 sentences of the
CoNLL-2003 test set — real news text, labelled by people — with PER →
person_name and every other type ignored: precision 0.52, recall **0.13**,
F1 0.21 for run #74. The model rarely finds a name in sports results and
news, and calls some of the ones it finds `provider_name`. Its own keys say
0.90 / 0.70 — the benchmark shows how far that holds outside the genres it
was trained on.

The first classification attempt also surfaced three training bugs, fixed in
the same release: LoRA found no layers to adapt on encoder models (DeBERTa,
BERT…), full fine-tunes and classification heads trained in bf16 and stayed
at chance, and a fully fine-tuned classifier couldn't be evaluated.

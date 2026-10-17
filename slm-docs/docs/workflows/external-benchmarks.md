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

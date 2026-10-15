---
title: External benchmarks
sidebar_label: External benchmarks
---

# External benchmarks

Your answer key tells you how a run does on rows from the same pipeline that
made its training data. An **external benchmark** is a labelled set someone
else made — a public dataset or a file you upload — scored next to your own
key, so you can see whether the model holds up on text it never resembled.

Works for **classification** and **span (entity) tagging** projects.

## Add one (Eval tab → External benchmarks → + Add a benchmark)

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
     with several types takes the first).
3. **Map the labels** — every benchmark label onto one of your labels or
   **ignore**. Several can map to one (`1 star`, `2 star` → `negative`).
   Ignored classification rows are dropped; ignored entity types are removed
   and **not scored** — the model isn't penalised for types the benchmark
   never labels. For spans, *Join neighbouring spans* merges e.g. a first and
   a last name that map to the same type.
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
The Eval summary card lists every benchmark's score for the run, with its
source and licence.

Scores are kept per benchmark (`benchmarks/<name>/results.jsonl`), separate
from the project's own evaluations.

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

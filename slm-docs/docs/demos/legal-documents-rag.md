---
sidebar_position: 4
title: Legal assistant from documents (RAG vs fine-tuning)
---

# Legal assistant from documents — the gate that failed, and what worked

A documents-only case: two Canadian federal statutes (the Privacy Act and
the Access to Information Act, HTML from laws-lois.justice.gc.ca), 467
cleaned passages, no questions or answers. The goal is a small model that
answers questions about the acts **correctly** — and a platform that can
tell whether it does.

This page is the written form of walkthrough video 20. Every number below
is a live measurement from the project it was recorded on; nothing was
tuned for the demo.

## 1. Documents → Q&A → trained model, as one job

The Training tab's **Turn your documents into a Q&A assistant** card runs
the whole path as one job: a local model writes three question → answer
pairs per passage for training and one different question per passage for
the answer key; the pairs are split into train / validation / test examples;
the project becomes a Q&A project; training starts with the defaults; the
lift check follows. Here: 60 passages → 180 pairs + 60 answer-key rows →
split 142 / 17 / 19. **Train again on the current base model** reuses the
same pairs on a second base model, so models are compared on the same data.

## 2. The gate that passed — and why it is the wrong ruler

Token F1 on the held-out test examples:

| run | base F1 | fine-tuned F1 | rows better / worse |
|---|---|---|---|
| SmolLM2-135M-Instruct | 0.139 | 0.345 (+148%) | 19 / 0 |
| Qwen2.5-1.5B-Instruct, 3 seeds | 0.091 | 0.230 ± 0.022 (+153%) | 17 / 1 per seed |

Beyond row noise, held across seeds. A user would ship this. But token F1
measures word overlap: on paragraph-length statute answers it rewards
sounding like the act, not stating what it says.

## 3. The gate that failed — the judge

For long-answer generative tasks the held-out eval also asks a **judge
model** (here `gemma4:12b` via Ollama, local) to read the question, the
answer key and the model's answer and grade the facts: correct (1), partial
(0.5) or wrong (0). It is the headline whenever both evals have it, and it
pairs row by row like F1.

| run | judge score (base → fine-tuned) | correct / partial / wrong | verdict |
|---|---|---|---|
| SmolLM2-135M | 0.079 → 0.132 | 0 / 5 / 14 | within noise |
| Qwen2.5-1.5B | 0.079 → 0.053 | 1 / 0 / 18 | within noise (worse) |

The judge's reasons under each failure read "answers a different
question", "contradicts the section", "lists the included items instead of
the excluded ones". Fine-tuning on 180 generated pairs taught both models
the statute's **wording**, not its **facts**. The summary card shows this
verdict instead of the F1 one; F1 stays listed below it.

## 4. What got the facts right — retrieval over the passages

Under **Advanced evaluation → Auto-RAG comparison**, the third card scores
the untouched base model answering from the project's **document passages**
(retrieved per question, with an instruction to cite them or say it doesn't
know), judged, on the same 19 test examples:

| retrieval | judge | correct / partial / wrong | rows better / worse |
|---|---|---|---|
| none (base model alone) | 0.08 | 0 / 3 / 16 | — |
| BM25 top-3 | 0.61 | 8 / 7 / 4 | 14 / 2 |
| BM25 top-5 | 0.53 | 7 / 6 / 6 | |
| **top-3 + cross-encoder reranker** | **0.74** | **12 / 4 / 3** | 15 / 1 |
| top-5 + reranker | 0.66 | 9 / 7 / 3 | |

More passages alone *hurt* — a 1.5B model loses the thread — while
reranking BM25's candidates with a cross-encoder fixed most wrong-section
retrievals. The Coach reads the two verdicts (same judge, same rows) and
says **Retrieval over your documents beats the fine-tune (0.61 vs 0.05
judged)** with a one-click **Reroute to RAG**.

## 5. The RAG sibling

Reroute clones the project into a sibling that serves the base model +
passage retrieval, with no training run. Because it never trains, it gets
its own check automatically when the clone finishes: base model with vs
without passage retrieval on the test examples, judged, as a **retrieval
sweep** (the table above) — and the winning retrieval becomes the project's
setting, which the playground serves. The sibling's Eval summary reads
"Base model + your document passages vs the base model alone … Retrieval
served: top-3 + reranker ms-marco-MiniLM-L-6-v2 — chosen by the judge over
top-3 (0.61), top-5 (0.53), top-5 + reranker (0.66)".

In the playground, *"Who must retain a record of the use of personal
information, and what does that record form part of?"* gets the section 9
answer, cited from the passage it came from, with no fine-tuning.

## What is still open (said on camera)

- 3 of 19 answers are still wrong after the sweep — mostly a neighbouring
  section retrieved; chunk size, a stronger reranker and a larger candidate
  pool are untried.
- The answer key was written by a model from the same passages. It is a
  consistent ruler, not a lawyer's.
- One judge is one opinion. The judge's name is stamped on every result, so
  two numbers judged by different models are never compared silently.

## Reproduce

1. New project; upload the two statute HTML files; run cleaning
   (chunk size 900).
2. Training tab → **Build a Q&A assistant from these documents** (needs a
   local generation model — Ollama with a non-thinking chat model, or
   thinking turned off; gemma4 works).
3. Start the backend with a judge: `EVAL_JUDGE_BACKEND=ollama:gemma4:12b`
   (or any reachable local model; the first Ollama model is picked by
   default). The lift check judges automatically; **Re-run lift check**
   re-scores an older run.
4. Advanced evaluation → Auto-RAG comparison → **Run** on *Base model +
   document passages*; or on the CLI
   `python scripts/auto_rag_ab.py --project <id> --base-only --corpus documents --sweep-retrieval`.
5. Coach (Eval tab) → **Reroute to RAG**. The sibling's check and sweep
   run on their own; the playground answers from passages.

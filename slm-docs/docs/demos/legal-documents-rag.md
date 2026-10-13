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

This page is the written form of **walkthrough video 20 — "Fine-tuning
learned the words. Retrieval got the facts."** (6:41, captioned). Every
number below is a live measurement from the project it was recorded on;
nothing was tuned for the demo.

## Watch the walkthrough

<!-- Replace with the YouTube embed once published (docs-demo/videos/20-legal-rag-vs-sft/PUBLISH.md has the kit). -->
*YouTube link: pending publication.* Until then the video ships with the
repo's demo assets as `docs-demo/videos/20-legal-rag-vs-sft/final_captioned.mp4`
(rebuild: `build.py`, then `captions.py --burn`).

| time | scene |
|---|---|
| 0:00 | Two statutes, 467 passages, no Q&A |
| 0:32 | One job: documents → Q&A pairs → answer key → train → lift check |
| 1:34 | The gate that passed — F1 +150%, every seed |
| 2:12 | The gate that failed — the judge: 1 of 19 correct |
| 3:20 | Base model + retrieved passages: 8 of 19 |
| 4:14 | The Coach: retrieval beats the fine-tune → reroute to RAG |
| 4:42 | The RAG sibling and the retrieval sweep (12 of 19) |
| 5:16 | Live: a cited answer in the playground |
| 6:10 | What is still open |

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

## What a new user sees today (clean run, 2026-10-13)

The tables above are the recording's numbers. This is the same case run
again from an empty project with today's defaults — one click on the flow
card, the auto-picked local generation model (`ollama` → gemma4:latest on
this box), the judge from the backend env (`gemma4:12b`), base model
Qwen2.5-1.5B-Instruct. Today the flow **sizes itself to the decision**: it
generated from 150 passages so the test split has 46 rows, enough for a
0.15 judge-score gap to clear noise. Every step happened without choosing
anything:

| step | result |
|---|---|
| documents → Q&A flow (sized) | 467 passages, 150 used, 450 pairs + 150 answer-key rows, split 358 / 44 / 46, 26 min |
| pre-training gate (retrieval sweep, judged, 46 test examples) | the sweep could finally call a difference: **top-5 + reranker beat top-3 beyond noise** (0.53 vs 0.38; 14 rows better, 4 worse, +0.15, CI +0.02 to +0.29) and became the setting. Gate on it: **20 / 9 / 17**, 0.53 → *promising* (too many wrong to stop) → training went ahead with the note |
| fine-tune (run #45, judged on the same 46 rows) | **4 / 3 / 39**, 0.05 → 0.12, within noise (token F1 said 0.11 → 0.31) |
| Coach, Eval tab | "Retrieval over your documents beats the fine-tune (0.53 vs 0.12 judged)" → one-click reroute |
| RAG sibling (#26), its automatic sweep | kept top-5 + reranker (top-3 now *worse* beyond noise: 4 better / 14 worse); base alone 0.04 → with passages 0.53, 27 rows better, 2 worse |
| playground, sibling | the section 9 answer, cited "[1]", 17.9 s on first load |

*Before the sizing* (same day, 60 passages, 19 test rows): the gate read
5 / 3 / 11 (0.34) with plain top-3, every reranked arm was within noise, the
fine-tune judged 1 / 4 / 14, and the sweep kept BM25. The sizing didn't
change what is true — retrieval beats the fine-tune either way — it changed
what the platform could *prove*: on 19 rows the reranker's +0.15 was a
coin toss; on 46 it is a finding, and the project serves it.

What this run says plainly: the gate did **not** stop training (20 right of
46 with 17 wrong is not an assistant), so the flow trained and let the
judged comparison settle it, which it did on the same rows; and the
fine-tune on 358 generated pairs still learned the wording, not the facts
(4 of 46 right).

## What is still open (said on camera)

- 17 of 46 answers are still wrong with the best retrieval (3 of 19 in the
  recording) — mostly a neighbouring section retrieved. The staged sweep has
  since tried the reranker's candidate pool (50 → 0.50) and passages re-cut
  at 600 / 1200 chars (0.46 / 0.45) against the served top-5 + reranker
  (0.53): all within noise, none better, so it stayed. A stronger
  reranker (`BAAI/bge-reranker-v2-m3`, 25× the size) scored exactly the
  same 0.53 at top-5 (19 / 11 / 16; 3 rows better, 3 worse). Nothing tried
  gets past 0.6 on these 46 rows — the ceiling is not retrieval shape or
  reranker quality.
- **Retrieval or reading?** The judge now also says, per row, whether the
  retrieved passages held the answer. With the served retrieval it found it
  in **37 of 46** rows (80%); of the 17 wrong answers, retrieval **missed 7**
  and **found 10** that the model then misread — so the reader, not
  retrieval, owns most of the misses. Trying the sibling on
  **Qwen2.5-3B-Instruct** (same retrieval) did not change that: 0.53 again
  (19 / 11 / 16), with the wrong answers split 8 missed / 8 misread. Twice
  the reader bought nothing on these rows; what the "misread" rows have in
  common is a reference answer that demands a specific detail (a section
  number, a list) the model paraphrases past.
- **Reading the ten "misread" rows** (reference beside the passage the judge
  pointed at): four were the *key's* fault — a reference that misread its
  passage ("the Act does not apply to persons or organizations other than
  government institutions"), a question only answerable with the passage's
  own framing ("what does section 21 or 22 provide an indication of"), a
  reference that picked two duties from a list of four, a reference that
  began "the passage mentions…"; two were built on a sentence cut at a chunk
  boundary; four were genuine misreads. So the flow's evaluation question is
  now held to a **quotable** standard — self-contained, the passage's own
  complete statement as the answer, never on a cut-off fragment — with a
  deterministic gate behind the prompt. Re-running the flow on a fresh
  project (150 passages, no training): 144 answer-key rows kept, **6
  dropped** (5 pointed at the passage, 1 unsupported), and the kept answers
  are quotes ("agenda of Council or records recording deliberations or
  decisions of Council") rather than paraphrases. The judged numbers above
  are from the earlier key; a new run on the quotable key is the next
  measurement.
- The answer key was written by a model from the same passages. It is a
  consistent ruler, not a lawyer's.
- One judge is one opinion. The judge's name is stamped on every result, so
  two numbers judged by different models are never compared silently.

## Reproduce

1. New project; upload the two statute HTML files; run cleaning
   (chunk size 900). The tables use base model Qwen2.5-1.5B-Instruct (set
   it on the project, or keep the task type's default); the flow uses the
   first reachable local generation model unless you pick one.
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

# Evaluation — Narration

Status: **synced** with the actual narrated take produced by
`tts/generate_v10_narration.py` (Orpheus voice "leo") on 2026-05-20.

The **Python script** at
[tts/generate_v10_narration.py](../../../tts/generate_v10_narration.py)
is the **authoritative source** of the spoken text. This file mirrors
the same text plus stage directions / Playwright cues. Edit the
script first.

Total runtime: **1:17** (matches
`docs-demo/recordings/raw/10-evaluation-narrated.mp4`).
Section timings come from `tts/audio/v10-durations.json`.

Companion to:
[docs-demo/videos/10-evaluation/recording-plan.md](../../videos/10-evaluation/recording-plan.md).

Second runtime-dependent video. Scores the V09 trained checkpoint
against the 200-row support-faq gold set. Honest result: exact-
match=0, F1≈0.12 — the loop closes, the model is just too small to
hit the gate threshold.

---

## Pre-roll (not narrated)

Playwright logs in as admin, opens the **Demo · Support FAQ**
project, clicks the **Evaluation** tab. The spec asserts that
`v09-narrated-run` exists with status=completed before continuing.

## Section 1 — Cold open (0:00–0:12)

**On screen**: Evaluation tab; "Run at least one evaluation"
empty-state hint visible.

> "Now we score the model we just trained against the gold set.
> Two hundred rows, hand-labelled, never seen during training. The
> eval handler dispatches to question-and-answer mode because the
> task profile is instruction-following."

## Section 2 — Setup recap (0:12–0:29)

**On screen**: hold on the Evaluation tab.

> "Quick recap. Eval against gold is how we actually measure
> quality. The model generates an answer for each question in the
> held-out set, and the handler scores it against the expected
> answer. For this sample we get two headline numbers: exact match
> for the strict score, and token-level F1 as a more forgiving
> secondary."

## Section 3 — Kickoff (0:29–0:39)

**On screen**: spec fires
`POST /api/projects/<id>/evaluation/run-heldout` (non-blocking).
The eval handler begins loading the LoRA adapter and generating
predictions in the background.

> "Launching an eval run via the API. Held-out dataset is gold
> dev — twenty samples for this recording to keep it short, but
> the same call works against all two hundred."

## Section 4 — Watching (0:39–0:52)

**On screen**: spec awaits the eval promise; refresh + re-click
the Evaluation tab when the call returns.

> "Eval pipeline: load the trained checkpoint, run generation for
> each sample, score per sample, aggregate. On this hardware
> that's about twenty seconds for twenty samples — model load is
> one-shot, then it's roughly half a second per sample."

## Section 5 — Results (0:52–1:09)

**On screen**: Auto-Gate panel renders with **FAIL** (failed
required gates: min_exact_match, min_f1). Eval pack
`evalpack.general.default`, experiment #13 checked. Below: experiment
selector card with `v09-narrated-run` button.

> "Done. Exact match landed at zero — the model is too small to
> produce verbatim matches yet. Token-level F1 is in the low tens
> of percent, which says there's some overlap with the gold
> answers but the model is far from production quality. The point
> isn't the score. The point is the loop closed."

## Section 6 — Wrap (1:09–1:17)

**On screen**: hold on the completed Evaluation tab.

> "Eval result is now stored against the experiment. Next video
> compresses the trained adapter into a quantized artifact ready
> to serve."

---

## Things to **not** say

- Don't claim the model "passes" or "works." The Auto-Gate FAIL on
  screen would contradict you.
- Don't promise specific score numbers. The recording's f1=0.1217
  is real, but variance is high at 16 train rows + 20 eval samples.
  The narration says "low tens of percent" so a re-run with a
  slightly different score doesn't invalidate the take.
- Don't read literal metric keys aloud (`exact_match`, `pass_rate`,
  `min_f1`). Say them as English ("exact match", "pass rate",
  "minimum F1 gate").

## Optional technical notes (background; not spoken)

- The `instruction_sft` task profile dispatches to the `qa` handler
  in
  [backend/app/services/eval_task_handler_service.py](../../../backend/app/services/eval_task_handler_service.py)
  — this resolves Q9 from
  [10-open-questions.md](../../evidence/10-open-questions.md).
- `dataset_name="gold_dev"` resolves to `DatasetType.GOLD_DEV` via
  the alias map in `evaluation_service.py:_resolve_dataset_alias`.
- The eval result row is also visible at
  `GET /api/projects/<id>/evaluation/results/<experiment_id>` and
  feeds the Auto-Gate panel via the gate-resolution service.
- Inference uses HuggingFace transformers on CUDA with bf16 dtype
  — confirmed from the result's `metrics.inference` blob.
- The recording deliberately uses `exact_match` over `llm_judge` to
  keep the wall time short. For higher-quality scoring on SFT,
  `llm_judge` (which calls the local Ollama judge model) is the
  better choice; that variant lives in V10's follow-up if/when we
  decide to record it separately.

## What V11 picks up

Video 11 takes the V09 LoRA adapter, merges it into the base
model, and quantizes the merged weights to GGUF for Ollama serve.
The narration handoff at the end of V10 sets that up explicitly.

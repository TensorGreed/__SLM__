# Video 10 — Evaluation · Recording Plan

Status: **shipped 2026-05-20**. Second runtime-dependent video.
Scores the V09 trained checkpoint against the support-faq
`gold_dev` dataset (200 hand-labelled rows, never seen during
training) via the `/evaluation/run-heldout` endpoint. The eval
handler dispatches to QA mode because `task_profile=instruction_sft`,
and the recording captures real generation latency + real exact-
match / token-F1 numbers.

## Goal

Demonstrate the eval loop closing: a trained checkpoint scored
against held-out gold returns aggregate metrics + per-sample
predictions, the result is persisted, the Auto-Gate panel updates
to PASS or FAIL based on the configured thresholds. Honest result:
exact-match is 0 and F1 is in the low tens of percent — the loop
works, the model is just too small to score well.

## Audience

Intermediate, continuing from V09. Assumes the viewer just saw the
training loop complete and wants to know "how well did it do?"

## Final length

**1:17** (target was 4–6 min; tight narration came in well under).

## Prerequisites

- V09's training experiment (`v09-narrated-run`, status=completed)
  must still exist. Spec hard-fails with a clear error if it
  doesn't.
- Trained checkpoint on disk at
  `data/projects/<id>/experiments/<exp_id>/`.
- Ollama not strictly required for `exact_match` eval; would be
  required for `llm_judge` mode (which this recording does not use
  to keep wall time short).
- All runtime services from V09 still up: Celery, Redis, backend,
  frontend.

## Eval configuration used

| Field | Value | Why |
|---|---|---|
| `experiment_id` | resolved at runtime from `v09-narrated-run` | Skip-friendly lookup so the spec doesn't hard-code an id. |
| `dataset_name` | `gold_dev` | The 200-row hand-labelled gold set (vs `test` which is the 2-row held-out split). |
| `eval_type` | `exact_match` | Simplest scorer. Returns both `exact_match` and `f1` in the metrics blob anyway. `llm_judge` would be more realistic for SFT but adds Ollama latency. |
| `max_samples` | `20` | Wall time scales linearly. 20 fits inside the watching-section audio (~13s) plus model-load (~1s). |
| `max_new_tokens` | `64` | Support FAQ answers are short. Limits gen latency. |
| `temperature` | `0.0` | Deterministic generation; eval metrics need to be reproducible. |

## Exact starting state

1. Backend + frontend running.
2. Logged in as **admin**.
3. Seeded support-faq project opened.
4. V09 experiment exists with status=completed.
5. Evaluation tab loaded. May show 0 prior results for this
   experiment or 1 from a dry-run depending on lifecycle. The
   dry-run for V10 was cleaned up before recording.

## Recording arc (6 sections)

| # | Section | Audio (s) | What happens on screen |
|---|---|---:|---|
| 1 | Cold open | 11.61 | Evaluation tab loaded; "Run at least one evaluation" hint visible |
| 2 | Setup recap | 17.41 | Hold on eval tab |
| 3 | Kickoff | 9.56 | Spec fires `POST /evaluation/run-heldout` (non-blocking) |
| 4 | Watching | 13.48 | Spec awaits the eval promise; refresh + click eval tab |
| 5 | Results | 16.73 | Auto-Gate panel renders with PASS/FAIL + metrics; experiment selector shows v09-narrated-run |
| 6 | Wrap | 7.85 | Hand-off to Video 11 (compression + export) |

Total audio: **76.6s**. Final muxed video: **1:17**.

## Why API kickoff over UI

Same reasoning as V09: the eval form has a `/workbench` surface
with many controls, and the API path is the canonical entry point
used by Autopilot + scripts + CI. The spec POSTs to
`run-heldout` directly. The UI's polling then reflects the new
result without any form selectors needed.

## Result values from the recording

| Metric | Value | Notes |
|---|---|---|
| `exact_match` | 0.0 | No verbatim matches across 20 samples; expected for a 135M-param model with 16 train rows. |
| `f1` (token-level) | 0.1217 | Some token overlap with gold answers. |
| `pass_rate` | 0.1217 | Headline number the gate threshold compares against. |
| Auto-Gate | **FAIL** | Failed required gates: `min_exact_match`, `min_f1` (both well below threshold). |
| Eval pack | `evalpack.general.default` | Default platform pack; configurable per project. |
| Inference latency | ~620ms/sample | bf16, cuda, transformers backend. ~100 tokens/sec throughput. |

The FAIL gate is the **right** outcome to show. A green PASS on a
tiny model would either require gates so loose they're meaningless,
or fake numbers. The narration acknowledges the score and
emphasizes that the loop closing is the point.

## Answered open question

**Q9** — Which eval handler dispatches for `instruction_sft`? The
result's `details.handler_id` resolves to `"qa"`. Confirmed: the
QA handler in `eval_task_handler_service.py` is the one that runs
for support-faq.

## Things to not say

- Don't claim the model "works" or "passes" — it doesn't, and the
  Auto-Gate FAIL on screen contradicts it.
- Don't read literal endpoint paths or metric keys aloud (TTS
  mispronounces snake_case). Say "exact match" / "F1" / "pass
  rate" as English.
- Don't promise a specific score number — the recording's f1=0.12
  is real, but tiny dataset + tiny model gives variance. The
  narration says "low tens of percent" so the recording isn't
  invalidated by re-runs.

## Failure modes

| Symptom | Cause | Fix |
|---|---|---|
| `experiment_id` not found | V09 wasn't run or was deleted | Run `09-training-run-narrated.spec.ts` first |
| Eval timeout | Model load fails / disk full / dataset empty | Check `data/projects/<id>/experiments/<exp_id>/` and backend logs |
| Auto-Gate green | Either the gate thresholds are too loose or the model genuinely is good | Inspect the gate config under the project's `gate_policy` JSON column |
| `f1` significantly different from 0.12 | The trained checkpoint differs; expected if V09 ran with different config | Acceptable — the narration is hedged for this |

## Open questions parked for later

- **Q10** — Does cleaning ever actually remove duplicate rows, or
  only compute hashes? Not relevant for this video but still open.
- **Q9** — Answered (`qa` handler). Update `10-open-questions.md`
  separately if you want a docs-only commit.

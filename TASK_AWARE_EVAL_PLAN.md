# Task-Aware Evaluation Plan

A staged rollout for replacing the one-size-fits-all eval pipeline with a
per-task-profile handler architecture. Each phase is independently
shippable; the dispatcher is purely additive, so any phase you skip just
falls through to today's behavior.

## Why this exists

Today's eval pipeline:

1. Reads every dataset row as `(prompt, reference)`, regardless of task.
2. Feeds the bare input to the model (with the tokenizer's chat template
   wrapper from Phase 5.2 if available).
3. Scores with SQuAD-style EM/F1 (from Phase 5.2) or LLM judge.

That's correct for short-answer QA and adequate-but-wrong for everything
else. The smoking gun: the sentiment-classifier demo prompts the model
with bare review text, gets back paragraph-long product reviews, and
scores F1 ≈ 0 — because token overlap between rambling generation and
the literal string `"positive"` is hopeless. The system already knows
each dataset's `task_profile` (it's written into
`prepared/manifest.json` by the Phase 4.1 demo seeder and the dataset
prep flow), but no eval-time consumer reads it.

## Architectural shape

One service file: `app/services/eval_task_handler_service.py`.

Each task handler is a small object that answers two questions:

```python
class TaskHandler(Protocol):
    profile: str                    # e.g. "classification"

    def build_prompts(
        self,
        rows: list[dict],
        ctx: EvalContext,
    ) -> list[BuiltPrompt]: ...

    def score(
        self,
        predictions: list[Prediction],
        ctx: EvalContext,
    ) -> dict[str, float]: ...
```

A single dispatcher:

```python
def resolve_task_handler(task_profile: str | None) -> TaskHandler:
    return _REGISTRY.get(normalize(task_profile), GenericHandler())
```

**Strict rule**: `task_profile` is only ever read from
`prepared/manifest.json`. We do **not** sniff task shape from the row
data. The cost of declaring tasks explicitly is one field in a manifest
the system already writes; the benefit is that seq2seq datasets with
short dev samples never get auto-mistaken for classification.

Handlers register themselves; adding a new task type is one new file,
no edits to existing handlers, no global behavior change.

## Cross-cutting decisions (locked in once at 5.3.0)

- **Eval pack contract unchanged.** Eval packs still declare metric IDs
  (`accuracy`, `f1`, `bleu`, `rouge_l`, etc.) and gates. The task
  handler is responsible for producing the metric IDs the pack
  expects. The pack/gates layer doesn't care which handler ran.
- **Predictions UI works for every handler.** The "Sample predictions"
  card we shipped in 5.2a renders any `(prompt, formatted_prompt,
  reference, prediction)` quadruple. Handlers populate
  `formatted_prompt` so the user can always see what the model
  actually saw.
- **Backwards compat with `eval_type` enum.** `exact_match`, `f1`,
  `llm_judge` keep working — they map to the GenericHandler. New
  handlers expose their own metric IDs without changing the enum.
- **Missing task_profile → GenericHandler.** Datasets not tagged in
  their manifest get today's exact behavior. Zero regression risk for
  anything not explicitly opted in.
- **No row-shape sniffing.** See "Architectural shape" above.

---

## Phase 5.3.0 — Dispatcher + GenericHandler

**Goal**: extract today's behavior into the new shape with zero
externally-visible change. Sets the foundation so every subsequent
phase is a pure addition.

### User stories
- *As a developer on this codebase*, I want the eval pipeline split
  into a dispatcher + handler so I can add a new task in one file
  without touching anything else.
- *As any existing user*, I want my current evals to score identically
  before and after this phase ships.

### Work
- Create `eval_task_handler_service.py` with `TaskHandler` protocol,
  `EvalContext`, `BuiltPrompt`, `Prediction` dataclasses.
- Extract today's `_extract_prompt_and_reference` →
  `GenericHandler.build_prompts`.
- Extract today's `exact_match` / `f1_score` / judge path →
  `GenericHandler.score`.
- Wire the dispatcher into `run_heldout_evaluation`. Read
  `task_profile` from `prepared/manifest.json` if present.
- Surface `task_profile_resolved` + `handler_id` in
  `result.details.inference` so the UI / logs can confirm which
  handler ran.

### Tests
- Regression: every existing eval test still passes byte-for-byte.
- New: `test_phase89_eval_dispatcher.py` covers `resolve_task_handler`
  fallback when profile is missing, unknown, malformed.

### Out of scope
- Any actual behavior change. This phase is pure refactor.

---

## Phase 5.3.1 — Classification handler

**Goal**: make the sentiment-classifier demo actually score correctly.
This is the phase that addresses the screenshot you sent.

### User stories
- *As a newbie running the sentiment-classifier demo*, I want eval to
  ask the model to output **one** of `{positive, neutral, negative}`,
  not generate a product review.
- *As an ML engineer with a 3-class sentiment model*, I want per-class
  precision / recall / F1 so I can see whether one class is dragging
  the score down.
- *As a researcher with a 50-intent classifier*, I want the candidate
  list omitted from the prompt when there are too many labels (it
  doesn't fit), and the parser still resolves the model's output to a
  label.

### Work
- New `ClassificationHandler`.
- **Candidate set source**: read `labels` field from
  `prepared/manifest.json` if present (we'll extend the dataset prep
  service to write it). If absent, derive from the eval set's
  references — dedupe + sort + cap at 30.
- **Prompt template**:
  ```
  Classify the following as one of: {labels}.
  Text: {input}
  Label:
  ```
  Apply the tokenizer's chat template *over* this composed prompt
  (Phase 5.2 still runs).
- **Generation hint**: cap `max_new_tokens` at `max(16,
  longest_label_token_len * 2)` so the model can't ramble.
- **Output parser**: lowercase + strip, then check each candidate
  label as a substring against the output; pick the earliest match.
  Tie-break by longest label (so `"very_positive"` beats
  `"positive"`).
- **Metrics produced**: `accuracy`, `macro_f1`, `per_class_precision`,
  `per_class_recall`, `per_class_f1`, `unparseable_rate` (rows where
  no label could be extracted), `confusion_matrix` (small, capped).
- Update demo seeder (Phase 4.1) so the prepared manifest carries
  `labels: ["positive", "neutral", "negative"]` for classification
  datasets.

### Tests
- `test_phase90_eval_classification_handler.py`:
  - Candidate set built from references; cap at 30.
  - Prompt contains label list when ≤ 30 labels.
  - Prompt omits label list when > 30 labels but still asks for a
    label.
  - Parser picks earliest-match candidate; ties resolved by longest
    label.
  - `unparseable_rate` increments when output has no candidate match.
  - End-to-end: mock inference returning labels → accuracy reported
    correctly + confusion matrix.

### Out of scope
- Multi-label classification (separate phase, 5.3.1b).
- Hierarchical / nested labels.

---

## Phase 5.3.2 — QA / instruction-following handler

**Goal**: tighten short-answer QA scoring without changing seq2seq or
classification.

### User stories
- *As a newbie running the support-FAQ demo*, I want my model's answer
  compared to the gold answer with SQuAD-style normalization (today's
  Phase 5.2 behavior — preserve it).
- *As an ML engineer*, I want the option to add LLM-as-judge scoring
  alongside EM/F1 without re-running generation.
- *As an ML engineer doing chain-of-thought QA*, I want the scorer to
  match the final answer span rather than the whole reasoning string
  (extract from `"... Final answer: X"` patterns).

### Work
- New `QAHandler` (covers `task_profile in {qa, instruction_sft,
  chat_sft, language_modeling}`).
- Default: today's chat-template wrap + SQuAD EM/F1 (already shipped
  in 5.2). This phase mostly just registers QAHandler so the
  dispatcher routes correctly.
- Add answer-span extractor for CoT-style outputs: regex for `Final
  answer:`, `Answer:`, `Therefore:`, fall back to whole output.
- Metrics produced: `exact_match`, `f1`, `answer_span_extracted_rate`.

### Tests
- `test_phase91_eval_qa_handler.py`:
  - Identical EM/F1 to today's GenericHandler for clean Q/A rows.
  - Span extractor pulls `"Final answer: X"` correctly.
  - Falls through when no span marker present.

### Out of scope
- Multi-hop QA evaluation (later, with intermediate-step scoring).
- Tool-calling QA (covered by 5.3.7).

---

## Phase 5.3.3 — Seq2seq handler (translation, summarization, paraphrase)

**Goal**: report metrics seq2seq researchers actually need.

### User stories
- *As a researcher running an EN→FR translation eval*, I want **BLEU**
  and **chrF** scores so I can compare against public benchmarks.
- *As a developer running summarization*, I want **ROUGE-1 / ROUGE-2 /
  ROUGE-L** as the headline metrics.
- *As an ML engineer*, I want F1/EM still reported alongside so my
  existing gates don't break.

### Work
- New `Seq2SeqHandler`.
- **Sub-task detection** from manifest hint: `subtask` field can be
  `"translation"`, `"summarization"`, `"paraphrase"`. Defaults to
  `"summarization"` if missing.
- **Prompt template**:
  - translation: `Translate to {tgt_lang}: {src}` (read `tgt_lang`
    from manifest)
  - summarization: `Summarize the following:\n{src}\nSummary:`
  - paraphrase: `Paraphrase the following: {src}`
- **Generation hint**: `max_new_tokens` ~ 1.5× the longest gold
  reference, capped at 512.
- **Metrics produced**:
  - Translation: `bleu`, `chrf`, `f1` (for compat)
  - Summarization: `rouge_1`, `rouge_2`, `rouge_l`, `f1` (for compat)
  - Both: `length_ratio` (predicted / reference token count)
- **Dependency**: `sacrebleu` for BLEU + chrF; `rouge_score` for
  ROUGE. Both are MIT-licensed, small, pure-Python.

### Tests
- `test_phase92_eval_seq2seq_handler.py`:
  - BLEU computed correctly on a tiny pair (compare to sacrebleu
    reference).
  - ROUGE-L computed correctly.
  - Sub-task detection from manifest.
  - F1 still reported alongside (gate compat).

### Out of scope
- Multi-reference BLEU (most evals are single-reference; can add
  later).
- COMET / BLEURT (model-based metrics — heavy; add only if asked).

---

## Phase 5.3.4 — Structured extraction handler

**Goal**: score the model's ability to produce well-formed JSON with
the right field values.

### User stories
- *As a developer building invoice extraction*, I want
  **field-level F1** for each declared field (invoice_no, total,
  date), not a string-vs-string comparison of the whole JSON blob.
- *As an ML engineer*, I want **JSON validity rate** as a separate
  metric — a model that emits malformed JSON 30% of the time is
  unshippable regardless of field accuracy.
- *As a developer*, I want **schema compliance** scored separately
  from value accuracy (a row where every field is present but the
  values are wrong is a different bug from a row missing fields).

### Work
- New `StructuredExtractionHandler`.
- **Schema source**: read `output_schema` field from
  `prepared/manifest.json` (JSON Schema subset). If absent, derive
  field set from the first 20 reference rows.
- **Prompt template**: `Extract the following fields as JSON:
  {fields}.\nInput: {x}\nOutput:`
- **Output parser**: extract first balanced `{...}` block from the
  model output; try `json.loads`; record validity per row.
- **Metrics produced**:
  - `json_validity_rate`
  - `schema_compliance_rate` (all required fields present)
  - `field_exact_match_rate` (per field, averaged)
  - `field_f1` (per field, averaged across rows that have that field
    in both prediction and reference)
  - `overall_em` (whole-blob exact match — for gate compat)

### Tests
- `test_phase93_eval_structured_extraction_handler.py`:
  - JSON validity: malformed output → 0, valid → 1.
  - Schema compliance: missing field → 0.
  - Field-level F1: per-field scoring, robust to missing fields.
  - Code-fence stripping: model outputting ` ```json {...} ``` ` is
    parsed.

### Out of scope
- Free-form JSON schemas (only declared / sampled schemas).
- Nested objects beyond one level (later phase).

---

## Phase 5.3.4b — Span-set scoring mode (PII / NER / span-extraction)

**Goal**: production-grade entity-level evaluation for tasks whose
output is a list of typed spans `[{type, start, end, text}, ...]` —
PII / PCI, medical NER, legal clause extraction, financial entity
extraction, generic NER. Critically **not a new handler class** — it
lives inside `StructuredExtractionHandler` as a second scoring mode,
so the handler registry / dispatcher stay stable and BrewSLM doesn't
drift toward one-task-per-handler proliferation.

### User stories
- *As a compliance officer evaluating a PII detector*, I want
  per-class recall ("99.7% credit_card, 99.5% SSN, 98% email")
  rather than a single overall F1, because that's what shippable
  PII claims look like and the only way to gate the risky classes.
- *As an ML engineer tuning a span detector*, I want per-row P/R/F1
  + lists of which entities matched / were missed / were
  hallucinated, so I can spot where the model is weak instead of
  flying blind on a coarse exact-match.
- *As a developer integrating with LlamaFirewall*, I want strict
  span matching (same type + same offsets) since redaction breaks
  when boundaries are off — a "John" prediction for a gold "John
  Smith Jr." span shouldn't score the same as a full match.

### Work
- New scoring mode `span_set` inside `StructuredExtractionHandler`,
  triggered by `manifest.output_schema.scoring_mode == "span_set"`.
  Default stays `field_match` so invoice-style extraction is
  byte-for-byte unchanged.
- Strict matching: TP requires identical `(type, start, end)`.
  Counter semantics so duplicates count correctly (model has to
  find both emails if both are gold).
- Per-class P/R/F1 reported as `per_class: {type: {p, r, f1,
  support, tp, fp, fn}}`. Micro aggregate as `precision/recall/f1`,
  macro as `precision_macro/recall_macro/f1_macro`.
- `exact_match` legacy alias = row-level whole-set EM (every
  predicted entity matched + no entities missed). Gate compat.
- Per-row enrichment lands `row_matched_entities /
  row_missed_entities / row_hallucinated_entities` + per-row
  `row_precision / row_recall / row_f1` so the UI can render an
  entity-by-entity breakdown.
- UI: when `scoring_mode == "span_set"`, the Sample Predictions
  card swaps the per-field comparison for inline "X matched · Y
  missed · Z hallucinated" counts plus a "Show entity-by-entity
  breakdown" disclosure listing every TP / FN / FP entity with
  type / text / offsets.
- PII demo manifest sets `scoring_mode: span_set` so the demo
  immediately benefits.

### Tests
- 19 backend tests (test_phase94) — scoring mode dispatch,
  strict matching (perfect / partial / type-mismatch /
  boundary-mismatch / duplicate semantics), per-class breakdown,
  macro aggregates, per-row enrichment, edge cases
  (empty/empty trivially correct, empty pred non-empty gold,
  malformed entity payload, unparseable JSON, empty list).
- 4 frontend tests (EvalPanel.spanset.test.tsx) — matched/missed/
  hallucinated counts inline, entity-by-entity disclosure, missed
  + hallucinated badges, field_match mode regression guard.

### Out of scope
- Partial-credit matching (token-IoU for boundary errors). Strict
  is the load-bearing signal for compliance; partial scoring lands
  in a follow-up if a real use case wants it.
- Bipartite optimal matching for type-overlap mode. Current
  matching is exact-key only.

---

## Phase 5.3.5 — RAG / grounded QA handler

**Goal**: score grounded QA models on **answer quality** + **faithfulness
to context**.

### User stories
- *As a developer building a grounded support bot*, I want a
  **faithfulness** score — does the model's answer cite tokens that
  actually appear in the retrieved context? — so I can catch
  hallucination directly.
- *As an ML engineer*, I want **context recall** (how many gold-answer
  tokens are present in the context — a sanity check on the
  retriever, not the model).
- *As an ML engineer*, I want SQuAD EM/F1 on the answer span on top of
  the grounding metrics.

### Work (shipped)
- New `RAGHandler` covers `task_profile in {rag_qa, rag,
  grounded_qa}`.
- **Row shape**: reads `context` (or `passage` / `document` /
  `evidence` / `retrieved_context`) alongside `question` and
  `answer`. Falls back to plain QA when no context field is present
  — same project can mix context-bearing and context-less rows.
- **Prompt template**:
  ```
  Answer the question using only the context. If the context does
  not contain the answer, say you don't know.
  Context: {context}
  Question: {question}
  Answer:
  ```
- **Generation cap**: 64-token floor, 256-token hardcap. Grounded
  answers should be short; long answers usually mean the model
  lost the question.
- **Metrics produced**:
  - `exact_match`, `f1` — SQuAD-style on the answer span. Gate
    compat preserved.
  - `faithfulness_rate` — fraction of context-bearing rows scoring
    above the 0.7 token-grounding threshold. Binary at-threshold
    rate for gates.
  - `faithfulness_score_mean` — continuous mean for monitoring.
  - `context_recall_mean` — retriever-side diagnostic. Token
    overlap of the gold answer with the context.
  - `unsupported_token_rate_mean` — mean fraction of prediction
    tokens NOT in the context. Catches the "London is the capital
    of France" case where most tokens are grounded but the
    critical wrong one is not.
  - `grounded_rows` / `rows_with_context` — denominators for the
    rate metrics.
- Per-row enrichment lands `rag_faithfulness`, `rag_context_recall`,
  `rag_unsupported_rate`, `rag_is_faithful`, `rag_context` on each
  prediction so the UI can render the inline surface.
- UI: Sample Predictions card grows a RAG inline surface beneath
  each prediction — green "Faithful (1.00)" or red "Hallucinated
  (0.40)" badge, "context covers gold: X%" inline diagnostic, red
  "unsupported tokens: X%" when > 0, and a "Show retrieved
  context" disclosure with the context the model was given.

### Tests (shipped)
- 23 backend tests in `test_phase95_eval_rag_handler.py` cover
  dispatcher routing (rag_qa / rag / grounded_qa aliases all route
  here, other profiles unaffected), prompt assembly (context
  included, alternative field names, no-context fallback),
  generation cap, faithfulness (perfect / partial / fully
  extraneous / empty pred trivially faithful), context recall as
  retriever-side signal, per-row enrichment, mixed datasets
  (context-bearing + context-less rows in same eval), and a
  full build_prompts → score pipeline.
- 5 frontend tests in `EvalPanel.rag.test.tsx` lock in the new
  UI: green Faithful badge, red Hallucinated badge with
  unsupported-tokens note, "Show retrieved context" disclosure,
  hide-unsupported-when-zero, regression guard that non-RAG runs
  don't accidentally render the surface.

### Out of scope
- Retriever evaluation (separate concern, separate dataset
  structure).
- Reference-free faithfulness via NLI (heavy; add later if the
  token-overlap heuristic misses too many cases).
- Multi-hop grounding (answer requires combining two passages).
  Current scoring is single-context-blob only.

---

## Phase 5.3.6 — Alignment / preference handler (shipped)

**Goal**: score DPO/ORPO-trained models on preference adherence.

### What landed
- `AlignmentHandler` registered for `dpo` / `orpo` / `alignment` /
  `preference`.
- Row shape: `prompt`, `chosen`, `rejected` (with aliases
  preferred / dispreferred / accepted / rejected / response_chosen /
  response_rejected). Reference = chosen (legacy SQuAD EM/F1
  preserved against the preferred completion for gate compat).
- Scoring mode: **similarity-based proxy** — token-level F1 of the
  model's output against both completions. Row is "preference
  correct" when chosen_sim > rejected_sim. Cheap, deterministic,
  works with the existing generation pipeline. Documented as a
  proxy for judge-based win-rate; heavier judge mode and log-prob-
  margin mode are clean follow-ups (additive scoring modes inside
  this handler).
- Metrics: `preference_accuracy`, `mean_alignment_margin`,
  `chosen_alignment_mean`, `rejected_alignment_mean`, `exact_match`,
  `f1` (legacy), `rows_with_pair`.
- Per-row enrichment: `alignment_chosen_sim` /
  `alignment_rejected_sim` / `alignment_margin` /
  `alignment_preference_correct` for the UI's badge + disclosure.
- UI: Sample Predictions card grows a green "Preferred chosen" /
  red "Preferred rejected" badge + inline similarity values +
  "Show chosen vs rejected completions" disclosure rendering both
  completions side-by-side.
- 18 backend tests + 4 frontend tests pin the contract.

### Out of scope (clean follow-ups)
- Judge-based win-rate as an opt-in scoring mode (heavier — needs
  a judge model wired in).
- Log-prob-margin mode (heaviest — needs the inference path to
  expose per-completion logprobs).
- KL-divergence to reference policy.
- Constitutional AI / RLAIF.

---

## Phase 5.3.7 — Multimodal handlers (shipped)

**Goal**: route multimodal evals through dedicated handlers that
produce modality-appropriate metrics.

### What landed
- `VisionLanguageHandler` registered for `vision_language` /
  `image_captioning` / `vqa`.
- `AudioTranscriptHandler` registered for `audio_transcript` /
  `audio_transcription` / `speech_to_text`.
- Sub-task dispatch via `manifest.subtask`:
  - Vision-language: `captioning` (default) or `vqa`.
  - Audio: `transcription` (default) or `audio_qa`.
- Captioning scoring: BLEU-4 (via `sacrebleu`) + ROUGE-L (via
  `rouge_score`) + `length_ratio` + legacy EM/F1.
- VQA scoring: SQuAD EM/F1.
- Transcription scoring: WER + CER (via `jiwer` — new dep, pure
  Python, MIT) + legacy EM/F1.
- Audio QA scoring: SQuAD EM/F1.
- Both handlers carry image_path / audio_path through the prompt as
  `<image:path>` / `<audio:path>` tokens so plain-text inference at
  least has a paper trail; real multimodal runtimes pass the
  modality alongside.
- Per-row enrichment writes `vl_subtask` / `audio_subtask` plus
  `row_exact_match` / `row_f1` so the existing Status badge lights
  up unchanged. No new UI block — multimodal users care about the
  headline metrics (BLEU / ROUGE-L / WER / CER) which surface in
  the Metric History panel.
- 19 backend tests pin the contract.

### Out of scope (clean follow-ups)
- Real multimodal inference (handler is correct on any caller using
  a multimodal-aware runtime, but the existing transformers /
  llama_cpp inference paths are text-only).
- CIDEr / SPICE caption metrics (Java dependency in
  `pycocoevalcap`).
- Video / streaming modalities.
- Model-based caption quality (CLIPScore).

---

## Phase 5.3.8 — Safety integration (shipped)

**Goal**: bring the existing `SAFETY_PROMPTS` flow under the same
handler arch so safety eval is just another `task_profile`.

### What landed
- `SafetyHandler` registered for `safety` / `refusal`.
- Row shape: `prompt` + optional `test_type` (prompt_injection /
  secret_extraction / pii_regurgitation / jailbreak /
  unknown_answer / unknown). Defaults to `unknown` when absent.
- Reuses the existing `evaluate_safety_response` keyword heuristic
  unchanged (refusal-phrase detection). Drop-in replacement for the
  legacy `eval_type=safety` flow — accepts both `prediction` and
  `response` field names.
- Metrics: `pass_rate` (overall), `per_test_type` dict with
  per-category P/R/totals, plus flat top-level
  `{test_type}_pass_rate` keys (e.g. `prompt_injection_pass_rate`,
  `jailbreak_pass_rate`) so eval-pack gates can key on a single
  metric ID without dict-path lookups.
- Per-row enrichment: `safety_passed`, `safety_reason`,
  `safety_test_type`, plus `row_exact_match` / `row_f1` aliases
  that feed the Status badge (green for refused, red for complied).
- Legacy aliases: `exact_match` / `f1` = `pass_rate` for gate
  compat.
- 14 backend tests pin the contract.

### Out of scope (clean follow-ups)
- Domain-specific safety (medical, legal) — separate handlers per
  domain pack.
- Judge-based safety classification (more nuanced than the keyword
  heuristic) — would land as an opt-in scoring mode the way
  AlignmentHandler will gain judge mode.

---

## Phase 5.3.9 — Generic fallback handler (formalize)

Already exists from 5.3.0. This phase is just where we document the
guarantee: any `task_profile` not in the registry routes here, with
today's behavior preserved. Nothing to ship.

---

## Cross-cutting work (split across phases as needed)

### Dataset prep manifest extensions
The prepared manifest grows new optional fields. None of them break
existing manifests:
- `labels: [str]` — classification candidate set.
- `output_schema: JSONSchema` — structured extraction schema.
- `subtask: "translation" | "summarization" | "paraphrase"` —
  seq2seq sub-task.
- `tgt_lang: str` — translation target language.

Demo seeder (Phase 4.1) populates these for the demos that need
them.

### UI surfacing
- The "Sample predictions" card from 5.2a renders unchanged for every
  handler.
- Add a "Handler" badge next to the eval result row showing which
  task handler ran (or "generic" if none matched).
- The DatasetFitCard from 5.1 stays accurate — its task→shape mapping
  comes from the same registry.

### Eval pack alignment
- `evalpack.classification.default` should list classification
  metrics (`accuracy`, `macro_f1`) as its expected metric IDs.
- `evalpack.translation.default` (new) lists `bleu`, `chrf`.
- `evalpack.summarization.default` (new) lists `rouge_*`.
- `evalpack.rag.default` (new) lists faithfulness metrics.

### Documentation
- Each handler gets a docs page under
  `docs/concepts/evaluation/handlers/{task_profile}.md`.
- Each handler appears in the Glossary (`docs/concepts/glossary.md`)
  with its metric definitions.

---

## Suggested rollout order

| Order | Phase | User pain | Effort |
|------:|-------|-----------|--------|
| 1 | 5.3.0 dispatcher | dev-internal, foundation | small |
| 2 | 5.3.1 classification | **breaking the demo right now** | medium |
| 3 | 5.3.3 seq2seq | researchers can't compare to benchmarks | medium |
| 4 | 5.3.2 QA refinement | small lift on current behavior | small |
| 5 | 5.3.4 structured extraction | developers building extractors | medium |
| 6 | 5.3.5 RAG | grounded-bot developers | medium |
| 7 | 5.3.7 multimodal | depends on multimodal adoption | medium |
| 8 | 5.3.6 alignment | researcher-only | medium |
| 9 | 5.3.8 safety | pre-prod check, but separate flow exists | small |

5.3.0 has to ship first (it's the foundation). After that, each phase
is independent — pick whichever matches the next user we want to
unblock.

---

## Open questions

1. **Manifest format for `labels` in classification**: should it be
   `["positive", "neutral", "negative"]` (flat list) or
   `{"positive": "good outcome", ...}` (with descriptions)? Flat list
   for now; revisit when we want richer prompts.
2. **Single eval result, multiple handlers?** Sometimes a dataset
   genuinely fits two profiles (QA + RAG). Today's plan: one handler
   per eval run, declared in the manifest. Multi-handler eval is a
   later phase if anyone asks for it.
3. **Judge sharing**: alignment, safety, and structured-extraction
   handlers may all want an LLM judge. Should the judge be a separate
   service the handlers call into, or inline per-handler? Lean
   separate-service so we can swap judge models centrally.
4. **Per-row sampling cap.** Today's eval runs the whole `test`
   split. For long seq2seq runs that's expensive. Consider a
   dataset-prep-time `eval_sample_cap` field. Out of scope for 5.3
   but flag now.

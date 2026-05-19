# Support FAQ Pipeline — Narration Skeleton

Status: ready for first take of the *inspect-only* path. Real
training run is a separate video (Video 09).

Target length: 9–12 minutes (≈1500 words).

Companion to:
`docs-demo/videos/03-support-faq-pipeline/recording-plan.md`.

---

## Cold open (0:00–0:30)

> "Welcome to the Support FAQ pipeline walkthrough. We're going to
> take the simplest of the three official samples — twenty customer
> tickets with hand-written answers — and walk it through every
> pipeline tab that actually does something on a seeded demo. By the
> end you'll know what each tab is for, which surfaces work without
> any external runtime, and where you'd need to wire up a teacher
> model or a training runtime to keep going."

## Section 1 — Recap + seed (0:30–1:30)

> "Quick recap from the quickstart video. We log in with the local
> API key, we land on the project list, and we click the Demo Support
> FAQ tile. That POSTs to the demo-projects endpoint, the backend
> copies twenty rows of `tickets.csv` into the project's raw data,
> imports 200 gold rows, and pre-builds an SFT split. Twenty seconds
> later we're on the Data tab."

## Section 2 — Data Tab + raw rows (1:30–3:00)

**Action**: scroll documents list, expand one row.

> "Each document on this list is one source ticket. The seeder turned
> each CSV row into a RawDocument record. There's twenty of them
> because there are twenty CSV rows.
> 
> Expand a row…"

**Beat**.

> "…and you see the shape: a `question` and an `answer`. This is
> what the model needs to learn — the agent's writing style for
> these specific questions. Imagine pasting *thousands* of resolved
> tickets here and you've got the dataset for a real support
> assistant."

## Section 3 — Cleaning tab (3:00–4:00)

**Action**: switch to Cleaning.

> "Cleaning. Skip this for the support-faq sample because the source
> is already small and clean. But this is where you'd run chunking,
> regex PII redaction, toxicity masking, and quality scoring on a
> messy real-world corpus.
> 
> Take note of the PII redaction options. We'll talk about how those
> differ from the PII Detector sample in Video 06 — same word, two
> completely different features."

## Section 4 — Gold Set tab (4:00–5:00)

**Action**: switch to Gold Set.

> "Gold Set. Two hundred entries. Locked. This is *evaluation* data,
> not training data. The model never trains against gold — it gets
> measured against it.
> 
> Each gold row carries a `question`, an `expected` answer, and a
> `rationale` explaining what the row is testing. The eval handler
> walks the entire 200-row set after training and tells you what
> fraction the model got right."

## Section 5 — Synthetic tab (5:00–6:00)

**Action**: switch to Synthetic.

> "Synthetic. This is the lever you'd pull to scale 20 source rows
> into 2000 training rows. The synthetic generator runs a teacher
> LLM — Ollama locally, or any OpenAI-compatible endpoint — over
> your cleaned chunks, asking the teacher to generate
> question/answer pairs that fit the same style.
> 
> See the warning banner: `TEACHER_MODEL_API_KEY` is missing. The
> generator won't run without one or without the demo-fallback flag.
> We're not going to fix that right now — Video 04 is the full
> synthetic-generation walkthrough."

## Section 6 — Dataset Prep tab (6:00–7:30)

**Action**: switch to Dataset Prep; show Schema Profile.

> "Dataset Prep. This is where the contract gets made. The adapter
> applied here — `qa-pair` — is the transform that turns each row
> into the `{question, answer}` shape the trainer expects.
> 
> Notice the prepared-manifest panel: 16 train rows, 2 validation, 2
> test. That's the deterministic 70-15-15 split with a 2-row floor
> on val and test. For larger corpora you'd see the floor disappear,
> but with 20 source rows we get exactly this."

## Section 7 — Tokenization tab (7:30–8:30)

**Action**: switch to Tokenization.

> "Tokenization runs a tokenizer over your prepared splits and tells
> you the length distribution — how many tokens per row, what max
> sequence length you'd want to budget. The actual analysis needs
> the `transformers` library and a tokenizer download, which is its
> own setup. For the demo we'll skip running it."

## Section 8 — Training Tab + Training Config (8:30–10:00)

**Action**: switch to Training. Then click into Training Config.

> "Training tab. 'No experiments yet.' Normal.
> 
> Let me jump into the Training Config page — the dedicated config
> surface lives at `/project/<id>/training-config`."

**Beat** as page loads.

> "There's an Essentials view by default that gives you the
> launch-critical controls — base model, training mode, epochs,
> batch size, learning rate. If you flip to Advanced mode (toggle in
> the page header) you unlock the Power Tools tab with the PEFT
> controls: LoRA rank, alpha, target modules, optimizer choice.
> 
> Default LoRA rank is 8, target modules are `q_proj, v_proj`. The
> PII docs page recommend bumping to rank 16 and all four attention
> projections for span tasks — that detail is in Video 06."

## Section 9 — Eval tab + wrap (10:00–11:00)

**Action**: switch to Eval.

> "Evaluation tab. Empty until we have an experiment to evaluate.
> But this is where the per-class metrics would land — for the
> Support FAQ sample's `instruction_sft` task profile, the eval
> handler dispatch is still an open question in our evidence — see
> open question 9 in `10-open-questions.md`.
> 
> What we *do* know: when there's a completed experiment, this
> surface shows accuracy, F1, gates pass/fail, and the sample
> predictions card with side-by-side prompt / expected / model
> output."

## Wrap (11:00–11:30)

> "That's the Support FAQ pipeline. We touched ten tabs without
> launching a single runtime-heavy job. Next video walks the same
> shape for the PII Detector sample — where we'll actually
> *distinguish* the two PII features. After that, sentiment
> classification. Then we get into training runs."

---

## Things to **not** say

- Don't say "we trained a model" — we didn't.
- Don't say "the demo has 6 gold rows" — that's the stale manifest
  prose. Say 200.
- Don't say cleaning automatically removes duplicates — it computes
  hashes but row-removal is unverified (open Q10).

## Optional technical notes (cut for beginner audience)

- The `prepared-manifest` API at
  `GET /api/projects/<id>/prepared-manifest` is the headline endpoint
  for this walkthrough. It returns adapter id, task profile, field
  mapping, output schema, and the train/val/test counts in one shot.
- The `qa-pair` adapter is one of eight registered data adapters; see
  `backend/app/services/data_adapter_service.py`.
- The deterministic split uses `random.seed(42)` by default in the
  seeder; manual splits via `POST /api/projects/<id>/dataset/split`
  can override.

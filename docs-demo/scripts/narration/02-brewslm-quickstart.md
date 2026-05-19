# BrewSLM Quickstart — Narration Skeleton

Status: **ready** for first take. Every product claim below has been
verified by the 2026-05-19 selector-discovery pass. See
`docs-demo/videos/02-brewslm-quickstart/recording-plan.md` for the
matching screen actions.

Target length: 5–7 minutes (≈800–1000 words spoken).

---

## Cold open (0:00–0:20)

> "We're going to take a fresh local BrewSLM install and get from
> nothing to a fully-seeded demo project, in under five minutes.
> No training, no data prep — just login, click, and inspect.
> By the end of this video you'll know which surfaces actually work
> on a seeded demo, and which ones are next-video material."

## Section 1 — Login (0:20–1:10)

**On screen**: the `/login` page after `http://localhost:5173/login`.

> "BrewSLM auth is on by default. The login form takes any username —
> I'll use `demo` — and a password that matches the `API_KEY`
> environment variable in `backend/.env`.
> 
> The default value out of the box is `sk-mock-admin-key`. That's
> the local-development token. In production you'd set this to
> something serious; for the demo it's fine."

**Action**: type `demo`, type `sk-mock-admin-key`, click Sign in.

> "And we're in."

**Beat (~1 second)** as the project list loads.

## Section 2 — Project List + Demo Tiles (1:10–2:10)

**On screen**: project list with three demo tiles visible.

> "This is the project list. If you've used BrewSLM before, your
> previous projects show up here. For this video we're starting
> fresh.
> 
> Up top: three official demo tiles. These are the only official
> starter templates in the repo. Don't expect a marketplace — there
> isn't one. The three are: Support FAQ, PII / PCI Detector, and
> Sentiment Classifier.
> 
> Each tile is backed by a folder under `backend/data/demo_samples/`
> with a manifest, a source CSV, and a 200-row gold JSONL. Clicking
> a tile seeds a complete project — raw data imported, gold
> imported, train/val/test splits already written."

**Pause** ~2 seconds so the viewer can read the three tile names.

## Section 3 — Seed Support-FAQ (2:10–3:30)

**On screen**: tile-click + transition to `/project/<id>/pipeline/data`.

> "I'll click Support FAQ — it's the simplest of the three.
> Behind the scenes the browser fires a POST to
> `/api/demo-projects/support-faq`. The backend reads the manifest,
> copies the source CSV into the project's raw data, creates 20 raw
> documents, imports 200 gold rows from the gold JSONL, and pre-writes
> a 16-2-2 train-val-test split.
> 
> A few seconds later, we land on the Data tab of the new project."

**Beat** for the page to fully load.

> "Notice the pipeline-status badge. It already shows the project at
> training stage, 60% complete. That's because the *seed* already did
> the upstream work for us. We didn't train anything yet — we just
> got handed a ready-to-train project."

## Section 4 — Pipeline Tab Tour (3:30–5:30)

**On screen**: click through Cleaning → Gold Set → Dataset Prep →
Training → back to Data.

> "Let's walk the pipeline tabs. There are ten of them, and they go
> left-to-right in the order you'd touch them in a real
> end-to-end training run."

**Click Cleaning.**

> "Cleaning. This is where you'd chunk text, redact regex PII as
> `[REDACTED_TYPE]`, and run a quality score. Nothing's running here —
> the seed didn't need it because the support-faq corpus is already
> small and clean."

**Click Gold Set.**

> "Gold Set. Two hundred entries, locked. These are the
> evaluation ground truth. Notice the count — the manifest text says
> six, but the actual file has 200. The file wins; treat manifest
> prose as advisory."

**Click Dataset Prep.**

> "Dataset Prep. You can see the adapter applied — qa-pair — and the
> field mapping that turns each row into a question/answer pair.
> The splits are already written: 16 train, 2 val, 2 test."

**Click Training.**

> "Training tab. 'No experiments yet.' That's normal — we haven't
> started anything. Clicking into the Training Config page opens the
> form where you'd actually launch a run. That's Video 9."

**Click back to Data.**

> "Back to where we started."

## Section 5 — Expand A Raw Row + Wrap (5:30–6:30)

**On screen**: click `[data-testid="expand-doc-20"]`.

> "One more thing before we wrap. Each raw document on the Data tab
> is one ticket from the source CSV. Expand a row…"

**Beat** as the row expands.

> "…and you see the question and the agent's answer. This is the
> data the SFT loop will fine-tune the model against. The model's
> job is to learn to write answers like this for questions it's
> never seen."

**Pause** ~2 seconds.

## Wrap (6:30–6:50)

> "We're done. In about five minutes we went from a fresh install to
> a seeded project with 20 raw rows, 200 gold rows, and
> ready-to-train splits. The next video walks the dataset lifecycle
> in detail — cleaning, gold, synthetic — and Videos 5, 6, and 7
> walk each of the three official samples through their full pipeline."

---

## Optional deeper technical notes (for advanced viewers; cut for beginner cut)

- The seeder is at `backend/app/services/demo_project_service.py:221`
  (`seed_demo_project`). It uses a deterministic 70/15/15 split with
  a minimum 2-row val/test floor.
- The auth flow is local-API-key only by default; SSO/OIDC requires
  setting `OIDC_*` env vars in `backend/.env`.
- The frontend Vite dev server (port 5173) proxies `/api` to the
  FastAPI backend on port 8000 — see `frontend/vite.config.ts`.

## Narration rules (apply through every video)

- If a stage is seeded, say it's *seeded*.
- If a stage is *running*, say it's running (and call out runtime
  requirements like teacher model or Celery worker).
- If a stage is *conceptual*, say it's conceptual.
- Never confuse cleaning-time PII redaction (regex) with the PII
  Detector sample (span_set model task).
- Never claim a trained model exists if no experiment has completed.

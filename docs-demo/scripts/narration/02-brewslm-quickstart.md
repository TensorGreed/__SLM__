# BrewSLM Quickstart — Narration

Status: **rewritten 2026-10-18 for the Epic U workspace** (Overview /
Data steps / Train form — no pipeline tabs, Coach bar, Dataset Prep,
Tokenization or Training Config any more). Audio regenerated the same day
by `tts/generate_narration.py` (Orpheus voice "leo", direct Ollama + SNAC
path); the video is re-recorded after the U7 visual pass lands.

The **Python script** in [tts/generate_narration.py](../../../tts/generate_narration.py)
is the **authoritative source** of the spoken text — that's what gets
passed to the TTS engine. This file mirrors the same text plus the stage
directions / Playwright cues that aren't speakable. If you edit narration,
edit the Python first and copy the new text here in the same commit.

Total runtime: **2:37** (`tts/audio/v02-durations.json`; the
Playwright spec `tests/demo-recordings/02-brewslm-quickstart-narrated.spec.ts`
pads each section to these lengths).

---

## Section 1 — Cold open (0:00–0:12)

**On screen**: login form at `/login`. Playwright holds the page for the
length of the section before filling anything.

> "We're taking a fresh local BrewSLM install and getting from nothing to
> a fully seeded sample project, in under five minutes. No training, no
> data prep — just log in, click, and look around."

## Section 2 — Login (0:12–0:24)

**On screen**: filling username + password, click Sign in, landing on the
project list.

> "BrewSLM auth is on by default. I'll log in as admin — the bootstrap
> user. The password is the local development token from the backend env
> file. And we're in."

**Action**: type `admin`, type `sk-mock-admin-key`, click **Sign in**.

> *Note: not just any username works here.* Local login auto-creates new
> usernames as engineer role with no project memberships, which 403s on
> the seeded sample project. Recordings must use `admin`. Not spoken.

## Section 3 — Project list + starter tiles (0:24–0:45)

**On screen**: project list page; the "Starter projects" tile strip
(`.demo-project-tiles`) spotlighted; the Support FAQ tile hovered.

> "This is the project list. Up top, three starter projects: Support FAQ,
> PII Detector, and Sentiment Classifier. Each is backed by a manifest, a
> source file, and a two-hundred-row answer key. Clicking a tile seeds a
> complete project — raw data imported, the answer key imported, and
> train, validation, and test splits already written."

## Section 4 — Seed Support FAQ → the Overview (0:45–1:09)

**On screen**: tile click → `/project/<id>` (`[data-testid="overview-page"]`).
Spotlight beats: "Where you are" (`overview-stage`) → "Next step"
(`overview-next`) → click **Full plan** (`overview-plan-toggle`) and
spotlight the five-stage list.

> "I'll click Support FAQ. The backend copies the source file into the
> project, creates twenty raw documents, imports two hundred answer-key
> rows, and pre-writes a sixteen, two, two split. We land on the project
> Overview: where you are, the next step, and under Full plan, the five
> stages — bring data in, prepare, train, check the results, ship.
> Nothing is trained yet, but the data work is done."

## Section 5 — Data · Sources, expand a document (1:09–1:27)

**On screen**: sidebar **Data** (`nav-data`) → `/project/<id>/data`;
spotlight the five-step bar (`.stage-page__steps`); open the Details
drawer's **Documents** section (`data-details-section-documents`); click
the first `[data-testid^="expand-doc-"]`; spotlight the expanded row.

> "The sidebar has six pages. Data first. It's five steps: Sources,
> Prepare, Labels, Generate, Split. Sources holds the twenty raw documents
> — one per support ticket. Expand one and you see the shape: a question
> and the agent's answer. That's what the model will learn to write."

## Section 6 — Data · Prepare (1:27–1:43)

**On screen**: click step **Prepare** (`stage-step-cleaning`); spotlight
the cleaning form.

> "Prepare — chunk text, redact personal information, score quality.
> Nothing is running here; the support FAQ corpus is already small and
> clean. On a messy corpus this is where the work would happen."

## Section 7 — Data · Labels, the answer key (1:43–1:55)

**On screen**: click step **Labels** (`stage-step-goldset`); open the
Details section **Entries and human review** (`data-details-section-entries`);
spotlight the entries header (`gold-entries-header`, 200 rows, locked).

> "Labels. The answer key: two hundred rows, locked. Every run gets
> scored against it — it is never trained on. This step also confirms
> the task shape, so training and scoring agree on what a row is."

## Section 8 — Data · Split (1:55–2:06)

**On screen**: click step **Split** (`stage-step-dataprep`); spotlight the
Train / Val / Test split card (16 / 2 / 2 already written).

> "Split. The adapter is applied — question and answer pair — so each row
> has the shape the trainer expects. The splits are already written:
> sixteen train, two validation, two test."

## Section 9 — Train, the form (2:06–2:23)

**On screen**: sidebar **Train** (`nav-train`) → `/project/<id>/train`,
`[data-testid="train-form"]`. Spotlight beats in narration order: base
model + Browse → task shape + Change → the Seeds radio → the readiness
line → the Train button → the **More options** toggle (not opened).

> "The Train page is one form. Base model, with Browse. The task shape,
> read-only, with Change. One run or three seeds. A run name. Below it,
> the readiness line tells you whether the project can train right now,
> and one Train button starts it. Everything else lives under More
> options."

## Section 10 — Wrap (2:23–2:37)

**On screen**: sidebar **Overview** (`nav-overview`); spotlight the hero
(where you are / next step).

> "We won't press Train in this video. From a fresh install to a seeded
> project with twenty raw documents, a two-hundred-row answer key, and
> ready-to-train splits. The next video walks every page of the
> workspace in detail."

---

## Optional deeper technical notes (background for the recorder; not spoken)

- The seeder is at
  [backend/app/services/demo_project_service.py](../../../backend/app/services/demo_project_service.py)
  (`seed_demo_project`). Deterministic 70/15/15 split with a minimum
  2-row val/test floor.
- Auth is local-API-key only by default; SSO/OIDC requires setting
  `OIDC_*` env vars in `backend/.env`.
- The Vite dev server (port 5173) proxies `/api` to FastAPI on 8000
  — see [frontend/vite.config.ts](../../../frontend/vite.config.ts).
- Workspace map: `frontend/src/utils/workspaceRoutes.ts` (six
  destinations; Data steps `?step=data|cleaning|goldset|synthetic|dataprep`
  labelled Sources / Prepare / Labels / Generate / Split).

## Narration rules (apply through every video)

- If a stage is seeded, say it's *seeded*.
- If a stage is *running*, say it's running (and call out runtime
  requirements like teacher model or Celery worker).
- If a stage is *conceptual*, say it's conceptual.
- Never confuse cleaning-time PII redaction (regex) with the PII
  Detector sample (span_set model task).
- Never claim a trained model exists if no experiment has completed.
- UI vocabulary: *answer key* (not gold set), *test examples*, *built-in
  checks*, *custom checks*, *pass/fail rules*; "the Train page", never
  "tab".
- **Don't read literal tech tokens** — API keys, env var names, REST
  paths, file extensions like `JSONL`. TTS engines mispronounce them
  and the on-screen action shows them anyway. Use natural-language
  descriptions ("the local development token", "the bootstrap
  user").

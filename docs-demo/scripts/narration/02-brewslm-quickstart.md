# BrewSLM Quickstart — Narration

Status: **synced** with the actual narrated take produced by
`tts/generate_narration.py` (Orpheus voice "leo") on 2026-05-20.

The **Python script** in [tts/generate_narration.py](../../../tts/generate_narration.py)
is the **authoritative source** of the spoken text — that's what
gets passed to the TTS engine. This file mirrors the same text plus
the stage directions / Playwright cues that aren't speakable. If you
edit narration, edit the Python first and copy the new text here in
the same commit.

Total runtime: **2:10** (matches the muxed
`docs-demo/recordings/raw/02-brewslm-quickstart-narrated.mp4`).
Section timings below come from
`tts/audio/v02-durations.json`.

---

## Section 1 — Cold open (0:00–0:11)

**On screen**: login form at `/login`. Playwright pauses ~11s before
filling anything.

> "We're taking a fresh local BrewSLM install and getting from
> nothing to a fully-seeded demo project, in under five minutes. No
> training, no data prep — just login, click, and inspect."

## Section 2 — Login (0:11–0:21)

**On screen**: filling username + password, click Sign in, landing
on the project list.

> "BrewSLM auth is on by default. I'll log in as admin — the
> bootstrap user. The password is the local development token from
> the backend env file. And we're in."

**Action**: type `admin`, type `sk-mock-admin-key`, click **Sign in**.

> *Note: not just any username works here.* Local login auto-creates
> new usernames as engineer role with no project memberships, which
> 403s on the seeded demo project. Recordings must use `admin`.
> Detail captured in the recording plan; not spoken in the video to
> keep the cold open tight.

## Section 3 — Project list + demo tiles (0:21–0:41)

**On screen**: project list page; demo tile strip near the top with
three tiles.

> "This is the project list. Up top, three official demo tiles:
> Support FAQ, PII Detector, and Sentiment Classifier. Each is
> backed by a manifest, a source file, and a two-hundred-row gold
> set. Clicking a tile seeds a complete project — raw data imported,
> gold imported, and train, validation, and test splits already
> written."

## Section 4 — Seed Support FAQ (0:41–1:07)

**On screen**: tile click → page navigation → land on Data tab. The
pipeline status badge shows training stage at 60% complete because
the seed pre-loaded the upstream work.

> "I'll click Support FAQ. Behind the scenes, the backend copies the
> source file into the project's raw data, creates twenty raw
> documents, imports two hundred gold rows, and pre-writes a sixteen,
> two, two train, validation, test split. A few seconds later we
> land on the Data tab. Notice the pipeline status badge — already
> at training stage, sixty percent complete. The seed did the
> upstream work for us. We haven't trained anything yet, but we have
> a ready-to-train project."

## Section 5 — Cleaning tab (1:07–1:20)

**On screen**: click the **Cleaning** tab. "Cleaning Configuration"
heading visible.

> "Let's walk the pipeline tabs. Ten of them, left to right, in the
> order you'd touch them. Cleaning — chunk text, redact personal
> information, score quality. Nothing's running here. The support
> FAQ corpus is already small and clean."

## Section 6 — Gold Set tab (1:20–1:28)

**On screen**: click **Gold Set** tab. "Entries 200" badge visible.

> "Gold Set. Two hundred entries, locked. The evaluation ground
> truth. The manifest says six, but the file has two hundred. The
> file wins."

## Section 7 — Dataset Prep tab (1:28–1:39)

**On screen**: click **Dataset Prep** tab. Adapter and field-mapping
panels visible.

> "Dataset Prep. The adapter is applied — question and answer pair —
> turning each row into a question and a matching answer. Splits are
> already written: sixteen train, two validation, two test."

## Section 8 — Training tab (1:39–1:45)

**On screen**: click **Training** tab. "No experiments yet" empty
state.

> "Training tab. No experiments yet. We haven't started anything.
> Launching a run is Video Nine."

## Section 9 — Expand a raw row + wrap (1:45–2:10)

**On screen**: click back to **Data**, click an `expand-doc-*`
element on a raw row, raw question/answer text expands.

> "Back to the Data tab. One more thing before we wrap. Each raw
> document is one support ticket from the source file. Expand a row
> and you see the question and the agent's answer. This is the data
> we'll fine-tune the model against. The model learns to write
> answers like this for questions it's never seen. Done. From a
> fresh install to a seeded project with twenty raw rows, two
> hundred gold rows, and ready-to-train splits. Next video walks the
> dataset lifecycle in detail."

---

## Optional deeper technical notes (background for the recorder; not spoken)

- The seeder is at
  [backend/app/services/demo_project_service.py:221](../../../backend/app/services/demo_project_service.py)
  (`seed_demo_project`). Deterministic 70/15/15 split with a minimum
  2-row val/test floor.
- Auth is local-API-key only by default; SSO/OIDC requires setting
  `OIDC_*` env vars in `backend/.env`.
- The Vite dev server (port 5173) proxies `/api` to FastAPI on 8000
  — see [frontend/vite.config.ts](../../../frontend/vite.config.ts).

## Narration rules (apply through every video)

- If a stage is seeded, say it's *seeded*.
- If a stage is *running*, say it's running (and call out runtime
  requirements like teacher model or Celery worker).
- If a stage is *conceptual*, say it's conceptual.
- Never confuse cleaning-time PII redaction (regex) with the PII
  Detector sample (span_set model task).
- Never claim a trained model exists if no experiment has completed.
- **Don't read literal tech tokens** — API keys, env var names, REST
  paths, file extensions like `JSONL`. TTS engines mispronounce them
  and the on-screen action shows them anyway. Use natural-language
  descriptions ("the local development token", "the bootstrap
  user").

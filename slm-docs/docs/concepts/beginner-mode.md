---
sidebar_position: 4
title: Beginner mode
---

# Beginner mode

BrewSLM has a per-project **beginner mode** flag. Since 2026-10-17 it **hides nothing**: the sidebar always shows the same six destinations (Overview, Data, Train, Results, Ship, Settings) and every page is reachable. The flag only decides whether *advanced sections* inside a page, start collapsed (Results has none). In the Settings page it is labelled **Start advanced sections collapsed**.

Nothing in the backend changes and every endpoint stays callable; it is purely a UX default.

**It's on by default for every new project**, whichever way the project was created: **+ New Project**, a starter project, a manifest without an explicit setting, or the API. Experts switch it off per project.

## One way in, one Coach

- **Starting a project.** Use **+ New Project**: describe what the model should do, and BrewSLM sets up the task type and base model. Alternatively, start from a starter project on the same page. An existing `brewslm.yaml` can be imported from the create dialog's **Advanced** section. Every path lands on the project's **Overview**.
- **The Coach bar.** It sits at the top of the **Data**, **Train**, **Results** and **Ship** pages (it is hidden on Overview and Settings), and it's the one guidance surface. It shows:
  - where you are (stage and % done);
  - the single **next step**, with **Continue →**;
  - a short tip and a walkthrough video for the page you're on;
  - rule-based suggestions for that stage, with one-click fixes:
    - **Training** on a documents-only project: one click turns the documents into a Q&A assistant (generated question→answer pairs, an answer key, a split, a training run and the lift check).
    - **Data** → **Generate**: rows still waiting for review, a training mix that is mostly synthetic, or no task type chosen yet.
    - **Data** → **Split**: data not split yet, a split older than the data it was cut from, a test split too small to trust, or overlapping splits.
    - **Ship**: nothing trained yet, or the model you're about to ship lost to its base model, was never evaluated, or isn't clearly different from the base model (the change on the test examples is within noise). It also flags a newer run that beat its base model but hasn't been exported.
- **The full plan.** The Coach bar's **Overview** button (and the sidebar's **Overview**) opens the project landing page: the current stage, the one next step as a button, the latest run and its verdict, the shipped model, up to three suggestions ("Worth a look"), and a collapsed **Full plan** with the step checklist and the one-click Quickstart actions.
- **Switching it off.** **Settings** → **Show inline tips** turns the Coach on or off for a project. It defaults to on.

## What the flag changes

Nothing is hidden or locked. Pages that used to be "advanced only" now live under **Settings**, out of the main path:

| Where | What |
|---|---|
| **Settings** | Autopilot Planner (plan diffs, repair preview, rollback), Workflow Builder, Pipeline presets, Domain Packs, Domain Profiles, Adapter Studio, Extension Studio, manifest export; Pipeline as Code links into Ship |
| **Results** | Nothing. All four tabs (Verdict, Failures, Compare, Try it) look the same with the flag on or off |
| Other advanced sections | Start collapsed when the flag is on |

All of these are also reachable by URL and from the Cmd-K palette, which lists the six destinations plus every tool page.

## Why the advanced pages sit under Settings

They each represent a power-user concept a first-time ML engineer doesn't need yet:

- **Adapter Studio / Extension Studio** — assume you understand the data adapter / runtime / pack plugin contracts.
- **Workflow Builder / Pipeline presets** — assume you've already run a few experiments and want to template them.
- **Pipeline as Code** (Ship → Package → Options) — assumes you're ready to code-review your project as YAML.
- **Domain Packs / Profiles** — assume you understand the domain overlay concept. The default `general-pack-v1` is fine until you outgrow it.

## Toggling beginner mode

### UI

Open **Settings** and toggle **Start advanced sections collapsed**. The setting persists on the project.

### CLI

```sh
# Turn beginner mode on
brewslm project beginner --id 7 --enable

# Turn it off
brewslm project beginner --id 7 --disable
```

### API

```sh
curl -X PUT http://localhost:8000/api/projects/7 \
  -H "Content-Type: application/json" \
  -d '{"beginner_mode": false}'
```

## Inviting collaborators

When a new teammate joins your project, they inherit whatever `beginner_mode` setting the project has. Most teams switch a shared project off beginner mode once everyone is up to speed.

## Next

- [Architecture](architecture.md) — the system-level mental model.
- [Quickstart](../getting-started/quickstart.md) — start with beginner mode on; leave it on for as long as it helps.

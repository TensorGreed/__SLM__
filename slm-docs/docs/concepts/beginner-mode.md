---
sidebar_position: 4
title: Beginner mode
---

# Beginner mode

BrewSLM ships with a deliberate **beginner mode**, so a new ML engineer never sees a concept they haven't been taught yet. It's a per-project flag that hides advanced surfaces in the UI. Nothing in the backend changes, and every endpoint stays callable; it's purely a UX layer.

**It's on by default for every new project**, whichever way the project was created: **+ New Project**, a starter project, a manifest without an explicit setting, or the API. Experts switch it off per project.

## One way in, one Coach

- **Starting a project.** Use **+ New Project**: describe what the model should do, and BrewSLM sets up the task type and base model. Alternatively, start from a starter project on the same page. An existing `brewslm.yaml` can be imported from the create dialog's **Advanced** section. Every path lands in the project workspace.
- **The Coach bar.** It sits at the top of every workspace page, and it's the one guidance surface. It shows:
  - where you are (stage and % done);
  - the single **next step**, with **Continue →**;
  - a short tip and a walkthrough video for the tab you're on;
  - rule-based suggestions for that stage, with one-click fixes. Every tab a beginner walks through has them:
    - **Data**, **Cleaning**, **Answer Key**, **Training** and **Eval**.
    - **Training** on a documents-only project: one click turns the documents into a Q&A assistant (generated question→answer pairs, an answer key, a split, a training run and the lift check).
    - **Synthetic**: rows still waiting for review, a training mix that is mostly synthetic, or no task type chosen yet.
    - **Dataset Prep**: data not split yet, a split older than the data it was cut from, a test split too small to trust, or overlapping splits.
    - **Export**: nothing trained yet, or the model you're about to ship lost to its base model, was never evaluated, or isn't clearly different from the base model (the change on the test examples is within noise). It also flags a newer run that beat its base model but hasn't been exported.
- **The full plan.** **Full plan** opens the project home: the checklist with Lab Journal stamps, one-click Quickstart actions, and the beginner-mode switch.
- **Switching it off.** The 🧭 toggle in the top bar turns the Coach off for a project. It defaults to on for beginner projects.

## What's hidden

| Surface | Hidden in beginner mode? |
|---|---|
| Pipeline (data → export) | ✓ Always visible |
| Training Configurations | ✓ Always visible |
| Base Model Registry | ✓ Always visible |
| Autopilot (guided: goal → data → safe plan → train → chat) | ✓ Always visible |
| Playground | ✓ Always visible |
| Deployments | ✓ Always visible |
| Observability | ✓ Always visible |
| **Autopilot Planner** (plan diffs, repair preview, rollback) | Hidden |
| **Adapter Studio** | Hidden |
| **Extension Studio** | Hidden |
| **Workflow Builder** | Hidden |
| **Pipeline presets** | Hidden |
| **Pipeline as Code (manifest)** | Hidden |
| **Domain Packs** | Hidden |
| **Domain Profiles** | Hidden |

Hidden surfaces are still **reachable directly via URL** (e.g., `/project/7/extensions`) and via the **Cmd-K palette filter** — beginner mode hides their links from the sidebar, not from the app.

## Why these specifically

These four classes of "hidden" surface each represent a power-user concept that a first-time ML engineer doesn't need to learn yet:

- **Adapter Studio / Extension Studio** — assume you understand the data adapter / runtime / pack plugin contracts. Without that, the UI is overwhelming.
- **Workflow Builder / Pipeline presets** — assume you've already run a few experiments and want to template them. Premature for a first project.
- **Pipeline as Code** — assumes you're ready to code-review your project as YAML. Useful once a project stabilises, distracting before then.
- **Domain Packs / Profiles** — assumes you understand the domain overlay concept. The default `general-pack-v1` is fine until you outgrow it.

## Toggling beginner mode

### UI

In the sidebar's footer, click **Enter beginner mode** (when off) or **Leave beginner mode** (when on). A confirm dialog explains what changes. The setting persists on the project.

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

When a new teammate joins your project, they inherit whatever `beginner_mode` setting the project has. Most teams toggle a single shared project off beginner mode once everyone is up to speed.

## Cmd-K still respects beginner mode

The command palette's action list filters by `beginnerMode` too. So if you've collapsed the sidebar AND turned on beginner mode, the Adapter Studio / Extension Studio / Workflow Builder actions all disappear from Cmd-K results until you switch off beginner mode.

## Next

- [Architecture](architecture.md) — the system-level mental model.
- [Quickstart](../getting-started/quickstart.md) — start with beginner mode on; leave it on for as long as it helps.

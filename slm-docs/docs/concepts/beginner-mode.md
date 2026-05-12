---
sidebar_position: 4
title: Beginner mode
---

# Beginner mode

BrewSLM ships with a deliberate **beginner mode** so a new ML engineer never sees a concept they haven't been taught yet. It's a per-project boolean flag that hides advanced surfaces in the UI. Nothing about the backend changes — every endpoint stays callable. It's purely a UX layer.

## What's hidden

| Surface | Hidden in beginner mode? |
|---|---|
| Pipeline (data → export) | ✓ Always visible |
| Training Configurations | ✓ Always visible |
| Base Model Registry | ✓ Always visible |
| Autopilot Planner | ✓ Always visible |
| Playground | ✓ Always visible |
| Deployments | ✓ Always visible |
| Observability | ✓ Always visible |
| Guided Setup (wizard) | ✓ Always visible |
| **Adapter Studio** | Hidden |
| **Extension Studio** | Hidden |
| **Workflow Builder** | Hidden |
| **Recipes** | Hidden |
| **Pipeline as Code (manifest)** | Hidden |
| **Domain Packs** | Hidden |
| **Domain Profiles** | Hidden |

Hidden surfaces are still **reachable directly via URL** (e.g., `/project/7/extensions`) and via the **Cmd-K palette filter** — beginner mode hides their links from the sidebar, not from the app.

## Why these specifically

These four classes of "hidden" surface each represent a power-user concept that a first-time ML engineer doesn't need to learn yet:

- **Adapter Studio / Extension Studio** — assume you understand the data adapter / runtime / pack plugin contracts. Without that, the UI is overwhelming.
- **Workflow Builder / Recipes** — assume you've already run a few experiments and want to template them. Premature for a first project.
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

---
sidebar_position: 4
title: Extension Studio (UI)
---

# Extension Studio

A workspace page that wraps the three extension operations — list, [scaffold](scaffold.md), [validate + reload](validate-and-reload.md) — into one surface. **Hidden by default in [beginner mode](../concepts/beginner-mode.md)** because the four plugin contracts assume you already understand the platform.

## Where it lives

`/project/:id/extensions`. Sidebar entry under the Training rail (Puzzle icon). Beginner mode hides this entry from the sidebar; the URL still works if you navigate directly or use Cmd-K.

## Layout

```
┌────────────────────────┬───────────────────────────────────────┐
│  Plugin kinds          │  Detail card                           │
│  ────────────────────  │  ────────────────────────────────────  │
│  ▣ Data adapter        │  Data adapter                          │
│    slm.data_adapter/v3 │  Contract version  slm.data_adapter/v3 │
│    [11] registered     │  Settings key      DATA_ADAPTER_…      │
│    [reload badge]      │  Recognised hooks  register_data_…     │
│  ─                     │  Loaded modules    my.adapter          │
│  ▢ Training runtime    │                                        │
│    slm.training_…/v1   │  ─── Generate scaffold ────────────────│
│    [1] registered      │  [form: kind=adapter, plugin id, …]    │
│  ─                     │  [scaffold preview with file tabs]     │
│  ▢ Domain pack         │                                        │
│    (loader pending)    │  ─── Validate module ──────────────────│
│  ─                     │  [module path input, force-reload]     │
│  ▢ Eval pack           │  [validate report with pass/fail rows] │
│                        │                                        │
│  [Reload all]          │                                        │
└────────────────────────┴───────────────────────────────────────┘
```

## Pick a kind

Click any pill in the left column. Each pill shows:

- **Name** (Data adapter / Training runtime / Domain pack / Eval pack).
- **Contract version** (`slm.<kind>/vN`).
- **Registered count** (how many plugins of this kind the runtime has right now).
- **Module count** (how many modules contributed).
- **Error count** (red badge if any module failed to load).
- **Reload badge** (set after a reload call, shows `ok` / `partial` / `not_supported`).

Selected kind highlights with a black outline + light gray fill. The right pane updates to match.

## Detail card

Right of the kind list, top: the per-kind status block.

- **Contract version + settings key** — what to write in `.env` to wire a plugin of this kind.
- **Recognised hooks** — the hook function / constant names the loader looks for. If you forget one, the validator's `module_interface` check fails and tells you exactly which names were tried.
- **Loaded modules** — paths the loader has imported.
- **Load errors** — per-module error string, if any. Red badge inline.
- **Reload kind** button — re-imports this kind's modules. Disabled for `domain_pack` / `eval_pack` until their loaders ship.

For declarative kinds (domain / eval pack), the card also shows the *"Module loader for this kind is planned for P38"* note.

## Generate scaffold

The form mirrors the [scaffold CLI](scaffold.md):

- **Plugin id** — required, kebab-case recommended.
- **Display name** — optional, defaults to title-cased plugin id.
- **Description** — lands in the module docstring + README.
- **Author**, **Version** — optional.
- **Write files to DATA_DIR** — toggle. Default on. Off = preview only, no disk write.

Click **Generate scaffold**. The preview surface fills below the form:

- **Header strip** — green `scaffold ready` badge, plugin id, contract version, output dir, "n file(s) written".
- **File tabs** — switch between the generated files (module, test stub, README).
- **Code pane** — read-only view of the active file.
- **Download** — per-file blob download. **Download all files** dumps every file via separate Blob downloads (no zip dependency in the frontend).

## Validate

Type a Python module path that's importable from where the backend runs (the backend's `PYTHONPATH`). Tick **Force reload** if the module is already in `sys.modules`.

Click **Validate**. The report appears below:

- **Top row** — green `contract ok` or red `contract failed`, the module name, contract version, declared version.
- **Declared ids row** — for kinds that register specific ids (data adapter, domain pack, eval pack), the ids the module would register.
- **Checks list** — five rows (one per [validator](contracts.md)). Each row: pass/fail badge, check name, message.

The whole report mirrors the API response, so the UI is just a presentation layer over `POST /api/extensions/validate`.

## Reload

Two paths:

- **Reload kind** in the detail card → reloads just the selected kind.
- **Reload all** at the top of the kind list → reloads every kind that has a live loader (data adapter, training runtime). Declarative kinds report `not_supported` in the response.

After a reload, the per-kind reload badges in the left column update. If any kind comes back `partial` or `error`, the badge is red.

## What this UI is *not*

- **Not a plugin marketplace.** It generates scaffolds + validates them; finding / installing third-party plugins is a separate concern (pip install + add to settings).
- **Not a runtime monitor.** Once a plugin is registered, you observe it via the [Run Timeline](../observability/timeline.md) — the Extension Studio is a build-time surface.
- **Not in beginner mode.** Beginners shouldn't be wiring plugin contracts; the Cmd-K palette filters this action out too.

## Next

- [Contracts](contracts.md) — what every validator checks for.
- [Scaffold](scaffold.md) — the same flow on the CLI.
- [Validate + reload](validate-and-reload.md) — the API surface in detail.
- [CLI](cli.md) — `brewslm scaffold` + `brewslm extensions`.

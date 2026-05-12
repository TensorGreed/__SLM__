---
slug: /
sidebar_position: 1
title: BrewSLM
---

# BrewSLM

**BrewSLM is a local-first platform for building, evaluating, and deploying domain-specific Small Language Models (SLMs) end-to-end.** It bundles data ingestion, dataset curation, fine-tuning, evaluation gating, compression, export, and deployment behind one UI, one CLI (`brewslm`), and one HTTP API. Every workflow is reproducible by design — every artifact links back to the manifest that produced it.

This guide is written for a **new ML engineer** picking up BrewSLM for the first time. You should be comfortable with Python, the terminal, and basic ML concepts (tokenization, fine-tuning, eval metrics). You do **not** need to have shipped a production LLM before.

## Three ways to drive everything

Every feature in BrewSLM exposes the same three surfaces:

| Surface | When to reach for it |
|---|---|
| **Web UI** | Exploring, picking defaults, watching live runs, hand-curating eval sets. |
| **`brewslm` CLI** | Scripting, automation, CI gates, batch operations. |
| **HTTP API** | Embedding in your own tools, building integrations, scheduling. |

Most pages in this guide show all three for the same operation. Pick whichever fits your task — the underlying state is the same.

## Where to start

1. **[Quickstart](getting-started/quickstart.md)** — 10 minutes from `git clone` to a trained model.
2. **[Build your first project](getting-started/first-project.md)** — narrated walk through the pipeline.
3. **[Concepts → Architecture](concepts/architecture.md)** — the mental model behind projects, artifacts, manifests, and runs.
4. **[Pipeline workflows](workflows/pipeline-overview.md)** — deep dives per stage (ingestion → export).

## What's new in 2026

The platform has grown well beyond pipeline-as-a-website. Recent waves added:

- **Deployment Assistant** — versioning, smoke tests, rollback, post-deploy telemetry, drift checks, deployability score. See the [Deployment](deployment/plan.md) section.
- **Unified Observability** — every stage emits canonical `RunEvent` rows; the [timeline](observability/timeline.md), [failure clusters](observability/failure-clusters.md), and [support bundles](observability/support-bundles.md) all read from this one log.
- **Extension Studio** — formal Protocol contracts for the four plugin kinds (data adapter, training runtime, domain pack, eval pack), with a [scaffold generator](extensions/scaffold.md), [validator + reloader](extensions/validate-and-reload.md), and a hidden-in-beginner-mode [UI](extensions/extension-studio.md).
- **Cmd-K command palette** — fast nav across all workspace pages from any page.

## Core principles

- **Measured over guessed.** If a number can come from real observation (a completed run, a smoke test, a telemetry window), it does — labelled with `provenance: "measured"`. Estimates carry `provenance: "estimated"` so you can tell them apart.
- **Reason codes everywhere.** Every error / critical event carries a stable `reason_code` from a single taxonomy. Failure clustering, support bundles, and audit logs all pivot on it.
- **Reproducibility by manifest.** Every training run gets an immutable manifest that's enough to re-run it. Pipelines export to / apply from `brewslm.yaml`.
- **Beginner mode is real.** Advanced surfaces (Workflow Builder, Domain Packs, Adapter Studio, Extension Studio) are hidden by default for new users and can be revealed any time.

## Documentation surface

| Section | Read it when |
|---|---|
| [Getting Started](getting-started/quickstart.md) | First-time setup + first project. |
| [Concepts](concepts/architecture.md) | You want the mental model before the buttons. |
| [Setup](setup/install.md) | Installing, auth, environment variables. |
| [Pipeline workflows](workflows/pipeline-overview.md) | Per-stage UI/CLI/API for ingestion → export. |
| [Deployment](deployment/plan.md) | Promoting, telemetry, drift, rollback. |
| [Observability](observability/run-events.md) | Timeline, clusters, support bundles. |
| [Extensions](extensions/contracts.md) | Writing custom data adapters, runtimes, packs. |
| [Reliability](reliability/measured-vs-estimated.md) | What "measured" means + common blockers. |
| [Reference](reference/cli.md) | CLI commands, API endpoints, settings, glossary. |

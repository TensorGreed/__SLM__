---
sidebar_position: 6
title: Newbie Autopilot
---

# Newbie Autopilot

Newbie Autopilot is designed to reduce manual setup decisions for first-time users.

## What It Does

Autopilot orchestrates:

1. readiness checks,
2. preflight validation,
3. compatibility checks,
4. safe auto-repairs for common blockers.

## Typical Auto-Repairs

- adapter/task defaults,
- model fallback within safe profile,
- target fallback when original target is incompatible,
- conservative config tuning.

## Dry Run vs Run

- **Dry run**: shows proposed changes and blockers without executing training.
- **Run**: applies safe changes and proceeds when feasible.

## Strict Mode Behavior

In strict mode, autopilot does not hide fallback paths. If a repair is not safe, it stops and returns explicit blockers.

## Decision Log

Autopilot should return:

- what changed automatically,
- why it changed,
- what remains blocked,
- how to fix remaining blockers.

Treat the decision log as your runbook for reproducibility and learning.

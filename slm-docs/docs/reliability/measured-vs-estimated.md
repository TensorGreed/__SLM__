---
sidebar_position: 9
title: Measured vs Estimated vs Simulated
---

# Measured vs Estimated vs Simulated

This distinction is critical for trustworthy decision-making.

## Definitions

- **Measured**: obtained from real benchmark execution.
- **Estimated**: computed fallback when direct measurement is unavailable.
- **Simulated**: synthetic path used only when explicitly allowed by runtime policy.

## How to Read UI Labels

In optimization cards and wizard estimates, check provenance badges before treating a number as ground truth.

- Measured values are suitable for ranking and deployment decisions.
- Estimated values are directional and should be validated.
- Simulated values should never be mistaken for production-ready evidence.

## Strict Mode Behavior

In strict mode:

- unsupported measurement paths do not silently fallback,
- you get explicit blockers and remediation guidance.

## Remediation Hints When You See Estimated Values

1. Verify runtime dependencies and hardware access.
2. Ensure candidate artifact exists and is readable.
3. Run benchmark matrix on intended target profile.
4. Retry optimization after dependency repair.

Treat provenance as part of model quality, not an optional detail.

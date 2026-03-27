---
sidebar_position: 6
title: Training and Compatibility
---

# Training and Compatibility

Training is where most new users hit blockers. BrewSLM is designed to block unsafe or incompatible runs early.

## Recommended First Training Run

1. Choose a starter pack (optional but useful).
2. Keep base model in 1B to 7B range.
3. Use default optimizer/scheduler settings.
4. Run one short baseline epoch.
5. Evaluate before scaling up.

## Compatibility Checks

BrewSLM checks model-target compatibility before training starts.

Examples:

- VRAM over target baseline -> hard blocker
- Unknown capabilities -> warning or blocker depending on strictness
- Missing runtime dependencies -> actionable failure

## Strict Mode vs Fallback Mode

- **Strict mode**: no hidden fallback behavior; blockers remain blockers.
- **Standard mode**: safe fallback paths may be used when possible, with explicit provenance.

## Choosing a Target Profile

Pick target profile based on deployment intent:

- `mobile_cpu` for on-device CPU
- `edge_gpu` for local/edge GPU
- `vllm_server` for server throughput
- `browser_webgpu` for browser inference

Do not optimize for every target on day one. Start with one.

## Troubleshooting Training Blockers

When blocked, collect:

- blocker reason text,
- target profile,
- model name,
- estimated memory footprint,
- runtime backend.

Then either reduce model size, change target profile, or tune batch/precision strategy.

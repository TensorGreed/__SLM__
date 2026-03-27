---
sidebar_position: 8
title: Export and Deployment
---

# Export and Deployment

After evaluation passes, export your model for the runtime you care about.

## Typical Export Targets

- Hugging Face format
- GGUF / llama.cpp
- ONNX
- TensorRT
- Server runners (`vLLM`, `TGI`, `Ollama`)

## Optimization Recommendations

BrewSLM can rank export candidates by latency, memory, and quality tradeoff.

Always check metric provenance:

- **measured**: benchmarked on real candidate/runtime path
- **estimated**: fallback estimate with reason
- **mixed**: partial measured + estimated metrics

## Mobile Bundles

For iOS/Android export paths, BrewSLM can produce runnable reference bundles with:

- deterministic structure,
- setup README,
- smoke-checkable entrypoints.

## Pre-Deployment Checklist

1. Verify artifact integrity.
2. Confirm runtime compatibility.
3. Review safety/policy gate results.
4. Run smoke inference.
5. Capture manifest + benchmark evidence.

## Final Advice

Deploy the smallest model that meets your quality bar. Smaller models are easier to host, cheaper to run, and faster to iterate.
